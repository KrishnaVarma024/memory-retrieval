"""BEIR benchmark data: pinned downloads, checksum checks, and loading.

Every file comes from the BeIR organisation on the Hugging Face Hub at a fixed
commit, and its SHA-256 is checked before use, so a rerun next year reads the
same bytes. The pinned commits are the last ones that still ship the original
JSON Lines files; later commits converted the datasets to Parquet.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUB = "https://huggingface.co/datasets/BeIR"
DEFAULT_ROOT = Path("data/beir")


@dataclass(frozen=True)
class Source:
    corpus_commit: str
    qrels_commit: str
    sha256: dict[str, str]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(
            f.removeprefix("qrels/").removesuffix(".tsv") for f in self.sha256 if f.startswith("qrels/")
        )


SOURCES: dict[str, Source] = {
    "scifact": Source(
        corpus_commit="984eed826375f18d27936c4a32bf0f8491e3f414",
        qrels_commit="2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        sha256={
            "corpus.jsonl.gz": "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e",
            "queries.jsonl.gz": "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3",
            "qrels/train.tsv": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
            "qrels/test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
        },
    ),
    "nfcorpus": Source(
        corpus_commit="e28763e68d85db0fa71652ba3c4afabf8c5b3bb7",
        qrels_commit="a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
        sha256={
            "corpus.jsonl.gz": "ae4d6ab2ab88d880e7e180776c956a09c19c43a703457742cd850d2422035802",
            "queries.jsonl.gz": "baba3504a2736b0e32265962f04acc4fecfaf9649e0ea08e49ff441246f75364",
            "qrels/train.tsv": "6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a",
            "qrels/dev.tsv": "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8",
            "qrels/test.tsv": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
        },
    ),
    "fiqa": Source(
        corpus_commit="e31855201132ea2a257d7df77c828d7c02427521",
        qrels_commit="252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
        sha256={
            "corpus.jsonl.gz": "6d2c7e27aeb8c646791178e3e5591a373813642f38d9c992c0a4f6cbf4092f72",
            "queries.jsonl.gz": "a683bbedd2fd3b1fcb1a44fbd706daf9aff27291cd99e7a086b3cf7624785acb",
            "qrels/train.tsv": "e46a99529aa61b12e086a5c057e4e0ecaeda9502e53d5fe855efce778f1ec59a",
            "qrels/dev.tsv": "03b27e547cd29dc721c3b93ce8ddd70df6b2d0ce954d8ef3659d386e8cd72c11",
            "qrels/test.tsv": "6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44",
        },
    ),
}

# Datasets held out from every choice made so far: nothing is tuned or chosen on
# them, and they have test queries only. Kept apart from SOURCES so that commands
# looping over SOURCES never touch them. The Hub converted BEIR's files to Parquet
# in April 2026; the pinned 2022 commit still has the JSONL files.
HELD_OUT: dict[str, Source] = {
    "arguana": Source(
        corpus_commit="900330faa8ca2828a468dfe795f9d3c3887c8cfc",
        qrels_commit="ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7",
        sha256={
            "corpus.jsonl.gz": "6572c13e17aa51d42c0498e765c70cfc9f15550a2140213ea267767e3a9f9134",
            "queries.jsonl.gz": "faae9bb245700b764c29d98a1972d1b10ebb8dfb4e7bb349741e8d4f20d29b13",
            "qrels/test.tsv": "0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76",
        },
    ),
}

# The split used for choosing settings. SciFact has no dev split, so its
# training queries play that role. Test queries are never used to choose anything.
DEV_SPLIT = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}


@dataclass(frozen=True)
class Dataset:
    name: str
    split: str
    doc_ids: list[str]
    docs: list[str]  # title and text joined, as Anserini's "flat" BEIR indexes do
    queries: dict[str, str]  # only the queries judged in this split
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: graded relevance}


def source(name: str) -> Source:
    if name in SOURCES:
        return SOURCES[name]
    if name in HELD_OUT:
        return HELD_OUT[name]
    raise ValueError(f"unknown dataset {name!r}; choose from {sorted(SOURCES) + sorted(HELD_OUT)}")


def _url(name: str, src: Source, file: str) -> str:
    if file.startswith("qrels/"):
        return f"{HUB}/{name}-qrels/resolve/{src.qrels_commit}/{file.removeprefix('qrels/')}"
    return f"{HUB}/{name}/resolve/{src.corpus_commit}/{file}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Fetch the files of dataset `name` that are missing or corrupt; return its folder."""
    src = source(name)
    folder = root / name
    for file, expected in src.sha256.items():
        path = folder / file
        if path.exists() and sha256_of(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename only after the checksum matches, so
        # an interrupted or corrupted download never sits under the real name.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with urllib.request.urlopen(_url(name, src, file), timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            actual = sha256_of(tmp)
            if actual != expected:
                raise ValueError(f"{name}/{file}: SHA-256 is {actual}, expected {expected}")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return folder


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        header = next(rows, None)
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for line, row in enumerate(rows, start=2):
            if len(row) != 3:
                raise ValueError(f"{path}:{line}: expected 3 columns, got {len(row)}")
            query_id, doc_id, score = row
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


def load_folder(folder: Path, name: str, split: str) -> Dataset:
    """Load a dataset in BEIR's layout: corpus.jsonl.gz, queries.jsonl.gz, qrels/<split>.tsv."""
    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")

    doc_ids: list[str] = []
    docs: list[str] = []
    seen: set[str] = set()
    with gzip.open(folder / "corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc_id = record["_id"]
            if doc_id in seen:
                raise ValueError(f"{name}: duplicate document id {doc_id!r}")
            seen.add(doc_id)
            doc_ids.append(doc_id)
            docs.append(f"{record.get('title') or ''} {record.get('text') or ''}".strip())

    queries: dict[str, str] = {}
    with gzip.open(folder / "queries.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["_id"] in qrels:
                queries[record["_id"]] = record["text"]
    missing = qrels.keys() - queries.keys()
    if missing:
        raise ValueError(f"{name}/{split}: {len(missing)} judged queries have no text, e.g. {min(missing)!r}")
    return Dataset(name, split, doc_ids, docs, queries, qrels)


def load(name: str, split: str, root: Path = DEFAULT_ROOT) -> Dataset:
    src = source(name)
    if split not in src.splits:
        raise ValueError(f"{name} has splits {src.splits}, not {split!r}")
    return load_folder(download(name, root), name, split)
