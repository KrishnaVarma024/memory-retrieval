import gzip
import hashlib
import io
import json

import pytest

from memret import datasets
from memret.datasets import DEV_SPLIT, HELD_OUT, SOURCES, Source, load_folder, read_qrels


def write_beir(folder, corpus, queries, qrels_text, split="test"):
    (folder / "qrels").mkdir(parents=True)
    with gzip.open(folder / "corpus.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in corpus)
    with gzip.open(folder / "queries.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(q) + "\n" for q in queries)
    (folder / "qrels" / f"{split}.tsv").write_text(qrels_text)


CORPUS = [{"_id": "d1", "title": "Cats", "text": "Cats purr."}, {"_id": "d2", "title": "", "text": "Dogs bark."}]
QUERIES = [{"_id": "q1", "text": "do cats purr"}, {"_id": "q2", "text": "unjudged query"}]
QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n"


def test_loads_beir_layout(tmp_path):
    write_beir(tmp_path, CORPUS, QUERIES, QRELS)
    ds = load_folder(tmp_path, "tiny", "test")
    assert ds.doc_ids == ["d1", "d2"]
    assert ds.docs == ["Cats Cats purr.", "Dogs bark."]  # title joined to text; empty title adds nothing
    assert ds.queries == {"q1": "do cats purr"}  # only judged queries
    assert ds.qrels == {"q1": {"d1": 1, "d2": 0}}


def test_duplicate_document_ids_are_rejected(tmp_path):
    write_beir(tmp_path, CORPUS + [CORPUS[0]], QUERIES, QRELS)
    with pytest.raises(ValueError, match="duplicate"):
        load_folder(tmp_path, "tiny", "test")


def test_judged_query_without_text_is_rejected(tmp_path):
    write_beir(tmp_path, CORPUS, QUERIES[1:], QRELS)
    with pytest.raises(ValueError, match="no text"):
        load_folder(tmp_path, "tiny", "test")


def test_qrels_header_and_columns_are_checked(tmp_path):
    bad_header = tmp_path / "a.tsv"
    bad_header.write_text("qid\tdocid\trel\nq1\td1\t1\n")
    with pytest.raises(ValueError, match="header"):
        read_qrels(bad_header)
    bad_row = tmp_path / "b.tsv"
    bad_row.write_text("query-id\tcorpus-id\tscore\nq1\td1\n")
    with pytest.raises(ValueError, match="3 columns"):
        read_qrels(bad_row)


def test_every_dataset_is_pinned_and_has_its_splits():
    for name, src in SOURCES.items():
        assert len(src.corpus_commit) == len(src.qrels_commit) == 40
        assert all(len(d) == 64 for d in src.sha256.values())
        assert "test" in src.splits and DEV_SPLIT[name] in src.splits


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_verifies_checksums_and_never_leaves_partial_files(tmp_path, monkeypatch):
    payload = b"hello"
    good = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(SOURCES, "tiny", Source("0" * 40, "0" * 40, {"corpus.jsonl.gz": good}))
    monkeypatch.setattr(datasets.urllib.request, "urlopen", lambda url, timeout: FakeResponse(payload))
    folder = datasets.download("tiny", tmp_path)
    assert (folder / "corpus.jsonl.gz").read_bytes() == payload

    monkeypatch.setitem(SOURCES, "tiny", Source("0" * 40, "0" * 40, {"queries.jsonl.gz": "f" * 64}))
    with pytest.raises(ValueError, match="SHA-256"):
        datasets.download("tiny", tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == ["corpus.jsonl.gz"]  # nothing half-written


def test_unknown_dataset_names_the_choices():
    with pytest.raises(ValueError, match="scifact"):
        datasets.source("msmarco")


def test_held_out_datasets_are_pinned_and_kept_apart():
    assert not HELD_OUT.keys() & SOURCES.keys()  # commands that loop over SOURCES must never reach them
    for name, src in HELD_OUT.items():
        assert len(src.corpus_commit) == len(src.qrels_commit) == 40
        assert all(len(d) == 64 for d in src.sha256.values())
        assert src.splits == ("test",)
        assert datasets.source(name) is src
