"""Shared pieces of the evaluation commands: indexing, querying, timing, output."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .bm25 import BM25, InvertedIndex
from .datasets import Dataset
from .text import Analyzer


def tuned_bm25(name: str, reports: Path = Path("reports/phase1")) -> tuple[float, float]:
    """The k1 and b Phase 1 chose for a dataset on its development queries.

    The one reader of reports/phase1/tuning_<dataset>.json. A missing report is
    an error, not a fallback: comparing against default BM25 would quietly
    weaken the baseline and flatter every method measured against it.
    """
    path = reports / f"tuning_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `uv run python -m memret.tune --dataset {name}` first")
    best = json.loads(path.read_text())["best"]
    return float(best["k1"]), float(best["b"])


def index_corpus(docs: list[str], analyzer: Analyzer) -> tuple[InvertedIndex, dict[str, float]]:
    start = time.perf_counter()
    analyzed = [analyzer(doc) for doc in docs]
    middle = time.perf_counter()
    index = InvertedIndex(analyzed)
    end = time.perf_counter()
    return index, {"analyze_s": middle - start, "index_s": end - middle}


def run_queries(bm25: BM25, analyzer: Analyzer, dataset: Dataset, *, k: int = 1000,
                drop_query_id: bool = True) -> tuple[dict[str, list[str]], np.ndarray]:
    """Search every judged query. Returns the run and each query's latency in milliseconds.

    drop_query_id removes a document whose id equals the query's id. BEIR and
    Anserini both do this, because in some BEIR datasets the queries are also
    in the corpus and would trivially retrieve themselves. In FiQA the ids are
    unrelated numbers that sometimes collide, so it matters there too.
    """
    run: dict[str, list[str]] = {}
    latencies = []
    for query_id in sorted(dataset.queries):
        start = time.perf_counter()
        hits = bm25.search(analyzer(dataset.queries[query_id]), k + 1 if drop_query_id else k)
        ranked = [dataset.doc_ids[h.doc] for h in hits]
        if drop_query_id:
            ranked = [d for d in ranked if d != query_id]
        run[query_id] = ranked[:k]
        latencies.append((time.perf_counter() - start) * 1000)
    return run, np.array(latencies)


def latency_summary(ms: np.ndarray) -> dict[str, float]:
    return {"n": int(len(ms)), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "max_ms": float(ms.max())}


def machine() -> dict[str, str | int]:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "cpus": os.cpu_count() or 0}


def rounded(obj, digits: int = 6):
    """Round floats so reruns differ only where results really differ (timings), not in float noise."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rounded(obj), indent=2) + "\n")


def write_trec_run(path: Path, run: dict[str, list[str]], tag: str) -> None:
    """TREC run format, readable by trec_eval: query Q0 doc rank score tag (score = -rank here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for query_id, docs in run.items():
            for rank, doc in enumerate(docs, start=1):
                f.write(f"{query_id} Q0 {doc} {rank} {-rank} {tag}\n")
