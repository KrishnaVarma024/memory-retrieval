"""Exact against approximate search: the recall-versus-latency curve, and where exact stops being fast enough.

    uv run python -m memret.ann --dataset fiqa

Uses the cached FiQA embeddings from `memret.evaluate_dense` (whole documents,
57,638 vectors). Everything runs on one thread, one query at a time, because
that is how a search service answers a single request.

1. Exact search (ours, NumPy) and FAISS's exact index: latency.
2. FAISS HNSW, swept over ef_search: recall of the exact top 10 and top 100,
   nDCG@10 against the judgements, and latency.
3. Scaling: exact and HNSW latency as the collection grows to 400,000 vectors,
   padded with perturbed copies of the real ones, to find where exact search
   breaks the Phase 0 budget of 50 ms at the 95th percentile.

Writes reports/phase2/ann_<dataset>.json and reports/phase2/ann.md.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before NumPy loads its BLAS: one thread, like one request

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .cache import EmbeddingCache  # noqa: E402
from .datasets import DEFAULT_ROOT, SOURCES, load  # noqa: E402
from .dense import ExactIndex, HNSWIndex  # noqa: E402
from .embed import BGE_SMALL, Embedder  # noqa: E402
from .evaluate_dense import CACHE_ROOT  # noqa: E402
from .experiment import machine, write_json  # noqa: E402
from .metrics import evaluate  # noqa: E402

EF_SEARCH = (10, 20, 40, 80, 160, 320, 640)
HNSW_M, HNSW_EF_CONSTRUCTION = 32, 200
SIZES = (57_638, 100_000, 200_000, 400_000)
BUDGET_P95_MS = 50.0


def percentiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


def timed(search, queries: np.ndarray, *, repeats: int = 1) -> tuple[list, list[float]]:
    """Run search one query at a time; return the results and every call's latency in ms.

    A warm-up pass over the first 20 queries goes untimed, so one-off costs (page faults on a freshly
    built array, lazy initialisation) don't land in the percentiles. With repeats > 1 every query is
    timed that many times, which steadies the 95th percentile on a shared machine.
    """
    for q in queries[:20]:
        search(q[None, :])
    out, ms = [], []
    for rep in range(repeats):
        for q in queries:
            start = time.perf_counter()
            res = search(q[None, :])
            ms.append((time.perf_counter() - start) * 1000)
            if rep == 0:
                out.append(res)
    return out, ms


def recall(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Share of the exact top k that the approximate top k also found, averaged over queries."""
    return float(np.mean([len(set(a[:k]) & set(e[:k])) / k for a, e in zip(approx, exact)]))


def ndcg10(ids: np.ndarray, dataset, query_ids: list[str]) -> float:
    run = {}
    for q, row in zip(query_ids, ids):
        ranked = [dataset.doc_ids[d] for d in row if d >= 0]
        run[q] = [d for d in ranked if d != q][:10]
    return evaluate(run, dataset.qrels).mean()["nDCG@10"]


def padded(vectors: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """vectors followed by perturbed unit-length copies of random rows, up to size rows."""
    out = np.empty((size, vectors.shape[1]), dtype=np.float32)
    out[:len(vectors)] = vectors
    rng = np.random.default_rng(seed)
    for s in range(len(vectors), size, 50_000):
        e = min(s + 50_000, size)
        block = vectors[rng.integers(0, len(vectors), e - s)] + rng.normal(0, 0.03, (e - s, vectors.shape[1]))
        out[s:e] = block / np.linalg.norm(block, axis=1, keepdims=True)
    return out


def benchmark(name: str, *, root: Path = DEFAULT_ROOT, cache_root: Path = CACHE_ROOT) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)
    dataset = load(name, "test", root)
    embedder = Embedder(BGE_SMALL)
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root)
    vectors = cache.get_many(list(dataset.docs), _refuse)
    query_ids = sorted(dataset.queries)
    queries = cache.get_many([embedder.query_text(dataset.queries[q]) for q in query_ids], _refuse)

    exact = ExactIndex(vectors)
    exact_res, exact_ms = timed(lambda q: exact.search(q, 101)[0][0], queries)
    exact_ids = np.array(exact_res)
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    flat_res, flat_ms = timed(lambda q: flat.search(q, 101)[1][0], queries)
    flat_ids = np.array(flat_res)
    result = {"dataset": name, "vectors": len(vectors), "queries": len(queries), "dim": int(vectors.shape[1]),
              "exact": {"ours": percentiles(exact_ms), "faiss_flat": percentiles(flat_ms),
                        "nDCG@10": ndcg10(exact_ids, dataset, query_ids),
                        "faiss_flat_nDCG@10": ndcg10(flat_ids, dataset, query_ids),
                        # The two differ only in rounding of near-equal scores and in how exact ties are broken.
                        "queries_with_identical_top101": int((exact_ids == flat_ids).all(axis=1).sum())}}

    start = time.perf_counter()
    hnsw = HNSWIndex(vectors, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)
    build_s = time.perf_counter() - start
    sweep = []
    for ef in EF_SEARCH:
        # Top 11, so dropping a hit whose id equals the query's still leaves 10 for nDCG@10.
        res, ms = timed(lambda q: hnsw.search(q, 11, ef_search=ef)[0][0], queries)
        ids = np.array(res)
        row = {"ef_search": ef, "recall@10": recall(ids, exact_ids, 10), "nDCG@10": ndcg10(ids, dataset, query_ids),
               **percentiles(ms)}
        if ef >= 100:  # a top-100 search explores at least 100 candidates whatever ef_search says
            wide, _ = timed(lambda q: hnsw.search(q, 100, ef_search=ef)[0][0], queries)
            row["recall@100"] = recall(np.array(wide), exact_ids, 100)
        sweep.append(row)
    result["hnsw"] = {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "build_s": build_s,
                      "bytes_per_vector_graph": HNSW_M * 2 * 4, "sweep": sweep}
    result["machine"] = machine()
    return result


def scaling(name: str, target_ef: int, *, root: Path = DEFAULT_ROOT, cache_root: Path = CACHE_ROOT,
            sizes: tuple[int, ...] = SIZES, scaling_queries: int = 200) -> dict:
    """Exact and HNSW latency as the collection grows. Exact needs no build; HNSW is built on every core
    here, since only its search is timed."""
    import faiss

    faiss.omp_set_num_threads(1)
    dataset = load(name, "test", root)
    embedder = Embedder(BGE_SMALL)
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root)
    vectors = cache.get_many(list(dataset.docs), _refuse)
    query_ids = sorted(dataset.queries)
    queries = cache.get_many([embedder.query_text(dataset.queries[q]) for q in query_ids], _refuse)
    sample = queries[:scaling_queries]
    rows = []
    for size in sizes:
        big = padded(vectors, size)
        big_exact = ExactIndex(big)
        exact_top, ms_exact = timed(lambda q: big_exact.search(q, 10)[0][0], sample, repeats=3)
        row = {"vectors": size, "exact": percentiles(ms_exact)}
        if size <= 200_000:
            start = time.perf_counter()
            big_hnsw = HNSWIndex(big, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, threads=os.cpu_count() or 1)
            faiss.omp_set_num_threads(1)
            row["hnsw_build_s"] = time.perf_counter() - start
            approx, ms_hnsw = timed(lambda q: big_hnsw.search(q, 10, ef_search=target_ef)[0][0], sample, repeats=3)
            row["hnsw"] = {"ef_search": target_ef, "recall@10": recall(np.array(approx), np.array(exact_top), 10),
                           **percentiles(ms_hnsw)}
            del big_hnsw
        rows.append(row)
        print(f"{size:>9,} vectors: exact p95 {row['exact']['p95_ms']:.1f} ms"
              + (f", HNSW p95 {row['hnsw']['p95_ms']:.2f} ms" if "hnsw" in row else ""), flush=True)
        del big, big_exact
    sizes_ms = np.array([[r["vectors"], r["exact"]["p95_ms"]] for r in rows])
    slope, intercept = np.polyfit(sizes_ms[:, 0], sizes_ms[:, 1], 1)
    return {"dataset": name, "padding": "perturbed copies of real vectors (Gaussian noise, sd 0.03 per dimension, "
                                        "renormalised)", "queries": len(sample), "repeats": 3, "rows": rows,
            "exact_p95_ms_per_100k": slope * 100_000, "exact_p95_intercept_ms": intercept,
            "exact_breaks_budget_at": (BUDGET_P95_MS - intercept) / slope if slope > 0 else None,
            "budget_p95_ms": BUDGET_P95_MS, "machine": machine()}


def _refuse(texts):
    raise RuntimeError(f"{len(texts)} texts are not in the embedding cache; run memret.evaluate_dense first")


def summary_markdown(r: dict, s: dict | None) -> str:
    h = r["hnsw"]
    lines = [
        f"# Phase 2: exact against approximate search on {r['dataset']}",
        "",
        f"{r['vectors']:,} document vectors ({r['dim']} dimensions), {r['queries']} test queries, one thread, one "
        "query at a time. Recall is the share of the exact top k that the approximate search also returns.",
        "",
        "## Exact search",
        "",
        "| Implementation | p50 | p95 | nDCG@10 |",
        "| --- | --- | --- | --- |",
        f"| Ours (NumPy) | {r['exact']['ours']['p50_ms']:.2f} ms | {r['exact']['ours']['p95_ms']:.2f} ms "
        f"| {r['exact']['nDCG@10']:.4f} |",
        f"| FAISS IndexFlatIP | {r['exact']['faiss_flat']['p50_ms']:.2f} ms | {r['exact']['faiss_flat']['p95_ms']:.2f} ms "
        f"| {r['exact']['faiss_flat_nDCG@10']:.4f} |",
        "",
        f"Both are exact. Their top 101 is identical for {r['exact']['queries_with_identical_top101']} of "
        f"{r['queries']} queries; the rest differ only where two scores are equal or within about a millionth, "
        "which the two implementations round and break differently.",
        "",
        f"## HNSW (M = {h['m']}, efConstruction = {h['ef_construction']}, built in {h['build_s']:.0f} s on one thread)",
        "",
        "Latency is for a top-10 search. Recall@100 needs a top-100 search, which explores at least 100 "
        "candidates whatever ef_search says, so it is shown only from ef_search 160.",
        "",
        "| ef_search | Recall@10 | Recall@100 | nDCG@10 | p50 | p95 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in h["sweep"]:
        r100 = f"{row['recall@100']:.4f}" if "recall@100" in row else ""
        lines.append(f"| {row['ef_search']} | {row['recall@10']:.4f} | {r100} | {row['nDCG@10']:.4f} "
                     f"| {row['p50_ms']:.3f} ms | {row['p95_ms']:.3f} ms |")
    if s is None:
        m = r["machine"]
        return "\n".join(lines + ["", f"Machine: {m['platform']}, Python {m['python']}, NumPy {m['numpy']}, "
                                         f"{m['cpus']} CPUs.", ""])
    lines += ["", "## As the collection grows", "",
              f"Padded with {s['padding']}. {s['queries']} queries per size, each timed {s['repeats']} times. HNSW uses the smallest ef_search that "
              "reached recall@10 of at least 0.95 above; it is measured up to 200,000 vectors.", "",
              "| Vectors | Exact p50 | Exact p95 | HNSW p95 | HNSW recall@10 | HNSW build (all cores) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in s["rows"]:
        hn = row.get("hnsw")
        lines.append(f"| {row['vectors']:,} | {row['exact']['p50_ms']:.1f} ms | {row['exact']['p95_ms']:.1f} ms | "
                     + (f"{hn['p95_ms']:.2f} ms | {hn['recall@10']:.4f} | {row['hnsw_build_s']:.0f} s |" if hn else
                        " | | |"))
    brk = s["exact_breaks_budget_at"]
    lines += ["", f"A straight-line fit gives exact search about {s['exact_p95_ms_per_100k']:.1f} ms of p95 latency per "
                  f"100,000 vectors, crossing the {s['budget_p95_ms']:.0f} ms budget at about "
                  + (f"{brk:,.0f} vectors." if brk else "no size (latency did not grow)."), ""]
    m = r["machine"]
    lines += [f"Machine: {m['platform']}, Python {m['python']}, NumPy {m['numpy']}, {m['cpus']} CPUs.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="fiqa", choices=list(SOURCES))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase2"))
    parser.add_argument("--part", default="all", choices=["all", "curve", "scaling"],
                        help="the recall-latency curve, the scaling run (which needs the curve), or both")
    args = parser.parse_args()
    curve_path = args.out / f"ann_{args.dataset}.json"
    scaling_path = args.out / f"ann_scaling_{args.dataset}.json"
    if args.part in ("all", "curve"):
        result = benchmark(args.dataset, root=args.data_dir, cache_root=args.cache_dir)
        write_json(curve_path, result)
        best = next((s for s in result["hnsw"]["sweep"] if s["recall@10"] >= 0.95), None)
        print(f"exact p95 {result['exact']['ours']['p95_ms']:.2f} ms; "
              + (f"HNSW reaches recall@10 {best['recall@10']:.3f} at ef_search {best['ef_search']}, "
                 f"p95 {best['p95_ms']:.3f} ms" if best else "HNSW never reached recall@10 0.95"), flush=True)
    result = json.loads(curve_path.read_text())
    if args.part in ("all", "scaling"):
        target = next((s["ef_search"] for s in result["hnsw"]["sweep"] if s["recall@10"] >= 0.95), EF_SEARCH[-1])
        write_json(scaling_path, scaling(args.dataset, target, root=args.data_dir, cache_root=args.cache_dir))
    s = json.loads(scaling_path.read_text()) if scaling_path.exists() else None
    (args.out / "ann.md").write_text(summary_markdown(result, s))


if __name__ == "__main__":
    main()
