"""Score dense retrieval on BEIR and compare it with BM25 and the model's published numbers.

    uv run python -m memret.evaluate_dense --dataset all                  # whole documents, truncated at 512 tokens
    uv run python -m memret.evaluate_dense --dataset all --mode chunks    # long documents split, best chunk wins
    uv run python -m memret.evaluate_dense --dataset all --embed-only     # just fill the embedding cache

Embeddings are cached under data/cache/embeddings, keyed by model and exact
text, so only the first run pays for them. Writes
reports/phase2/dense_<mode>_<dataset>.json, and reports/phase2/dense.md when
run on all datasets in both modes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .cache import EmbeddingCache
from .chunking import chunk_corpus
from .datasets import DEFAULT_ROOT, DEV_SPLIT, SOURCES, Dataset, load
from .dense import ExactIndex
from .embed import BGE_SMALL, Embedder
from .experiment import index_corpus, latency_summary, machine, run_queries, tuned_bm25, write_json, write_trec_run
from .metrics import evaluate
from .stats import compare
from .text import DEFAULT

CACHE_ROOT = Path("data/cache/embeddings")
CHUNK_MAX_TOKENS, CHUNK_OVERLAP = 510, 64

# bge-small-en-v1.5's own model card (MTEB, test split), at the pinned revision.
PUBLISHED_URL = f"https://huggingface.co/BAAI/bge-small-en-v1.5/blob/{BGE_SMALL.revision}/README.md"
PUBLISHED = {
    "scifact": {"nDCG@10": 0.71275, "R@100": 0.94167},
    "nfcorpus": {"nDCG@10": 0.34305, "R@100": 0.31135},
    "fiqa": {"nDCG@10": 0.40343, "R@100": 0.69552},
}


def passages(dataset: Dataset, mode: str, embedder: Embedder) -> tuple[list[str], np.ndarray]:
    """The texts to embed and, for each, the position of its document."""
    if mode == "document":
        return list(dataset.docs), np.arange(len(dataset.docs))
    chunks = chunk_corpus(dataset.docs, embedder.token_spans, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP)
    return [c.text for c in chunks], np.array([c.doc for c in chunks])


def embed_cached(texts: list[str], embedder: Embedder, cache: EmbeddingCache) -> tuple[np.ndarray, dict]:
    """Vectors for texts, with what the cache saved and how long it took."""
    cache.reset_counters()
    start = time.perf_counter()
    vectors = cache.get_many(texts, embedder.encode)
    seconds = time.perf_counter() - start
    stats = cache.stats()
    return vectors, {"texts": len(texts), "hits": stats.hits, "misses": stats.misses,
                     "hit_rate": stats.hit_rate, "seconds": seconds,
                     "texts_per_s": len(texts) / seconds if seconds else None}


def dense_run(index: ExactIndex, chunk_doc: np.ndarray, query_vectors: np.ndarray, dataset: Dataset,
              query_ids: list[str], *, k: int = 1000, drop_query_id: bool = True) -> tuple[dict, np.ndarray]:
    """Rank documents for every query. Same conventions as BM25: drop a hit whose id is the query's."""
    run, latencies = {}, []
    for q, vec in zip(query_ids, query_vectors):
        start = time.perf_counter()
        ids, _ = index.search_documents(vec[None, :], chunk_doc, k + 1 if drop_query_id else k)
        ranked = [dataset.doc_ids[d] for d in ids[0] if d >= 0]
        if drop_query_id:
            ranked = [d for d in ranked if d != q]
        run[q] = ranked[:k]
        latencies.append((time.perf_counter() - start) * 1000)
    return run, np.array(latencies)


def evaluate_dataset(name: str, *, mode: str = "document", split: str = "test", root: Path = DEFAULT_ROOT,
                     cache_root: Path = CACHE_ROOT, embed_only: bool = False, runs_dir: Path | None = None,
                     shard_size: int = 1024) -> dict:
    dataset = load(name, split, root)
    embedder = Embedder(BGE_SMALL)
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root, shard_size=shard_size)
    texts, chunk_doc = passages(dataset, mode, embedder)
    passage_vectors, passage_embed = embed_cached(texts, embedder, cache)
    query_ids = sorted(dataset.queries)
    query_vectors, query_embed = embed_cached([embedder.query_text(dataset.queries[q]) for q in query_ids],
                                              embedder, cache)
    result = {"dataset": name, "split": split, "mode": mode, "model": asdict(BGE_SMALL),
              "chunking": {"max_tokens": CHUNK_MAX_TOKENS, "overlap": CHUNK_OVERLAP} if mode == "chunks" else None,
              "corpus": {"docs": len(dataset.docs), "passages": len(texts)},
              "embedding": {"passages": passage_embed, "queries": query_embed}}
    if embed_only:
        return result

    index = ExactIndex(passage_vectors)
    run, latencies = dense_run(index, chunk_doc, query_vectors, dataset, query_ids)
    dense_scores = evaluate(run, dataset.qrels)

    k1, b = tuned_bm25(name)
    bm25_index, _ = index_corpus(dataset.docs, DEFAULT)
    bm25_run, _ = run_queries(BM25(bm25_index, k1=k1, b=b), DEFAULT, dataset)
    bm25_scores = evaluate(bm25_run, dataset.qrels)
    assert dense_scores.query_ids == bm25_scores.query_ids
    vs_bm25 = {metric: asdict(compare(dense_scores.per_query[metric], bm25_scores.per_query[metric]))
               for metric in ("nDCG@10", "R@100")}
    dense_better = dense_scores.per_query["nDCG@10"] > bm25_scores.per_query["nDCG@10"]
    bm25_better = dense_scores.per_query["nDCG@10"] < bm25_scores.per_query["nDCG@10"]

    if runs_dir is not None:
        write_trec_run(runs_dir / f"{name}-{split}-dense-{mode}.trec", run, f"bge-small-{mode}")
    result.update({
        "queries": len(dense_scores.query_ids),
        "metrics": dense_scores.mean(),
        "bm25_params": {"k1": k1, "b": b},
        "bm25_metrics": bm25_scores.mean(),
        "dense_vs_bm25": vs_bm25,
        "queries_dense_wins_nDCG@10": int(dense_better.sum()),
        "queries_bm25_wins_nDCG@10": int(bm25_better.sum()),
        "published": PUBLISHED[name] if split == "test" else None,
        "search_ms": latency_summary(latencies),
        "machine": machine(),
    })
    return result


def choose_chunking(name: str, *, root: Path = DEFAULT_ROOT, cache_root: Path = CACHE_ROOT) -> dict:
    """Chunks against whole documents on development queries: the comparison that decides the default."""
    dataset = load(name, DEV_SPLIT[name], root)
    embedder = Embedder(BGE_SMALL)
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root)
    query_ids = sorted(dataset.queries)
    queries = cache.get_many([embedder.query_text(dataset.queries[q]) for q in query_ids], embedder.encode)
    per_mode = {}
    for mode in ("document", "chunks"):
        texts, chunk_doc = passages(dataset, mode, embedder)
        run, _ = dense_run(ExactIndex(cache.get_many(texts, embedder.encode)), chunk_doc, queries, dataset, query_ids)
        per_mode[mode] = evaluate(run, dataset.qrels).per_query["nDCG@10"]
    return {"dataset": name, "split": DEV_SPLIT[name], "queries": len(per_mode["document"]),
            "chunks_vs_document_nDCG@10": asdict(compare(per_mode["chunks"], per_mode["document"]))}


def summary_markdown(results: dict[tuple[str, str], dict], chunking: list[dict] | None = None) -> str:
    names = [n for n in SOURCES if (n, "document") in results]
    params = {n: results[(n, "document")].get("bm25_params", {"k1": 0.9, "b": 0.4}) for n in names}
    bm25_note = "; ".join(f"{n} {params[n]['k1']} and {params[n]['b']}" for n in names)
    lines = [
        "# Phase 2: dense retrieval with bge-small-en-v1.5",
        "",
        f"Test queries. Model `{BGE_SMALL.name}` at revision `{BGE_SMALL.revision[:12]}`, [CLS] pooling, "
        "unit-length vectors, exact inner-product search. **Document** embeds each document whole, truncated "
        f"at 512 tokens, as the model's published evaluation does. **Chunks** splits documents longer than "
        f"{CHUNK_MAX_TOKENS} tokens into overlapping windows ({CHUNK_OVERLAP}-token overlap) and scores a document "
        f"by its best chunk. **BM25** uses the k1 and b each dataset got from Phase 1's tuning ({bm25_note}). "
        f"**Published** is the model card's MTEB result ([source]({PUBLISHED_URL})).",
        "",
        "| Dataset | Queries | Metric | BM25 | Dense, document | Dense, chunks | Published |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        doc, chk = results[(n, "document")], results.get((n, "chunks"))
        for metric in ("nDCG@10", "R@100", "R@1000", "MRR@10"):
            pub = doc["published"].get(metric) if doc["published"] else None
            lines.append(f"| {n} | {doc['queries']} | {metric} | {doc['bm25_metrics'][metric]:.4f} "
                         f"| {doc['metrics'][metric]:.4f} | {chk['metrics'][metric]:.4f} | "
                         f"{'' if pub is None else f'{pub:.4f}'} |" if chk else
                         f"| {n} | {doc['queries']} | {metric} | {doc['bm25_metrics'][metric]:.4f} "
                         f"| {doc['metrics'][metric]:.4f} | | {'' if pub is None else f'{pub:.4f}'} |")
    lines += ["", "## Dense against BM25", "",
              "Paired over the same test queries; 95% bootstrap interval and two-sided randomization p.", "",
              "| Dataset | Mode | nDCG@10 difference | 95% interval | p | Queries dense wins | Queries BM25 wins |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        for mode in ("document", "chunks"):
            r = results.get((n, mode))
            if not r:
                continue
            c = r["dense_vs_bm25"]["nDCG@10"]
            lines.append(f"| {n} | {mode} | {c['difference']:+.4f} | {c['ci_low']:+.4f} to {c['ci_high']:+.4f} "
                         f"| {c['p_value']:.3g} | {r['queries_dense_wins_nDCG@10']} | {r['queries_bm25_wins_nDCG@10']} |")
    lines += ["", "## Corpus and search cost", "",
              "| Dataset | Documents | Passages, chunks mode | Exact search p50 | Exact search p95 |",
              "| --- | --- | --- | --- | --- |"]
    for n in names:
        doc, chk = results[(n, "document")], results.get((n, "chunks"))
        lines.append(f"| {n} | {doc['corpus']['docs']:,} | {chk['corpus']['passages']:,} "
                     f"| {doc['search_ms']['p50_ms']:.2f} ms | {doc['search_ms']['p95_ms']:.2f} ms |" if chk else
                     f"| {n} | {doc['corpus']['docs']:,} | | {doc['search_ms']['p50_ms']:.2f} ms "
                     f"| {doc['search_ms']['p95_ms']:.2f} ms |")
    if chunking:
        lines += ["", "## Chunking, decided on development queries", "",
                  "Chunks minus whole documents, nDCG@10, paired over each dataset's development queries. "
                  "The default mode is chosen here, before looking at the test columns above.", "",
                  "| Dataset | Split | Queries | Difference | 95% interval | p |", "| --- | --- | --- | --- | --- | --- |"]
        for c in chunking:
            d = c["chunks_vs_document_nDCG@10"]
            lines.append(f"| {c['dataset']} | {c['split']} | {c['queries']} | {d['difference']:+.4f} "
                         f"| {d['ci_low']:+.4f} to {d['ci_high']:+.4f} | {d['p_value']:.3g} |")
    mach = results[(names[0], "document")]["machine"]
    lines += ["", f"Search timings: {mach['platform']}, Python {mach['python']}, NumPy {mach['numpy']}, "
                  f"{mach['cpus']} CPUs, one query at a time with NumPy's default threads; single-threaded latency and "
                  "approximate search are in [ann.md](ann.md). Embedding cost and cache behaviour are in "
                  "[cache.md](cache.md).", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", choices=[*SOURCES, "all"])
    parser.add_argument("--mode", default="document", choices=["document", "chunks", "both"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-only", action="store_true", help="fill the embedding cache and stop")
    parser.add_argument("--with-dev", action="store_true", help="with --embed-only, also embed development queries")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase2"))
    parser.add_argument("--choose-chunking", action="store_true",
                        help="compare chunks with whole documents on development queries and stop")
    parser.add_argument("--shard-size", type=int, default=1024,
                        help="vectors per cache file; a killed run loses at most one shard's work")
    args = parser.parse_args()

    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    if args.choose_chunking:
        for name in names:
            c = choose_chunking(name, root=args.data_dir, cache_root=args.cache_dir)
            write_json(args.out / f"chunking_{name}.json", c)
            d = c["chunks_vs_document_nDCG@10"]
            print(f"{name:9} {c['split']:5} chunks - document nDCG@10 {d['difference']:+.4f} "
                  f"({d['ci_low']:+.4f} to {d['ci_high']:+.4f}, p = {d['p_value']:.3g})", flush=True)
        return
    modes = ["document", "chunks"] if args.mode == "both" else [args.mode]
    results = {}
    for name in names:
        for mode in modes:
            splits = [args.split] + ([DEV_SPLIT[name]] if args.embed_only and args.with_dev else [])
            for split in splits:
                r = evaluate_dataset(name, mode=mode, split=split, root=args.data_dir, cache_root=args.cache_dir,
                                     embed_only=args.embed_only, runs_dir=Path("runs"), shard_size=args.shard_size)
                e = r["embedding"]["passages"]
                print(f"{name:9} {mode:8} {split:5} passages {e['texts']:,} (cache hits {e['hits']:,}, "
                      f"computed {e['misses']:,} in {e['seconds']:.0f} s)", flush=True)
                if args.embed_only:
                    continue
                m = r["metrics"]
                print(f"{'':9} nDCG@10 {m['nDCG@10']:.4f} (BM25 {r['bm25_metrics']['nDCG@10']:.4f}, published "
                      f"{PUBLISHED[name]['nDCG@10']:.4f})  R@100 {m['R@100']:.4f}  R@1000 {m['R@1000']:.4f}",
                      flush=True)
                if split == "test":
                    write_json(args.out / f"dense_{mode}_{name}.json", r)
    if not args.embed_only and args.split == "test":
        # The summary covers whatever test results exist for every dataset, from this run or an earlier one.
        for name in SOURCES:
            for mode in ("document", "chunks"):
                path = args.out / f"dense_{mode}_{name}.json"
                if path.exists():
                    results[(name, mode)] = json.loads(path.read_text())
        chunking = [json.loads(p.read_text()) for n in SOURCES if (p := args.out / f"chunking_{n}.json").exists()]
        if all((n, "document") in results for n in SOURCES):
            (args.out / "dense.md").write_text(summary_markdown(results, chunking))


if __name__ == "__main__":
    main()
