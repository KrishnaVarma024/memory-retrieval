"""Fuse BM25 and dense retrieval, choose the fusion on development queries, test it once.

    uv run python -m memret.evaluate_hybrid --dataset all

For each dataset: build both retrievers (BM25 with the k1 and b Phase 1 chose;
bge-small vectors for whole documents from Phase 2's cache), collect every
query's candidates once, and score a grid of fusion settings on the
development queries. Two settings are chosen there, before any test query is
looked at: the best for each dataset, and one global setting that is best on
average over the datasets. The global one is "the hybrid", because it is what
the framework will ship: a new user's memories come with no judgements to tune
on. The test queries are then scored once, with paired comparisons against
each single retriever.

It also writes, per dataset, the ten test queries where the hybrid falls
furthest below the better single retriever, to be read rather than counted,
and a set of hard queries (the hybrid finds nothing relevant in its top 10)
that Phase 4's reranker will be measured on.

Writes reports/phase3/hybrid_<dataset>.json, losers_<dataset>.md,
hard_queries.json and, when run on all datasets, hybrid.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .cache import EmbeddingCache
from .datasets import DEFAULT_ROOT, DEV_SPLIT, SOURCES, Dataset, load
from .dense import ExactIndex
from .embed import BGE_SMALL, Embedder
from .evaluate_dense import CACHE_ROOT
from .experiment import index_corpus, latency_summary, machine, write_json, write_trec_run
from .fusion import normalise
from .hybrid import THEORETICAL_MIN, CandidateSet, FusionConfig, HybridSearcher, candidate_set, fuse
from .metrics import evaluate
from .stats import compare
from .text import DEFAULT

DEPTH = 1000  # candidates each retriever contributes for the test runs (R@1000 needs 1,000 results)
DEPTHS = (100, DEPTH)
RRF_K = (1, 5, 10, 20, 40, 60, 100, 200)
RRF_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)
ALPHAS = tuple(round(i * 0.05, 2) for i in range(21))
NORMS = ("minmax", "tmm", "zscore")
DEFAULT_RRF = FusionConfig("rrf", depth=DEPTH, k=60.0, weight=0.5)
LOSERS = 10


def fusion_grid() -> list[FusionConfig]:
    """Every setting tried on development queries. On a tie the earlier one wins, so RRF's defaults come first."""
    grid = [DEFAULT_RRF]
    grid += [FusionConfig("rrf", depth=d, k=float(k), weight=w)
             for d in DEPTHS for k in RRF_K for w in RRF_WEIGHTS if (d, k, w) != (DEPTH, 60, 0.5)]
    grid += [FusionConfig("convex", depth=d, weight=a, norm=n) for d in DEPTHS for n in NORMS for a in ALPHAS]
    return grid


def tuned_bm25_params(name: str, reports: Path = Path("reports/phase1")) -> tuple[float, float]:
    """Phase 1's tuned k1 and b. Fails loudly: silently comparing against default BM25 would flatter the hybrid."""
    path = reports / f"tuning_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `uv run python -m memret.tune --dataset {name}` first")
    best = json.loads(path.read_text())["best"]
    return best["k1"], best["b"]


@dataclass
class Prepared:
    """One split's queries with their candidates, at each depth in DEPTHS."""

    dataset: Dataset
    query_ids: list[str]
    query_vectors: np.ndarray
    by_depth: dict[int, list[CandidateSet]]


@dataclass
class Built:
    searcher: HybridSearcher
    bm25_params: dict[str, float]
    dev: Prepared
    test: Prepared


def prepare(dataset: Dataset, searcher: HybridSearcher, query_vectors: np.ndarray) -> Prepared:
    query_ids = sorted(dataset.queries)
    deepest = []
    for s in range(0, len(query_ids), 64):  # dense scores in blocks: 64 queries x 57,638 documents is 15 MB
        dense = searcher.dense_scores(query_vectors[s:s + 64])
        for j, q in enumerate(query_ids[s:s + 64]):
            bm25 = searcher.bm25.scores(searcher.analyzer(dataset.queries[q]))
            deepest.append(candidate_set(bm25, dense[j], max(DEPTHS)))
    by_depth = {d: deepest if d == max(DEPTHS) else [c.truncate(d) for c in deepest] for d in DEPTHS}
    return Prepared(dataset, query_ids, query_vectors, by_depth)


@functools.cache
def _embedder() -> Embedder:
    return Embedder(BGE_SMALL)


def _encode(texts: list[str]) -> np.ndarray:
    """Called by the cache only for texts it lacks, so a warm cache needs neither torch nor the model."""
    return _embedder().encode(texts)


def build(name: str, root: Path, cache_root: Path) -> Built:
    dev, test = load(name, DEV_SPLIT[name], root), load(name, "test", root)
    k1, b = tuned_bm25_params(name)
    index, _ = index_corpus(test.docs, DEFAULT)  # both splits share one corpus
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root)
    searcher = HybridSearcher(BM25(index, k1=k1, b=b), DEFAULT, ExactIndex(cache.get_many(list(test.docs), _encode)))
    prepared = {}
    for split, ds in (("dev", dev), ("test", test)):
        qids = sorted(ds.queries)
        texts = [BGE_SMALL.query_prefix + ds.queries[q] for q in qids]  # what Embedder.query_text adds
        prepared[split] = prepare(ds, searcher, cache.get_many(texts, _encode))
    return Built(searcher, {"k1": k1, "b": b}, prepared["dev"], prepared["test"])


def run_config(p: Prepared, config: FusionConfig, k: int) -> dict[str, list[str]]:
    """Rank every query. Same convention as Phases 1 and 2: drop a hit whose id is the query's."""
    ids, run = p.dataset.doc_ids, {}
    for q, c in zip(p.query_ids, p.by_depth[config.depth]):
        run[q] = [d for d in (ids[i] for i in fuse(c, config, k + 1)) if d != q][:k]
    return run


def dev_ndcg(p: Prepared, config: FusionConfig) -> float:
    return float(evaluate(run_config(p, config, 10), p.dataset.qrels).mean()["nDCG@10"])


def effective_dense_share(candidates: list[CandidateSet], config: FusionConfig) -> float | None:
    """How much of a convex fusion's ordering comes from dense, once normalisation has had its say.

    alpha is dense's weight on *normalised* scores, but normalisations leave the
    two retrievers with different spreads: theoretical min-max maps cosines of
    0.5 to 0.8 into a band about 0.1 wide while BM25 spans 0 to 1. A retriever
    moves the ranking in proportion to weight times spread, so the share is
    alpha * sd_dense / (alpha * sd_dense + (1 - alpha) * sd_bm25), spread being
    the standard deviation over the query's candidates. Median over queries.
    """
    if config.method != "convex":
        return None
    a, shares = config.weight, []
    for c in candidates:
        c = c.truncate(config.depth)
        sd = [normalise(c.scores[r], config.norm, THEORETICAL_MIN[r] if config.norm == "tmm" else None).std()
              for r in (0, 1)]
        if sd[0] + sd[1] > 0:
            shares.append(a * sd[1] / (a * sd[1] + (1 - a) * sd[0]))
    return float(np.median(shares))


def first_relevant_rank(ranked: list[str], judged: dict[str, int]) -> int | None:
    return next((i for i, d in enumerate(ranked, start=1) if judged.get(d, 0) > 0), None)


def evaluate_test(name: str, built: Built, grid: list[FusionConfig], dev_scores: np.ndarray,
                  global_config: FusionConfig, global_from: list[str], runs_dir: Path | None) -> tuple[dict, dict]:
    test = built.test
    per_dataset = grid[int(np.argmax(dev_scores))]
    systems = {
        "BM25": FusionConfig("bm25", depth=DEPTH),
        "Dense": FusionConfig("dense", depth=DEPTH),
        "RRF, k=60": DEFAULT_RRF,
        "Hybrid": global_config,
        "Hybrid, tuned per dataset": per_dataset,
        "Raw score sum": FusionConfig("raw_sum", depth=DEPTH),
    }
    runs = {s: run_config(test, c, DEPTH) for s, c in systems.items()}
    evals = {s: evaluate(r, test.dataset.qrels) for s, r in runs.items()}
    query_ids = evals["BM25"].query_ids
    assert all(e.query_ids == query_ids for e in evals.values())
    ndcg = {s: e.per_query["nDCG@10"] for s, e in evals.items()}
    best_single = np.maximum(ndcg["BM25"], ndcg["Dense"])

    def vs(a: str, b: str) -> dict:
        return {m: asdict(compare(evals[a].per_query[m], evals[b].per_query[m])) for m in ("nDCG@10", "R@100")}

    delta = ndcg["Hybrid"] - best_single
    loser_idx = [i for i in np.argsort(delta, kind="stable")[:LOSERS] if delta[i] < 0]
    hybrid_r100 = evals["Hybrid"].per_query["R@100"]
    qrels = test.dataset.qrels
    hard = {"recall": [], "ranking": []}
    for i, q in enumerate(query_ids):
        if ndcg["Hybrid"][i] > 0:
            continue
        hard["recall" if hybrid_r100[i] == 0 else "ranking"].append({
            "query_id": q, "query": test.dataset.queries[q],
            "first_relevant_rank": first_relevant_rank(runs["Hybrid"][q], qrels[q]),
            "bm25_nDCG@10": float(ndcg["BM25"][i]), "dense_nDCG@10": float(ndcg["Dense"][i]),
        })

    if runs_dir is not None:
        write_trec_run(runs_dir / f"{name}-test-hybrid.trec", runs["Hybrid"], "hybrid")

    result = {
        "dataset": name,
        "dev_split": DEV_SPLIT[name],
        "dev_queries": len(built.dev.query_ids),
        "test_queries": len(query_ids),
        "bm25_params": built.bm25_params,
        "dense_model": {"name": BGE_SMALL.name, "revision": BGE_SMALL.revision, "mode": "document"},
        "dev_grid": [{**asdict(c), "label": c.label(), "dev_nDCG@10": float(v)} for c, v in zip(grid, dev_scores)],
        "chosen": {
            "global": {**asdict(global_config), "label": global_config.label(), "chosen_on": global_from,
                       "dev_nDCG@10_here": float(dev_scores[grid.index(global_config)])},
            "per_dataset": {**asdict(per_dataset), "label": per_dataset.label(),
                            "dev_nDCG@10": float(dev_scores.max())},
        },
        "effective_dense_share": {"global": effective_dense_share(test.by_depth[global_config.depth], global_config),
                                  "per_dataset": effective_dense_share(test.by_depth[per_dataset.depth], per_dataset)},
        "systems": {s: c.label() for s, c in systems.items()},
        "test_metrics": {s: e.mean() for s, e in evals.items()},
        "test_metrics_note": "A fusion at depth 100 ranks at most the union of two top-100 lists, so its R@1000 "
                             "is capped and not comparable with the single retrievers'. nDCG@10 and R@100 are.",
        "oracle_nDCG@10": float(best_single.mean()),
        "comparisons": {
            "Hybrid vs BM25": vs("Hybrid", "BM25"),
            "Hybrid vs Dense": vs("Hybrid", "Dense"),
            "Hybrid vs RRF, k=60": vs("Hybrid", "RRF, k=60"),
            "Hybrid, tuned per dataset vs Hybrid": vs("Hybrid, tuned per dataset", "Hybrid"),
        },
        "queries_vs_better_single": {
            "hybrid_above": int((delta > 0).sum()), "hybrid_equal": int((delta == 0).sum()),
            "hybrid_below": int((delta < 0).sum()),
            "bm25_wins_dense": int((ndcg["BM25"] > ndcg["Dense"]).sum()),
            "dense_wins_bm25": int((ndcg["Dense"] > ndcg["BM25"]).sum()),
        },
        "hard_queries": {"recall": len(hard["recall"]), "ranking": len(hard["ranking"]),
                         "found_by_a_single_retriever": sum(1 for h in hard["recall"] + hard["ranking"]
                                                            if h["bm25_nDCG@10"] > 0 or h["dense_nDCG@10"] > 0)},
        "losers": [query_ids[i] for i in loser_idx],
    }
    details = {"runs": runs, "ndcg": ndcg, "query_ids": query_ids, "loser_idx": loser_idx, "hard": hard}
    return result, details


def snippet(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def losers_markdown(name: str, result: dict, details: dict, dataset: Dataset) -> str:
    pos = {d: i for i, d in enumerate(dataset.doc_ids)}
    ndcg, runs, qids = details["ndcg"], details["runs"], details["query_ids"]
    lines = [f"# {name}: where the hybrid falls furthest below the better single retriever", "",
             f"Test queries, nDCG@10. Hybrid is {result['chosen']['global']['label']}. For each query: the "
             "relevant documents, each system's top three, and where each system's first relevant document "
             "landed. ✓ marks a document judged relevant. The causes are read and labelled in "
             "docs/design/03-hybrid.md.", ""]
    for n, i in enumerate(details["loser_idx"], start=1):
        q = qids[i]
        judged = dataset.qrels[q]
        relevant = sorted((d for d, g in judged.items() if g > 0), key=lambda d: (-judged[d], d))
        ranks = {s: first_relevant_rank(runs[s][q], judged) for s in ("BM25", "Dense", "Hybrid")}
        lines += [f"## {n}. Query {q}: {snippet(dataset.queries[q], 200)}", "",
                  "nDCG@10: BM25 {:.3f}, dense {:.3f}, hybrid {:.3f}. First relevant at rank: BM25 {}, dense {}, "
                  "hybrid {}.".format(ndcg["BM25"][i], ndcg["Dense"][i], ndcg["Hybrid"][i],
                                      *(ranks[s] or "none in 1,000" for s in ("BM25", "Dense", "Hybrid"))), "",
                  f"Relevant ({len(relevant)} judged): `{relevant[0]}` (grade {judged[relevant[0]]}) "
                  f"{snippet(dataset.docs[pos[relevant[0]]], 300)}", ""]
        for system in ("BM25", "Dense", "Hybrid"):
            lines.append(f"**{system}**")
            lines.append("")
            for rank, d in enumerate(runs[system][q][:3], start=1):
                mark = "✓ " if judged.get(d, 0) > 0 else ""
                lines.append(f"{rank}. {mark}`{d}` {snippet(dataset.docs[pos[d]], 140)}")
            lines.append("")
    return "\n".join(lines)


def measure_latency(built: Built, config: FusionConfig, *, queries: int = 200, repeats: int = 3,
                    warmup: int = 20) -> dict:
    """One query at a time, end to end, split into its three stages. Query embedding is excluded (cached)."""
    s, test = built.searcher, built.test
    items = [(test.dataset.queries[q], v) for q, v in zip(test.query_ids[:queries], test.query_vectors[:queries])]
    stages = {"bm25_ms": [], "dense_ms": [], "fusion_ms": [], "total_ms": []}

    def one(text: str, vector: np.ndarray, record: bool) -> None:
        t0 = time.perf_counter()
        bm25 = s.bm25.scores(s.analyzer(text))
        t1 = time.perf_counter()
        dense = s.dense_scores(vector)[0]
        t2 = time.perf_counter()
        fuse(candidate_set(bm25, dense, config.depth), config, 10)
        t3 = time.perf_counter()
        if record:
            for key, value in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
                stages[key].append(value * 1000)

    for text, vector in items[:warmup]:
        one(text, vector, False)
    for _ in range(repeats):
        for text, vector in items:
            one(text, vector, True)
    return {"config": config.label(), "queries": len(items), "repeats": repeats,
            **{k: latency_summary(np.array(v)) for k, v in stages.items()},
            "excludes": "embedding the query (vectors come from the cache)", "machine": machine()}


def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down adjustment: controls the chance of any false positive across a family of tests."""
    m, adjusted, running = len(p_values), [0.0] * len(p_values), 0.0
    for i, idx in enumerate(np.argsort(p_values, kind="stable")):
        running = max(running, min(1.0, (m - i) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def summary_markdown(results: dict[str, dict]) -> str:
    names = [n for n in SOURCES if n in results]
    first = results[names[0]]
    g = first["chosen"]["global"]
    systems = list(first["test_metrics"])
    lines = [
        "# Phase 3: hybrid retrieval",
        "",
        "Test queries. **BM25** uses each dataset's tuned k1 and b from Phase 1; **Dense** is bge-small-en-v1.5 on "
        "whole documents from Phase 2. Fusion reorders the union of each retriever's top 100 or top 1,000 (the "
        "depth is chosen with the rest of the setting). **Hybrid** is one "
        f"setting for every dataset, chosen on development queries by average nDCG@10: **{g['label']}**. "
        "**Hybrid, tuned per dataset** uses each dataset's own best development setting. **RRF, k=60** is the "
        "untuned textbook setting. **Raw score sum** adds BM25 and cosine without normalising. **Oracle** picks, "
        "per query, whichever single retriever did better: a ceiling for any method that chooses between them.",
        "",
        "## nDCG@10",
        "",
        "| Dataset | " + " | ".join(systems) + " | Oracle |",
        "| --- |" + " --- |" * (len(systems) + 1),
    ]
    for n in names:
        m = results[n]["test_metrics"]
        lines.append(f"| {n} | " + " | ".join(f"{m[s]['nDCG@10']:.4f}" for s in systems)
                     + f" | {results[n]['oracle_nDCG@10']:.4f} |")
    lines += ["", "## Recall@100", "", "| Dataset | " + " | ".join(systems) + " |",
              "| --- |" + " --- |" * len(systems)]
    for n in names:
        m = results[n]["test_metrics"]
        lines.append(f"| {n} | " + " | ".join(f"{m[s]['R@100']:.4f}" for s in systems) + " |")

    family = [(n, label) for n in names for label in ("Hybrid vs BM25", "Hybrid vs Dense")]
    adjusted = dict(zip(family, holm([results[n]["comparisons"][label]["nDCG@10"]["p_value"] for n, label in family])))
    lines += ["", "## Paired comparisons", "",
              "Differences in mean over the same test queries, with 95% bootstrap intervals and two-sided "
              "randomization p-values. The six hybrid-against-single-retriever nDCG@10 tests are one family, so "
              "their p-values are also given after Holm's correction.", "",
              "| Dataset | Comparison | nDCG@10 difference | 95% interval | p | Holm p | R@100 difference | 95% interval |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        for label, c in results[n]["comparisons"].items():
            a, r = c["nDCG@10"], c["R@100"]
            holm_p = _p(adjusted[(n, label)]) if (n, label) in adjusted else ""
            lines.append(f"| {n} | {label} | {a['difference']:+.4f} | {a['ci_low']:+.4f} to {a['ci_high']:+.4f} "
                         f"| {_p(a['p_value'])} | {holm_p} | {r['difference']:+.4f} "
                         f"| {r['ci_low']:+.4f} to {r['ci_high']:+.4f} |")

    lines += ["", "## Settings chosen on development queries", "",
              "| Dataset | Split | Best RRF | Dev nDCG@10 | Best convex | Dev nDCG@10 | Global setting here |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        r = results[n]
        grid = r["dev_grid"]
        best = {m: max((row for row in grid if row["method"] == m), key=lambda row: row["dev_nDCG@10"])
                for m in ("rrf", "convex")}
        lines.append(f"| {n} | {r['dev_split']} ({r['dev_queries']}) | {best['rrf']['label']} "
                     f"| {best['rrf']['dev_nDCG@10']:.4f} | {best['convex']['label']} "
                     f"| {best['convex']['dev_nDCG@10']:.4f} | {r['chosen']['global']['dev_nDCG@10_here']:.4f} |")

    lines += ["", "Alpha is dense's weight on normalised scores, not its share of the ranking: the normalisations leave "
              "the two retrievers with different spreads. The effective dense share (alpha times dense's spread, over "
              "the total; median over test queries) puts the settings on one scale.", "",
              "| Dataset | Hybrid's effective dense share | Per-dataset setting's |", "| --- | --- | --- |"]
    for n in names:
        e = results[n]["effective_dense_share"]
        lines.append(f"| {n} | " + " | ".join("n/a" if v is None else f"{v:.2f}" for v in (e["global"], e["per_dataset"]))
                     + " |")

    shown = (0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
    depth = g["depth"]
    lines += ["", "## Sensitivity on development queries", "",
              f"Dev nDCG@10 at depth {depth}, the best value in each row in bold. For convex combination, against "
              "alpha (0 is BM25 alone, 1 dense alone):"]
    for norm in NORMS:
        lines += ["", f"Convex {norm}", "", "| Dataset | " + " | ".join(f"{a:g}" for a in shown) + " |",
                  "| --- |" + " --- |" * len(shown)]
        for n in names:
            rows = {row["weight"]: row["dev_nDCG@10"] for row in results[n]["dev_grid"]
                    if row["method"] == "convex" and row["norm"] == norm and row["depth"] == depth}
            best = max(rows.values())
            lines.append(f"| {n} | " + " | ".join(f"**{rows[a]:.4f}**" if rows[a] == best else f"{rows[a]:.4f}"
                                                   for a in shown) + " |")
    lines += ["", "RRF with equal weights, against k:", "",
              "| Dataset | " + " | ".join(f"k={k}" for k in RRF_K) + " |", "| --- |" + " --- |" * len(RRF_K)]
    for n in names:
        rows = {row["k"]: row["dev_nDCG@10"] for row in results[n]["dev_grid"]
                if row["method"] == "rrf" and row["weight"] == 0.5 and row["depth"] == depth}
        best = max(rows.values())
        lines.append(f"| {n} | " + " | ".join(f"**{rows[float(k)]:.4f}**" if rows[float(k)] == best
                                               else f"{rows[float(k)]:.4f}" for k in RRF_K) + " |")

    lines += ["", "## Query by query", "",
              "Against the better of the two single retrievers on each query (nDCG@10). Hard queries: the hybrid "
              "puts nothing relevant in its top 10. A recall failure has nothing relevant in the top 100 either, "
              "so no reranker can fix it; a ranking failure has, so Phase 4's reranker might.", "",
              "| Dataset | Queries | Hybrid above | Equal | Below | Hard: ranking | Hard: recall | Hard, though one retriever found something |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        r = results[n]
        c, h = r["queries_vs_better_single"], r["hard_queries"]
        lines.append(f"| {n} | {r['test_queries']} | {c['hybrid_above']} | {c['hybrid_equal']} | {c['hybrid_below']} "
                     f"| {h['ranking']} | {h['recall']} | {h['found_by_a_single_retriever']} |")
    lines += ["", "The ten worst losses per dataset, with the documents involved: "
              + ", ".join(f"[{n}](losers_{n}.md)" for n in names) + ". Hard queries: [hard_queries.json](hard_queries.json).", ""]

    lat = next((results[n].get("latency") for n in names if results[n].get("latency")), None)
    if lat:
        mach = lat["machine"]
        lines += ["## Cost", "",
                  f"FiQA, {lat['queries']} test queries × {lat['repeats']}, one at a time, {lat['config']}. "
                  f"Excludes {lat['excludes']}.", "",
                  "| Stage | p50 | p95 |", "| --- | --- | --- |"]
        for key, label in (("bm25_ms", "BM25: analyze and score"), ("dense_ms", "Dense: score every document"),
                           ("fusion_ms", "Candidates and fusion"), ("total_ms", "Total")):
            lines.append(f"| {label} | {lat[key]['p50_ms']:.2f} ms | {lat[key]['p95_ms']:.2f} ms |")
        lines += ["", f"Machine: {mach['platform']}, Python {mach['python']}, NumPy {mach['numpy']}, "
                      f"{mach['cpus']} CPUs, NumPy's default threads.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", choices=[*SOURCES, "all"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()

    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    grid = fusion_grid()
    built, dev_scores = {}, {}
    for name in names:
        start = time.perf_counter()
        built[name] = build(name, args.data_dir, args.cache_dir)
        dev_scores[name] = np.array([dev_ndcg(built[name].dev, c) for c in grid])
        best = grid[int(np.argmax(dev_scores[name]))]
        print(f"{name:9} {len(grid)} settings on {len(built[name].dev.query_ids)} dev queries in "
              f"{time.perf_counter() - start:.0f} s; best {best.label()} ({dev_scores[name].max():.4f})", flush=True)

    average = np.mean([dev_scores[n] for n in names], axis=0)
    global_config = grid[int(np.argmax(average))]  # the first of any tie, so the simpler setting
    print(f"global    {global_config.label()} (average dev nDCG@10 {average.max():.4f})", flush=True)

    results, hard = {}, {"definition": "Test queries where the hybrid's nDCG@10 is 0. recall: nothing relevant in "
                                       "its top 100; ranking: something relevant in the top 100, none in the top 10.",
                         "hybrid": global_config.label()}
    for name in names:
        result, details = evaluate_test(name, built[name], grid, dev_scores[name], global_config, names, Path("runs"))
        if name == "fiqa":
            result["latency"] = measure_latency(built[name], global_config)
        write_json(args.out / f"hybrid_{name}.json", result)
        (args.out / f"losers_{name}.md").write_text(losers_markdown(name, result, details, built[name].test.dataset))
        hard[name] = details["hard"]
        results[name] = result
        m = result["test_metrics"]
        print(f"{name:9} test nDCG@10: BM25 {m['BM25']['nDCG@10']:.4f}  dense {m['Dense']['nDCG@10']:.4f}  "
              f"hybrid {m['Hybrid']['nDCG@10']:.4f}  oracle {result['oracle_nDCG@10']:.4f}", flush=True)
    if names == list(SOURCES):
        write_json(args.out / "hard_queries.json", hard)
        (args.out / "hybrid.md").write_text(summary_markdown(results))


if __name__ == "__main__":
    main()
