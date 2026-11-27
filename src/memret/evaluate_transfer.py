"""Run the shipped hybrid, unchanged, on a dataset held out from every choice so far.

    uv run python -m memret.evaluate_transfer --dataset arguana

Phase 3 chose its fusion setting on development queries of SciFact, NFCorpus and
FiQA, and reported it on their test queries. That shows the setting works on the
data it was chosen on, not that it travels. A held-out dataset is a new user's
situation: no judgements to tune on. So nothing is tuned here. BM25 keeps Phase
1's default k1 and b, and the hybrid is `hybrid.SHIPPED` as it stands.

The first run embeds the collection through Phase 2's cache (about 10 minutes on
2 CPUs for ArguAna); later runs take seconds.

Writes reports/phase3/transfer_<dataset>.json and transfer_<dataset>.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .bm25 import BM25
from .cache import EmbeddingCache
from .datasets import DEFAULT_ROOT, HELD_OUT, load
from .dense import ExactIndex
from .embed import BGE_SMALL
from .evaluate_dense import CACHE_ROOT
from .evaluate_hybrid import (DEFAULT_RRF, DEPTH, _encode, effective_dense_share, holm, pairwise_dense_share,
                              prepare, run_config)
from .experiment import index_corpus, write_json
from .hybrid import SHIPPED, FusionConfig, HybridSearcher, fuse
from .metrics import evaluate
from .stats import compare
from .text import DEFAULT

SYSTEMS = {
    "BM25, default k1 and b": FusionConfig("bm25", depth=DEPTH),
    "Dense": FusionConfig("dense", depth=DEPTH),
    "RRF, k=60": DEFAULT_RRF,
    "Hybrid": SHIPPED,
}


def transfer(name: str, root: Path, cache_root: Path) -> dict:
    ds = load(name, "test", root)
    index, _ = index_corpus(ds.docs, DEFAULT)
    bm25 = BM25(index)  # no development queries, so Phase 1's defaults, untuned
    cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, cache_root)
    searcher = HybridSearcher(bm25, DEFAULT, ExactIndex(cache.get_many(list(ds.docs), _encode)))
    qids = sorted(ds.queries)
    p = prepare(ds, searcher, cache.get_many([BGE_SMALL.query_prefix + ds.queries[q] for q in qids], _encode))

    evals = {s: evaluate(run_config(p, c, DEPTH), ds.qrels) for s, c in SYSTEMS.items()}
    tests = {}
    for other in ("BM25, default k1 and b", "Dense"):
        tests[f"Hybrid vs {other}"] = {m: asdict(compare(evals["Hybrid"].per_query[m], evals[other].per_query[m]))
                                       for m in ("nDCG@10", "R@100")}
    adjusted = holm([t["nDCG@10"]["p_value"] for t in tests.values()])
    for t, h in zip(tests.values(), adjusted):
        t["nDCG@10"]["holm_p"] = h

    # A query whose own id is a document id is usually its own best match, and unjudged.
    # run_config drops it, as BEIR's evaluation does; count how often that mattered.
    doc_ids = set(ds.doc_ids)
    own = [i for i, q in enumerate(p.query_ids) if q in doc_ids]
    own_in_top10 = {s: sum(p.query_ids[i] in {ds.doc_ids[d] for d in fuse(p.by_depth[c.depth][i], c, 10)} for i in own)
                    for s, c in SYSTEMS.items()}

    tuned = {}
    for n in ("scifact", "nfcorpus", "fiqa"):
        path = Path("reports/phase3") / f"hybrid_{n}.json"
        if path.exists():
            r = json.loads(path.read_text())
            tuned[n] = {"proxy": r.get("effective_dense_share", {}).get("global"),
                        "pairwise": r.get("pairwise_dense_share_top20", {}).get("global")}
    return {
        "dataset": name,
        "split": "test",
        "queries": len(p.query_ids),
        "documents": len(ds.docs),
        "tuned_on_this_dataset": "nothing",
        "bm25_params": {"k1": bm25.k1, "b": bm25.b},
        "dense_model": {"name": BGE_SMALL.name, "revision": BGE_SMALL.revision, "mode": "document"},
        "systems": {s: c.label() for s, c in SYSTEMS.items()},
        "test_metrics": {s: e.mean() for s, e in evals.items()},
        "test_metrics_note": "The hybrid ranks at most the union of two top-100 lists, so its R@1000 is capped and "
                             "not comparable. nDCG@10 and R@100 are.",
        "comparisons": tests,
        "dense_share": {"pairwise_top20": pairwise_dense_share(p.by_depth[SHIPPED.depth], SHIPPED),
                        "proxy": effective_dense_share(p.by_depth[SHIPPED.depth], SHIPPED),
                        "on_the_tuning_datasets": tuned},
        "queries_whose_id_is_a_document": len(own),
        "own_document_in_top10_before_dropping": own_in_top10,
    }


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def summary_markdown(r: dict) -> str:
    m, share = r["test_metrics"], r["dense_share"]
    lines = [
        f"# Phase 3, held out: {r['dataset']}",
        "",
        f"{r['queries']:,} test queries over {r['documents']:,} documents. Nothing was tuned on this dataset: BM25 uses "
        f"Phase 1's defaults (k1 {r['bm25_params']['k1']:g}, b {r['bm25_params']['b']:g}) and the hybrid is the "
        f"shipped setting, **{r['systems']['Hybrid']}**, chosen on the three Phase 3 datasets.",
        "",
        "| System | nDCG@10 | R@100 |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {s} | {m[s]['nDCG@10']:.4f} | {m[s]['R@100']:.4f} |" for s in r["systems"]]
    lines += ["", "| Comparison | nDCG@10 difference | 95% interval | Holm p | R@100 difference | 95% interval |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, c in r["comparisons"].items():
        a, b = c["nDCG@10"], c["R@100"]
        lines.append(f"| {label} | {a['difference']:+.4f} | {a['ci_low']:+.4f} to {a['ci_high']:+.4f} "
                     f"| {_p(a['holm_p'])} | {b['difference']:+.4f} | {b['ci_low']:+.4f} to {b['ci_high']:+.4f} |")
    tuned = share["on_the_tuning_datasets"]
    lines += ["", "Dense's share of the shipped setting's ordering (see `hybrid.md` for both measures):", "",
              "| Dataset | Pairwise, top 20 | Proxy |", "| --- | --- | --- |",
              f"| {r['dataset']} (held out) | {share['pairwise_top20']:.2f} | {share['proxy']:.2f} |"]
    lines += [f"| {n} | {v['pairwise']:.2f} | {v['proxy']:.2f} |" for n, v in tuned.items()
              if v["pairwise"] is not None and v["proxy"] is not None]
    if r["queries_whose_id_is_a_document"]:
        lines += ["", f"{r['queries_whose_id_is_a_document']:,} of the queries are themselves documents in the "
                      "collection. Each system's top 10 contains the query's own document on this many of them before "
                      "it is dropped: " + ", ".join(f"{s} {n:,}" for s, n in r["own_document_in_top10_before_dropping"].items())
                  + ". The own document is unjudged, so it is dropped from every ranking, as in Phases 1 to 3 and "
                    "BEIR's evaluation."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="arguana", choices=sorted(HELD_OUT))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()

    r = transfer(args.dataset, args.data_dir, args.cache_dir)
    write_json(args.out / f"transfer_{args.dataset}.json", r)
    (args.out / f"transfer_{args.dataset}.md").write_text(summary_markdown(r))
    m = r["test_metrics"]
    print("  ".join(f"{s}: {m[s]['nDCG@10']:.4f}" for s in r["systems"]))


if __name__ == "__main__":
    main()
