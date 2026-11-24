"""Score BM25 on BEIR test queries and compare with Anserini's published baseline.

    uv run python -m memret.evaluate --dataset scifact
    uv run python -m memret.evaluate --dataset all

Writes reports/phase1/bm25_<dataset>.json, and reports/phase1/bm25.md when run
on all datasets. Run files and per-query scores go to runs/ (not committed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bm25 import BM25
from .datasets import DEFAULT_ROOT, SOURCES, load
from .experiment import index_corpus, latency_summary, machine, run_queries, write_json, write_trec_run
from .metrics import evaluate
from .text import LUCENE_STOPWORDS, Analyzer

# Anserini's BM25 "flat" BEIR regressions (title and text in one field, k1 = 0.9,
# b = 0.4, test queries), and the index statistics its configs record.
REFERENCE_URL = ("https://github.com/castorini/anserini/tree/master/src/main/resources/"
                 "reproduce/from-document-collection/configs")
REFERENCE = {
    "scifact": {"nDCG@10": 0.6789, "R@100": 0.9253, "R@1000": 0.9767, "docs": 5183, "total_terms": 838128},
    "nfcorpus": {"nDCG@10": 0.3218, "R@100": 0.2457, "R@1000": 0.3704, "docs": 3633, "total_terms": 637485},
    "fiqa": {"nDCG@10": 0.2361, "R@100": 0.5395, "R@1000": 0.7393, "docs": 57600, "total_terms": 5288635},
}


def evaluate_dataset(name: str, *, split: str = "test", k1: float = 0.9, b: float = 0.4, stem: bool = True,
                     stopwords: bool = True, drop_query_id: bool = True, root: Path = DEFAULT_ROOT,
                     runs_dir: Path | None = None) -> dict:
    dataset = load(name, split, root)
    analyzer = Analyzer(stopwords=LUCENE_STOPWORDS if stopwords else frozenset(), stem=stem)
    index, build = index_corpus(dataset.docs, analyzer)
    bm25 = BM25(index, k1=k1, b=b)
    run, latencies = run_queries(bm25, analyzer, dataset, drop_query_id=drop_query_id)
    scores = evaluate(run, dataset.qrels)
    # The same ranking with Lucene's one-byte document lengths: a parity check
    # against Anserini, not a different system.
    lucene_run, _ = run_queries(BM25(index, k1=k1, b=b, lengths="lucene"), analyzer, dataset,
                                drop_query_id=drop_query_id)
    if runs_dir is not None:
        tag = f"bm25-k1{k1}-b{b}"
        write_trec_run(runs_dir / f"{name}-{split}-{tag}.trec", run, tag)
        with (runs_dir / f"{name}-{split}-{tag}.per-query.tsv").open("w") as f:
            f.write("query\t" + "\t".join(scores.per_query) + "\n")
            for i, q in enumerate(scores.query_ids):
                f.write(q + "\t" + "\t".join(f"{v[i]:.6f}" for v in scores.per_query.values()) + "\n")
    return {
        "dataset": name,
        "split": split,
        "config": {"k1": k1, "b": b, "stem": stem, "stopwords": stopwords, "drop_query_id": drop_query_id},
        "corpus": {"docs": len(dataset.docs), "non_empty_docs": index.num_docs, "total_terms": index.total_terms,
                   "vocabulary": len(index.vocab), "postings": int(len(index.postings_doc))},
        "queries": len(scores.query_ids),
        "metrics": scores.mean(),
        "metrics_lucene_lengths": evaluate(lucene_run, dataset.qrels).mean(),
        "timing": {**build, "query": latency_summary(latencies)},
        "machine": machine(),
    }


def summary_markdown(results: list[dict]) -> str:
    lines = [
        "# Phase 1: BM25 from scratch against Anserini",
        "",
        "Test queries, k1 = 0.9, b = 0.4. **Ours** is the analyzer and BM25 in `src/memret`. "
        "**Lucene lengths** is the same code storing document lengths the way Lucene does, in one byte. "
        f"**Anserini** is the published BM25 flat baseline ([configs]({REFERENCE_URL})).",
        "",
        "| Dataset | Queries | Metric | Ours | Lucene lengths | Anserini |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m, lm, ref = r["metrics"], r["metrics_lucene_lengths"], REFERENCE[r["dataset"]]
        for metric in ("nDCG@10", "R@100", "R@1000"):
            lines.append(f"| {r['dataset']} | {r['queries']} | {metric} | {m[metric]:.4f} | {lm[metric]:.4f} "
                         f"| {ref[metric]:.4f} |")
    lines += [
        "",
        "## Index statistics against Anserini's",
        "",
        "Total terms is the number of tokens left after analysis. It is the most direct check that our "
        "analyzer matches Lucene's.",
        "",
        "| Dataset | Docs indexed | Anserini | Total terms | Anserini | Difference | Vocabulary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        c, ref = r["corpus"], REFERENCE[r["dataset"]]
        diff = (c["total_terms"] - ref["total_terms"]) / ref["total_terms"]
        lines.append(f"| {r['dataset']} | {c['non_empty_docs']:,} | {ref['docs']:,} | {c['total_terms']:,} "
                     f"| {ref['total_terms']:,} | {diff:+.2%} | {c['vocabulary']:,} |")
    lines += [
        "",
        "## Speed",
        "",
        "| Dataset | Analyze corpus | Build index | Query p50 | Query p95 | Queries |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        t = r["timing"]
        lines.append(f"| {r['dataset']} | {t['analyze_s']:.1f} s | {t['index_s']:.2f} s | {t['query']['p50_ms']:.2f} ms "
                     f"| {t['query']['p95_ms']:.2f} ms | {t['query']['n']} |")
    mach = results[0]["machine"]
    lines += ["", f"Machine: {mach['platform']}, Python {mach['python']}, NumPy {mach['numpy']}, "
                  f"{mach['cpus']} CPUs. Single-threaded.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", choices=[*SOURCES, "all"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--keep-query-id", action="store_true", help="do not drop documents whose id equals the query id")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase1"))
    args = parser.parse_args()

    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    results = []
    for name in names:
        result = evaluate_dataset(name, split=args.split, k1=args.k1, b=args.b, stem=not args.no_stem,
                                  stopwords=not args.no_stopwords, drop_query_id=not args.keep_query_id,
                                  root=args.data_dir, runs_dir=Path("runs"))
        results.append(result)
        m = result["metrics"]
        print(f"{name:9} nDCG@10 {m['nDCG@10']:.4f} (Anserini {REFERENCE[name]['nDCG@10']:.4f})  "
              f"R@100 {m['R@100']:.4f}  R@1000 {m['R@1000']:.4f}  MRR@10 {m['MRR@10']:.4f}", flush=True)
    default_config = (args.split == "test" and args.k1 == 0.9 and args.b == 0.4 and not args.no_stem
                      and not args.no_stopwords and not args.keep_query_id)
    if default_config:  # only the reference configuration is written to the committed reports
        for result in results:
            write_json(args.out / f"bm25_{result['dataset']}.json", result)
        if args.dataset == "all":
            (args.out / "bm25.md").write_text(summary_markdown(results))


if __name__ == "__main__":
    main()
