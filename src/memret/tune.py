"""Choose BM25's k1 and b on development queries, then test the choice once.

    uv run python -m memret.tune --dataset all

For each dataset: a grid of k1 and b scored by nDCG@10 on the development
queries, an analyzer ablation (no stemming, no stopword removal) on the same
queries, and finally one comparison on the test queries between the tuned
setting and the default (k1 = 0.9, b = 0.4). The test queries are touched only
by that last step. Writes reports/phase1/tuning_<dataset>.json and, for all
datasets, reports/phase1/tuning.md.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .bm25 import BM25
from .datasets import DEFAULT_ROOT, DEV_SPLIT, SOURCES, Dataset, load
from .experiment import index_corpus, run_queries, write_json
from .metrics import evaluate
from .stats import compare
from .text import DEFAULT, NO_STEM, NO_STOPWORDS, Analyzer

K1_GRID = (0.5, 0.7, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)
DEFAULT_PARAMS = (0.9, 0.4)


def ndcg10(bm25: BM25, analyzer: Analyzer, dataset: Dataset):
    run, _ = run_queries(bm25, analyzer, dataset, k=10)  # nDCG@10 needs only the top 10
    return evaluate(run, dataset.qrels).per_query["nDCG@10"]


def tune_dataset(name: str, root: Path = DEFAULT_ROOT) -> dict:
    dev = load(name, DEV_SPLIT[name], root)
    test = load(name, "test", root)
    index, _ = index_corpus(dev.docs, DEFAULT)  # both splits share one corpus

    grid = {(k1, b): float(ndcg10(BM25(index, k1, b), DEFAULT, dev).mean()) for k1 in K1_GRID for b in B_GRID}
    # Highest score wins; on an exact tie, the setting closest to the default does.
    best = max(grid, key=lambda p: (grid[p], -abs(p[0] - DEFAULT_PARAMS[0]) - abs(p[1] - DEFAULT_PARAMS[1])))

    default_dev = ndcg10(BM25(index, *DEFAULT_PARAMS), DEFAULT, dev)
    ablation = {"default analyzer": {"dev_nDCG@10": float(default_dev.mean())}}
    for label, analyzer in (("no stemming", NO_STEM), ("no stopword removal", NO_STOPWORDS)):
        other_index, _ = index_corpus(dev.docs, analyzer)
        scores = ndcg10(BM25(other_index, *DEFAULT_PARAMS), analyzer, dev)
        ablation[label] = {"dev_nDCG@10": float(scores.mean()), "vs_default": asdict(compare(scores, default_dev))}

    tuned_test = ndcg10(BM25(index, *best), DEFAULT, test)
    default_test = ndcg10(BM25(index, *DEFAULT_PARAMS), DEFAULT, test)
    return {
        "dataset": name,
        "dev_split": DEV_SPLIT[name],
        "dev_queries": len(dev.queries),
        "test_queries": len(test.queries),
        "grid_dev_nDCG@10": [{"k1": k1, "b": b, "nDCG@10": v} for (k1, b), v in grid.items()],
        "best": {"k1": best[0], "b": best[1], "dev_nDCG@10": grid[best]},
        "default_dev_nDCG@10": grid[DEFAULT_PARAMS],
        "analyzer_ablation_dev": ablation,
        "test_tuned_vs_default": asdict(compare(tuned_test, default_test)),
    }


def tuning_markdown(results: list[dict]) -> str:
    lines = ["# Phase 1: tuning BM25 on development queries", "",
             "Each grid is nDCG@10 on development queries only. The best setting is then compared with the "
             "default (k1 = 0.9, b = 0.4) on test queries, once. Intervals are 95% paired bootstrap over "
             "queries; p-values are two-sided paired randomization tests.", ""]
    for r in results:
        grid = {(g["k1"], g["b"]): g["nDCG@10"] for g in r["grid_dev_nDCG@10"]}
        best = (r["best"]["k1"], r["best"]["b"])
        lines += [f"## {r['dataset']}", "",
                  f"Development: {r['dev_queries']} queries from the {r['dev_split']} split. "
                  f"Test: {r['test_queries']} queries.", "",
                  "| k1 \\ b | " + " | ".join(str(b) for b in B_GRID) + " |",
                  "| --- |" + " --- |" * len(B_GRID)]
        for k1 in K1_GRID:
            cells = [(f"**{grid[(k1, b)]:.4f}**" if (k1, b) == best else f"{grid[(k1, b)]:.4f}") for b in B_GRID]
            lines.append(f"| {k1} | " + " | ".join(cells) + " |")
        t = r["test_tuned_vs_default"]
        lines += ["", f"Best on development: k1 = {best[0]}, b = {best[1]} "
                      f"({r['best']['dev_nDCG@10']:.4f} against {r['default_dev_nDCG@10']:.4f} for the default).", "",
                  f"On test, tuned against default: {t['mean_a']:.4f} against {t['mean_b']:.4f}, difference "
                  f"{t['difference']:+.4f} (95% interval {t['ci_low']:+.4f} to {t['ci_high']:+.4f}, "
                  f"p = {t['p_value']:.3g}).", "",
                  "| Analyzer (development queries, default k1 and b) | nDCG@10 | Difference from default | 95% interval | p |",
                  "| --- | --- | --- | --- | --- |"]
        for label, a in r["analyzer_ablation_dev"].items():
            if "vs_default" in a:
                c = a["vs_default"]
                lines.append(f"| {label} | {a['dev_nDCG@10']:.4f} | {c['difference']:+.4f} "
                             f"| {c['ci_low']:+.4f} to {c['ci_high']:+.4f} | {c['p_value']:.3g} |")
            else:
                lines.append(f"| {label} | {a['dev_nDCG@10']:.4f} | | | |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", choices=[*SOURCES, "all"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase1"))
    args = parser.parse_args()
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    results = []
    for name in names:
        result = tune_dataset(name, args.data_dir)
        write_json(args.out / f"tuning_{name}.json", result)
        results.append(result)
        t = result["test_tuned_vs_default"]
        print(f"{name:9} best k1={result['best']['k1']} b={result['best']['b']}  "
              f"test {t['mean_a']:.4f} vs default {t['mean_b']:.4f}  p={t['p_value']:.3g}", flush=True)
    if args.dataset == "all":
        (args.out / "tuning.md").write_text(tuning_markdown(results))


if __name__ == "__main__":
    main()
