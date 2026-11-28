import numpy as np
import pytest

import json
import sys
from pathlib import Path

import memret.evaluate_hybrid as eh
from memret.bm25 import BM25, InvertedIndex
from memret.datasets import SOURCES, Dataset
from memret.dense import ExactIndex
from memret.evaluate_hybrid import DEFAULT_RRF, effective_dense_share, fusion_grid, holm, pairwise_dense_share
from memret.hybrid import SHIPPED, FusionConfig, HybridSearcher, candidate_set
from memret.text import DEFAULT


def test_grid_starts_with_textbook_rrf_and_has_no_duplicates():
    grid = fusion_grid()
    assert grid[0] == DEFAULT_RRF  # wins any tie, so an untuned setting is never displaced by noise
    assert len(set(grid)) == len(grid)
    assert {c.method for c in grid} == {"rrf", "convex"}


def test_holm_by_hand():
    # Sorted p: 0.01, 0.02, 0.04 -> times 3, 2, 1 -> 0.03, 0.04, 0.04 (kept non-decreasing).
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    # Capped at 1, and a later p can't come out below an earlier adjusted one.
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm([0.001]) == pytest.approx([0.001])


def test_effective_share_is_alpha_when_spreads_match():
    rng = np.random.default_rng(0)
    cands = [candidate_set(rng.random(50) * 10, rng.random(50), depth=50) for _ in range(5)]
    # minmax gives both retrievers similar spreads, so the share sits near alpha;
    # zscore gives both a standard deviation of exactly 1, so it is alpha.
    assert effective_dense_share(cands, FusionConfig("convex", depth=50, weight=0.7, norm="zscore")) == pytest.approx(0.7)
    assert effective_dense_share(cands, FusionConfig("convex", depth=50, weight=0.7, norm="minmax")) == pytest.approx(0.7, abs=0.05)


def test_effective_share_shrinks_under_tmm_when_cosines_are_bunched():
    rng = np.random.default_rng(1)
    # BM25 spans 0..10; cosines sit in 0.6..0.7, which tmm maps into a band about 0.06 wide.
    cands = [candidate_set(rng.random(50) * 10, 0.6 + 0.1 * rng.random(50), depth=50) for _ in range(5)]
    share = effective_dense_share(cands, FusionConfig("convex", depth=50, weight=0.9, norm="tmm"))
    assert share < 0.5  # alpha 0.9 on paper, a minority of the ranking in practice


def test_effective_share_only_for_convex():
    cands = [candidate_set(np.array([1.0, 2.0]), np.array([0.1, 0.2]), depth=2)]
    assert effective_dense_share(cands, DEFAULT_RRF) is None


def test_pairwise_share_is_one_for_dense_alone_and_zero_for_bm25_alone():
    rng = np.random.default_rng(2)
    cands = [candidate_set(rng.random(60) * 10, rng.random(60), depth=60) for _ in range(5)]
    assert pairwise_dense_share(cands, FusionConfig("dense", depth=60)) == 1.0
    assert pairwise_dense_share(cands, FusionConfig("bm25", depth=60)) == 0.0
    assert pairwise_dense_share(cands, FusionConfig("convex", depth=60, weight=1.0, norm="tmm")) == 1.0
    mixed = pairwise_dense_share(cands, FusionConfig("convex", depth=60, weight=0.5, norm="zscore"))
    assert 0.0 < mixed < 1.0


def test_pairwise_share_is_none_without_disputes():
    scores = np.array([3.0, 2.0, 1.0])
    cands = [candidate_set(scores, scores / 10, depth=3)]  # the two retrievers agree on every pair
    assert pairwise_dense_share(cands, FusionConfig("convex", depth=3, weight=0.5, norm="minmax")) is None


def test_shipped_setting_is_the_one_the_reports_chose():
    # hybrid.SHIPPED names the Phase 3 choice once; if a re-run chooses differently, this fails until it is updated.
    report = Path(__file__).resolve().parents[1] / "reports" / "phase3" / "hybrid_scifact.json"
    chosen = json.loads(report.read_text())["chosen"]["global"]
    assert (chosen["method"], chosen["depth"], chosen["weight"], chosen["norm"]) == \
        (SHIPPED.method, SHIPPED.depth, SHIPPED.weight, SHIPPED.norm)


def test_choose_decides_on_development_scores_alone(monkeypatch):
    grid = [FusionConfig("bm25"), FusionConfig("dense"), DEFAULT_RRF]
    dev_scores = {"a": [0.5, 0.7, 0.7], "b": [0.6, 0.2, 0.9]}
    monkeypatch.setattr(eh, "dev_ndcg", lambda p, c: dev_scores[p][grid.index(c)])
    choice = eh.choose({"a": "a", "b": "b"}, grid)
    assert choice.per_dataset == {"a": grid[1], "b": grid[2]}  # a's tie goes to the earlier setting
    assert choice.global_config == grid[2]  # averages 0.55, 0.45, 0.80
    assert choice.global_dev_average == pytest.approx(0.8)


TOY_DOCS = [
    "heart rate and pulse", "lower your mortgage payment", "pulse oximeters measure blood oxygen",
    "ways to reduce heart rate", "cats and dogs", "the heart of the city", "resting pulse rate in athletes",
    "mortgage rates and the heart of lending", "dogs that lower blood pressure", "city rates for oxygen",
    "athletes and their resting heart", "pressure on mortgage lending",
]


def toy_built(name: str) -> "eh.Built":
    """A tiny dataset with development and test queries, built the way evaluate_hybrid builds the real ones."""
    rng = np.random.default_rng(sorted(SOURCES).index(name))
    vectors = rng.standard_normal((len(TOY_DOCS), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    searcher = HybridSearcher(BM25(InvertedIndex([DEFAULT(d) for d in TOY_DOCS])), DEFAULT, ExactIndex(vectors))
    doc_ids = [f"d{i}" for i in range(len(TOY_DOCS))]

    def split(label: str, texts: list[str]) -> "eh.Prepared":
        queries = {f"{label}-{i}": t for i, t in enumerate(texts)}
        qrels = {q: {doc_ids[(3 * i + 1) % len(doc_ids)]: 1} for i, q in enumerate(sorted(queries))}
        qv = rng.standard_normal((len(queries), 8)).astype(np.float32)
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        return eh.prepare(Dataset(name, label, doc_ids, TOY_DOCS, queries, qrels), searcher, qv)

    return eh.Built(searcher, {"k1": 0.9, "b": 0.4},
                    split("dev", ["heart rate", "mortgage payment", "blood oxygen", "resting athletes"]),
                    split("test", ["pulse rate", "lending rates", "dogs and cats", "city heart"]))


def test_main_runs_no_test_query_before_both_choices_are_fixed(monkeypatch, tmp_path):
    # The protocol in one assertion: every development query is scored before any test query is run.
    calls, chosen = [], []
    real_run_config, real_choose = eh.run_config, eh.choose

    def logged_run_config(p, config, k):
        calls.append(p.dataset.split)
        return real_run_config(p, config, k)

    def logged_choose(dev, grid):
        choice = real_choose(dev, grid)
        chosen.append(len(calls))  # how many runs had happened when the choice was fixed
        return choice

    monkeypatch.setattr(eh, "build", lambda name, root, cache_root: toy_built(name))
    monkeypatch.setattr(eh, "run_config", logged_run_config)
    monkeypatch.setattr(eh, "choose", logged_choose)
    monkeypatch.setattr(sys, "argv", ["evaluate_hybrid", "--dataset", "all", "--out", str(tmp_path / "out")])
    monkeypatch.chdir(tmp_path)
    eh.main()

    assert calls.count("dev") == len(SOURCES) * len(fusion_grid())
    first_test = calls.index("test")
    assert "dev" not in calls[first_test:]
    assert chosen == [first_test]  # the choice was fixed after the last dev run and before the first test run
    report = json.loads((tmp_path / "out" / "hybrid_fiqa.json").read_text())
    assert report["chosen"]["global"]["chosen_on"] == list(SOURCES)
