import math

import pytest

from memret.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    judged = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], judged, 10) == pytest.approx(1.0)


def test_ndcg_by_hand():
    judged = {"a": 1, "b": 1}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)  # relevant at ranks 2 and 4
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y", "b"], judged, 10) == pytest.approx(dcg / idcg)


def test_ndcg_uses_graded_relevance():
    judged = {"a": 2, "b": 1}
    swapped = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], judged, 10) == pytest.approx(swapped)


def test_cutoffs():
    ranked = [f"x{i}" for i in range(10)] + ["a"]  # the relevant document is 11th
    assert ndcg_at_k(ranked, {"a": 1}, 10) == 0.0
    assert recall_at_k(ranked, {"a": 1}, 100) == 1.0
    assert mrr_at_k(ranked, {"a": 1}, 10) == 0.0


def test_zero_judgements_are_not_relevant():
    judged = {"a": 1, "b": 0}
    assert recall_at_k(["b"], judged, 10) == 0.0
    assert recall_at_k(["a"], judged, 10) == 1.0
    assert mrr_at_k(["b", "a"], judged, 10) == 0.5


def test_recall_counts_each_relevant_document_once():
    assert recall_at_k(["a", "c"], {"a": 1, "b": 1, "c": 1}, 2) == pytest.approx(2 / 3)


def test_evaluate_counts_unanswered_queries_and_skips_unjudgeable_ones():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    result = evaluate({"q1": ["a"]}, qrels)
    assert result.query_ids == ["q1", "q2"]  # q3 has no relevant document
    assert result.per_query["nDCG@10"].tolist() == [1.0, 0.0]
    assert result.mean()["R@100"] == pytest.approx(0.5)


def test_oracle_scores_one_and_reversed_oracle_scores_less():
    qrels = {"q1": {"a": 2, "b": 1, "c": 0}, "q2": {"d": 1}}
    filler = [f"x{i}" for i in range(20)]
    oracle = {q: sorted(j, key=lambda d: -j[d]) + filler for q, j in qrels.items()}
    worst = {q: filler + sorted(j, key=lambda d: j[d]) for q, j in qrels.items()}
    assert evaluate(oracle, qrels).mean() == {"nDCG@10": 1.0, "MRR@10": 1.0, "R@100": 1.0, "R@1000": 1.0}
    worse = evaluate(worst, qrels).mean()
    assert worse["nDCG@10"] == 0.0 and worse["R@100"] == 1.0
