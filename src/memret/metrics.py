"""Ranking metrics, computed per query.

Definitions follow trec_eval, the tool behind the published BEIR numbers:
graded relevance as the gain in nDCG, a log2 position discount, and a document
counts as relevant when its judgement is above zero. Every function returns one
value per query; averaging, and comparing two systems, is left to the caller,
because the per-query values are what significance tests need.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]  # query id -> {doc id: relevance}
Run = Mapping[str, Sequence[str]]  # query id -> doc ids, best first


def ndcg_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain of the top k, divided by the best achievable."""
    dcg = sum(judged.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sorted((r for r in judged.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(rank + 2) for rank, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    relevant = {doc for doc, r in judged.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    """1 / rank of the first relevant document in the top k, or 0 if there is none."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if judged.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


METRICS = {
    "nDCG@10": lambda ranked, judged: ndcg_at_k(ranked, judged, 10),
    "MRR@10": lambda ranked, judged: mrr_at_k(ranked, judged, 10),
    "R@100": lambda ranked, judged: recall_at_k(ranked, judged, 100),
    "R@1000": lambda ranked, judged: recall_at_k(ranked, judged, 1000),
}


@dataclass(frozen=True)
class Evaluation:
    query_ids: list[str]  # sorted; the order of every per-query array
    per_query: dict[str, np.ndarray]  # metric name -> value per query

    def mean(self) -> dict[str, float]:
        return {name: float(values.mean()) for name, values in self.per_query.items()}


def evaluate(run: Run, qrels: Qrels) -> Evaluation:
    """Score a run against judgements.

    Averages over every judged query with at least one relevant document,
    including queries the run returned nothing for (they score 0), like
    trec_eval's -c flag. Leaving those out would reward a system for failing
    silently on hard queries.
    """
    query_ids = sorted(q for q, judged in qrels.items() if any(r > 0 for r in judged.values()))
    per_query = {
        name: np.array([metric(run.get(q, ()), qrels[q]) for q in query_ids]) for name, metric in METRICS.items()
    }
    return Evaluation(query_ids, per_query)
