"""Combining the rankings of several retrievers into one.

Two families, both well studied.

Reciprocal rank fusion (Cormack, Clarke and Büttcher, SIGIR 2009) uses only ranks:

    rrf(d) = sum over retrievers r of  w_r / (k + rank_r(d))

rank_r(d) counts from 1, and a retriever that did not return d contributes
nothing. k damps the advantage of the very top: with k = 60, rank 1 is worth
only 1.6% more than rank 2, so a document that both retrievers place fairly
high beats one that a single retriever places first. RRF never looks at
scores, so it works whatever scale each retriever's scores are on.

A convex combination (studied for hybrid retrieval by Bruch, Gai and Ingber,
2023) uses scores:

    cc(d) = (1 - alpha) * norm(bm25(d)) + alpha * norm(dense(d))

The scores must first be put on one scale, because BM25 is unbounded (often
10 to 40) while cosine similarity lies in [-1, 1]. Added raw, BM25 would
decide almost every ranking alone. Three normalisations are offered:

- min-max: (s - min) / (max - min) over the candidates;
- theoretical min-max ("tmm"): the same, but with the lowest score the
  retriever could ever give (0 for BM25, -1 for cosine) in place of the lowest
  it happened to give, so one odd low candidate can't stretch the scale;
- z-score: (s - mean) / standard deviation over the candidates.

Every function works on one query's candidates, held as arrays aligned by
position. Ties in the final order go to the lower document position: the same
rule BM25 and dense search use, so a ranking never depends on how a sort
happened to order equal scores. RRF produces exact ties often (a document
ranked 3rd by one retriever and missing from the other scores exactly the same
as one ranked 3rd by the other), so the rule matters here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NORMALISATIONS = ("minmax", "tmm", "zscore", "none")


def rrf(ranks: np.ndarray, k: float = 60.0, weights: Sequence[float] | None = None) -> np.ndarray:
    """Reciprocal rank fusion score of every candidate.

    ranks has shape (retrievers, candidates): 1-based ranks, np.inf where a
    retriever did not return the candidate (it then contributes 0).
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be 2-D (retrievers, candidates), got shape {ranks.shape}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if ranks.size and ranks.min() < 1:
        raise ValueError("ranks count from 1")
    w = _weights(weights, len(ranks))
    return (w[:, None] / (k + ranks)).sum(axis=0)


def normalise(scores: np.ndarray, method: str, lower: float | None = None) -> np.ndarray:
    """One retriever's scores for one query's candidates, put on a common scale.

    A constant set of scores carries no ranking information and normalises to
    zeros, rather than dividing by zero.
    """
    s = np.asarray(scores, dtype=np.float64)
    if method not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {method!r}; choose from {NORMALISATIONS}")
    if method == "none" or s.size == 0:
        return s
    if method == "zscore":
        sd = s.std()
        return np.zeros_like(s) if sd == 0 else (s - s.mean()) / sd
    if method == "tmm":
        if lower is None:
            raise ValueError("tmm needs the retriever's theoretical minimum score")
        if s.min() < lower - 1e-9:
            raise ValueError(f"a score of {s.min()} is below the theoretical minimum {lower}")
        lo = lower
    else:
        lo = s.min()
    span = s.max() - lo
    return np.zeros_like(s) if span <= 0 else (s - lo) / span


def combine(scores: np.ndarray, weights: Sequence[float], method: str,
            lowers: Sequence[float | None] | None = None) -> np.ndarray:
    """Weighted sum of each retriever's normalised scores. scores has shape (retrievers, candidates)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (retrievers, candidates), got shape {scores.shape}")
    w = _weights(weights, len(scores))
    lowers = [None] * len(scores) if lowers is None else list(lowers)
    if len(lowers) != len(scores):
        raise ValueError(f"need one lower bound per retriever, got {len(lowers)} for {len(scores)}")
    total = np.zeros(scores.shape[1])
    for wi, s, lo in zip(w, scores, lowers):
        total += wi * normalise(s, method, lo)
    return total


def order(candidates: np.ndarray, fused: np.ndarray, k: int | None = None) -> np.ndarray:
    """Candidates sorted by fused score, best first; equal scores go to the lower document position."""
    candidates = np.asarray(candidates)
    fused = np.asarray(fused, dtype=np.float64)
    if candidates.shape != fused.shape or candidates.ndim != 1:
        raise ValueError(f"need two 1-D arrays of equal length, got {candidates.shape} and {fused.shape}")
    ranked = candidates[np.lexsort((candidates, -fused))]
    return ranked if k is None else ranked[:k]


def _weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"need {n} weights, got {w.shape}")
    if (w < 0).any():
        raise ValueError("weights must be >= 0")
    return w
