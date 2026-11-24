"""Comparing two systems on the same queries.

Both functions take per-query scores of systems A and B, paired by query. The
unit of resampling is the query, because the question is whether A beats B on
queries like these, not on documents like these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Comparison:
    mean_a: float
    mean_b: float
    difference: float  # mean of A - B
    ci_low: float
    ci_high: float
    p_value: float  # two-sided paired randomization test
    queries: int


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, resamples: int = 10_000, level: float = 0.95,
                        seed: int = 0) -> tuple[float, float]:
    """Percentile interval for mean(A - B), resampling queries with replacement."""
    d = _differences(a, b)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):  # chunks bound memory on large query sets
        stop = min(start + 1000, resamples)
        idx = rng.integers(0, len(d), size=(stop - start, len(d)))
        means[start:stop] = d[idx].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_randomization_p(a: np.ndarray, b: np.ndarray, *, permutations: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for 'A and B are interchangeable on every query'.

    Under that hypothesis each query's difference is as likely to have either
    sign, so we flip signs at random and count how often the mean difference is
    at least as far from zero as the one observed. Smucker, Allan and Carterette
    (CIKM 2007) recommend this test for retrieval experiments.
    """
    d = _differences(a, b)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, 1000):
        stop = min(start + 1000, permutations)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(d)))
        extreme += int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
    # Counting the observed arrangement itself keeps p above zero and the test exact.
    return (extreme + 1) / (permutations + 1)


def compare(a: np.ndarray, b: np.ndarray, *, seed: int = 0) -> Comparison:
    low, high = paired_bootstrap_ci(a, b, seed=seed)
    return Comparison(
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), difference=float(np.mean(a) - np.mean(b)),
        ci_low=low, ci_high=high, p_value=paired_randomization_p(a, b, seed=seed), queries=len(a),
    )


def _differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError(f"need two non-empty 1-D arrays of equal length, got {a.shape} and {b.shape}")
    return a - b
