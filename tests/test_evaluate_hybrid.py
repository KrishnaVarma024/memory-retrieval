import numpy as np
import pytest

from memret.evaluate_hybrid import DEFAULT_RRF, effective_dense_share, fusion_grid, holm
from memret.hybrid import FusionConfig, candidate_set


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
