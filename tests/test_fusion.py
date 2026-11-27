import numpy as np
import pytest

from memret.fusion import combine, normalise, order, rrf

INF = np.inf


def test_rrf_matches_the_formula_by_hand():
    ranks = np.array([[1, 2, INF], [3, INF, 1]])  # two retrievers, three candidates
    np.testing.assert_allclose(rrf(ranks, k=60), [1 / 61 + 1 / 63, 1 / 62, 1 / 61])


def test_rrf_weights_scale_each_retriever():
    ranks = np.array([[1, 2, INF], [3, INF, 1]])
    np.testing.assert_allclose(rrf(ranks, k=60, weights=[0, 1]), [1 / 63, 0, 1 / 61])
    np.testing.assert_allclose(rrf(ranks, k=60, weights=[2, 2]), 2 * rrf(ranks, k=60))


def test_rrf_k_decides_whether_agreement_beats_one_first_place():
    # A is first for one retriever and missing from the other; B is second for both.
    ranks = np.array([[1, 2], [INF, 2]])
    a, b = rrf(ranks, k=0)
    assert a == b  # k = 0: 1/1 against 1/2 + 1/2, an exact tie
    a, b = rrf(ranks, k=60)
    assert b > a  # k = 60: agreement wins


def test_rrf_rejects_bad_input():
    with pytest.raises(ValueError):
        rrf(np.array([1, 2]))  # needs (retrievers, candidates)
    with pytest.raises(ValueError):
        rrf(np.array([[0, 1]]))  # ranks count from 1
    with pytest.raises(ValueError):
        rrf(np.array([[1, 2]]), k=-1)
    with pytest.raises(ValueError):
        rrf(np.array([[1, 2]]), weights=[1, 1])
    with pytest.raises(ValueError):
        rrf(np.array([[1, 2]]), weights=[-1])


def test_minmax_maps_to_zero_one():
    np.testing.assert_allclose(normalise([2, 4, 6], "minmax"), [0, 0.5, 1])


def test_theoretical_minmax_uses_the_floor_not_the_lowest_candidate():
    np.testing.assert_allclose(normalise([2, 4], "tmm", lower=0), [0.5, 1])
    np.testing.assert_allclose(normalise([0.5, 1.0], "tmm", lower=-1), [0.75, 1])
    with pytest.raises(ValueError):
        normalise([-0.5], "tmm", lower=0)  # below the floor: the floor is wrong
    with pytest.raises(ValueError):
        normalise([1.0], "tmm")


def test_zscore_has_mean_zero_and_unit_spread():
    z = normalise([1.0, 2.0, 3.0, 10.0], "zscore")
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_constant_scores_normalise_to_zero_instead_of_dividing_by_zero(method):
    np.testing.assert_array_equal(normalise([3.0, 3.0, 3.0], method), [0, 0, 0])


def test_tmm_with_every_score_at_the_floor_is_zero():
    np.testing.assert_array_equal(normalise([0.0, 0.0], "tmm", lower=0), [0, 0])


def test_normalise_edge_cases():
    assert normalise([], "minmax").size == 0
    np.testing.assert_array_equal(normalise([5.0, 1.0], "none"), [5, 1])
    with pytest.raises(ValueError):
        normalise([1.0], "softmax")


def test_combine_is_a_weighted_sum_of_normalised_scores():
    scores = np.array([[10.0, 20.0, 30.0], [0.9, 0.7, 0.8]])
    expected = 0.25 * np.array([0, 0.5, 1]) + 0.75 * np.array([1, 0, 0.5])
    np.testing.assert_allclose(combine(scores, [0.25, 0.75], "minmax"), expected)


def test_combine_at_the_endpoints_reproduces_each_retriever():
    candidates = np.array([4, 7, 9])
    scores = np.array([[10.0, 30.0, 20.0], [0.9, 0.7, 0.8]])
    assert order(candidates, combine(scores, [1, 0], "minmax")).tolist() == [7, 9, 4]
    assert order(candidates, combine(scores, [0, 1], "minmax")).tolist() == [4, 9, 7]


def test_raw_scores_let_the_larger_scale_decide():
    candidates = np.array([0, 1, 2])
    scores = np.array([[12.0, 11.0, 2.0], [0.30, 0.95, 0.20]])  # BM25 mildly prefers 0, dense strongly 1
    assert order(candidates, combine(scores, [1, 1], "none")).tolist() == [0, 1, 2]
    assert order(candidates, combine(scores, [1, 1], "minmax")).tolist() == [1, 0, 2]


def test_combine_rejects_bad_shapes():
    with pytest.raises(ValueError):
        combine(np.array([1.0, 2.0]), [1], "minmax")
    with pytest.raises(ValueError):
        combine(np.ones((2, 3)), [1, 1], "tmm", lowers=[0])


def test_order_breaks_ties_towards_the_lower_position_and_truncates():
    candidates = np.array([9, 3, 5, 1])
    fused = np.array([0.5, 0.5, 0.9, 0.1])
    assert order(candidates, fused).tolist() == [5, 3, 9, 1]
    assert order(candidates, fused, k=2).tolist() == [5, 3]
    with pytest.raises(ValueError):
        order(candidates, fused[:3])
