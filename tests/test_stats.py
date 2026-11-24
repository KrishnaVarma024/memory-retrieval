import numpy as np
import pytest

from memret.stats import compare, paired_bootstrap_ci, paired_randomization_p


def test_identical_systems():
    a = np.linspace(0, 1, 40)
    c = compare(a, a.copy())
    assert c.difference == 0 and (c.ci_low, c.ci_high) == (0, 0)
    assert c.p_value == 1.0


def test_consistently_better_system_is_detected():
    b = np.random.default_rng(1).uniform(0, 0.8, 50)
    c = compare(b + 0.1, b)
    assert c.difference == pytest.approx(0.1)
    assert c.ci_low == pytest.approx(0.1) and c.ci_high == pytest.approx(0.1)
    assert c.p_value < 0.001


def test_noise_is_not_significant():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(size=200), rng.uniform(size=200)
    c = compare(a, b)
    assert c.ci_low < 0 < c.ci_high or c.p_value > 0.01


def test_interval_contains_the_observed_difference_and_is_seeded():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(size=100), rng.uniform(size=100)
    low, high = paired_bootstrap_ci(a, b, seed=7)
    assert low <= np.mean(a - b) <= high
    assert (low, high) == paired_bootstrap_ci(a, b, seed=7)
    assert paired_randomization_p(a, b, seed=7) == paired_randomization_p(a, b, seed=7)


def test_rejects_unpaired_input():
    with pytest.raises(ValueError):
        compare(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        compare(np.zeros(0), np.zeros(0))
