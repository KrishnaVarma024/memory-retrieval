import os

import numpy as np
import pytest

import memret.cache as cache_module
from memret.cache import EmbeddingCache


class Counter:
    """A fake embedder that records every batch it is asked to compute."""

    def __init__(self, dim=3):
        self.dim, self.batches = dim, []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return np.array([[len(t), ord(t[0]) if t else 0, 1.0] for t in texts], dtype=np.float32)


def test_hits_misses_and_duplicates_within_a_call():
    fake = Counter()
    cache = EmbeddingCache("model@rev", 3)
    out = cache.get_many(["a", "bb", "a"], fake)
    assert fake.batches == [["a", "bb"]]  # "a" computed once
    assert (cache.hits, cache.misses) == (1, 2)
    np.testing.assert_array_equal(out[0], out[2])
    cache.get_many(["bb", "ccc"], fake)
    assert fake.batches[-1] == ["ccc"]
    assert cache.stats().hit_rate == pytest.approx(2 / 5)


def test_results_come_back_in_the_order_asked():
    cache = EmbeddingCache("m", 3)
    texts = ["zz", "a", "bbb", "a"]
    out = cache.get_many(texts, Counter())
    assert out[:, 0].tolist() == [2, 1, 3, 1]


def test_the_key_depends_on_the_fingerprint():
    assert EmbeddingCache("model@1", 3).key("x") != EmbeddingCache("model@2", 3).key("x")


def test_vectors_survive_a_restart_and_nothing_is_recomputed(tmp_path):
    fake = Counter()
    first = EmbeddingCache("m@1", 3, tmp_path, shard_size=2)
    expected = first.get_many(["a", "b", "c", "d", "e"], fake)
    assert len(list(first.folder.glob("shard-*.npz"))) == 3  # 2 + 2 + 1
    assert not list(first.folder.glob("*.tmp"))

    again = Counter()
    second = EmbeddingCache("m@1", 3, tmp_path)
    np.testing.assert_array_equal(second.get_many(["e", "a"], again), expected[[4, 0]])
    assert again.batches == [] and second.stats().hit_rate == 1.0


def test_two_writers_storing_at_the_same_moment_keep_both_shards(tmp_path, monkeypatch):
    """Writer B stores the same texts while writer A sits between naming its shard and renaming it.

    With shard names built from a count of existing shards, both writers picked the same name
    and A's rename silently replaced B's file.
    """
    texts = ["a", "b"]
    first = EmbeddingCache("m@1", 3, tmp_path)
    second = EmbeddingCache("m@1", 3, tmp_path)
    real_replace = os.replace
    interleaved = []

    def replace(src, dst):
        if not interleaved:
            interleaved.append(True)
            second.get_many(texts, Counter())  # B writes in full before A's rename lands
        real_replace(src, dst)

    monkeypatch.setattr(cache_module.os, "replace", replace)
    first.get_many(texts, Counter())
    assert interleaved
    assert len(list(first.folder.glob("shard-*.npz"))) == 2
    assert all(t in EmbeddingCache("m@1", 3, tmp_path) for t in texts)


def test_a_different_model_never_sees_another_models_vectors(tmp_path):
    EmbeddingCache("m@1", 3, tmp_path).get_many(["a"], Counter())
    other = EmbeddingCache("m@2", 3, tmp_path)
    assert "a" not in other


def test_a_compute_function_returning_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        EmbeddingCache("m", 3).get_many(["a"], lambda texts: np.zeros((len(texts), 4)))


def test_a_shard_with_the_wrong_dimension_is_refused(tmp_path):
    EmbeddingCache("m@1", 3, tmp_path).get_many(["a"], Counter())
    with pytest.raises(ValueError):
        EmbeddingCache("m@1", 4, tmp_path)
