import numpy as np
import pytest

from memret.bm25 import BM25, InvertedIndex
from memret.dense import ExactIndex, top_k_rows
from memret.hybrid import CandidateSet, FusionConfig, HybridSearcher, candidate_set, fuse, top_depth
from memret.text import DEFAULT

DOCS = [
    "heart rate and pulse",
    "lower your mortgage payment",
    "pulse oximeters measure blood oxygen",
    "ways to reduce heart rate",
    "cats and dogs",
    "the heart of the city",
    "resting pulse rate in athletes",
    "mortgage rates and the heart of lending",
]


@pytest.fixture(scope="module")
def searcher():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((len(DOCS), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bm25 = BM25(InvertedIndex([DEFAULT(d) for d in DOCS]))
    return HybridSearcher(bm25, DEFAULT, ExactIndex(vectors))


@pytest.fixture(scope="module")
def query_vector():
    v = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_top_depth_follows_each_retrievers_own_rules(searcher, query_vector):
    terms = DEFAULT("heart rate pulse")
    bm25_scores = searcher.bm25.scores(terms)
    expected = [h.doc for h in searcher.bm25.search(terms, 3)]
    assert top_depth(bm25_scores, 3, positive_only=True).tolist() == expected
    dense_scores = searcher.dense_scores(query_vector)
    ids, _ = top_k_rows(dense_scores, 3)
    assert top_depth(dense_scores[0], 3).tolist() == ids[0].tolist()


def test_top_depth_breaks_ties_towards_the_lower_position():
    assert top_depth(np.array([1.0, 2.0, 2.0, 2.0]), 2).tolist() == [1, 2]
    assert top_depth(np.array([0.0, 0.0, 1.0]), 5, positive_only=True).tolist() == [2]


def test_candidates_hold_every_score_and_rank(searcher, query_vector):
    c = searcher.candidates("heart rate", query_vector, depth=3)
    bm25_scores = searcher.bm25.scores(DEFAULT("heart rate"))
    dense_scores = searcher.dense_scores(query_vector)[0]
    assert np.all(np.diff(c.docs) > 0)  # ascending positions
    np.testing.assert_allclose(c.scores[0], bm25_scores[c.docs])
    np.testing.assert_allclose(c.scores[1], dense_scores[c.docs], rtol=1e-6)
    for r, scores in enumerate((bm25_scores, dense_scores)):
        ranked = top_depth(scores, 3, positive_only=(r == 0))
        assert c.docs[np.argsort(c.ranks[r])][: len(ranked)].tolist() == ranked.tolist()
        assert np.isfinite(c.ranks[r]).sum() == len(ranked)


def test_truncating_equals_searching_with_the_smaller_depth(searcher, query_vector):
    bm25_scores = searcher.bm25.scores(DEFAULT("heart rate pulse mortgage"))
    dense_scores = searcher.dense_scores(query_vector)[0]
    deep = candidate_set(bm25_scores, dense_scores, 6).truncate(2)
    shallow = candidate_set(bm25_scores, dense_scores, 2)
    for field in ("docs", "scores", "ranks"):
        np.testing.assert_array_equal(getattr(deep, field), getattr(shallow, field))
    with pytest.raises(ValueError):
        shallow.truncate(6)


def test_single_retriever_configs_reproduce_each_search(searcher, query_vector):
    c = searcher.candidates("heart rate pulse", query_vector, depth=8)
    expected_bm25 = [h.doc for h in searcher.bm25.search(DEFAULT("heart rate pulse"), 8)]
    assert fuse(c, FusionConfig("bm25", depth=8), 8).tolist() == expected_bm25
    ids, _ = searcher.dense.search(query_vector, 8)
    assert fuse(c, FusionConfig("dense", depth=8), 8).tolist() == ids[0].tolist()


def test_convex_endpoints_put_each_retrievers_list_first(searcher, query_vector):
    c = searcher.candidates("heart rate pulse", query_vector, depth=8)
    bm25_list = fuse(c, FusionConfig("bm25", depth=8), 8)
    dense_list = fuse(c, FusionConfig("dense", depth=8), 8)
    alpha0 = fuse(c, FusionConfig("convex", depth=8, weight=0.0), 8)
    alpha1 = fuse(c, FusionConfig("convex", depth=8, weight=1.0), 8)
    assert alpha0[: len(bm25_list)].tolist() == bm25_list.tolist()
    assert alpha1.tolist() == dense_list.tolist()


def test_rrf_on_real_candidates_matches_the_formula(searcher, query_vector):
    c = searcher.candidates("heart rate pulse", query_vector, depth=8)
    fused = (0.5 / (60 + c.ranks[0]) + 0.5 / (60 + c.ranks[1]))
    expected = c.docs[np.lexsort((c.docs, -fused))][:5]
    assert fuse(c, FusionConfig("rrf", depth=8), 5).tolist() == expected.tolist()


def test_a_query_bm25_cant_match_falls_back_to_dense(searcher, query_vector):
    c = searcher.candidates("zebra", query_vector, depth=4)  # no document contains it
    assert not np.isfinite(c.ranks[0]).any()
    dense_list = fuse(c, FusionConfig("dense", depth=4), 4).tolist()
    for config in (FusionConfig("rrf", depth=4), FusionConfig("convex", depth=4, norm="tmm"),
                   FusionConfig("convex", depth=4, norm="zscore")):
        assert fuse(c, config, 4).tolist() == dense_list


def test_search_is_candidates_then_fuse(searcher, query_vector):
    config = FusionConfig("convex", depth=5, weight=0.6)
    expected = fuse(searcher.candidates("mortgage rate", query_vector, 5), config, 3)
    assert searcher.search("mortgage rate", query_vector, config, k=3).tolist() == expected.tolist()


def test_chunked_documents_score_as_their_best_chunk():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]], dtype=np.float32)
    bm25 = BM25(InvertedIndex([DEFAULT(d) for d in ("a b", "c d", "e f")]))
    searcher = HybridSearcher(bm25, DEFAULT, ExactIndex(vectors), chunk_doc=np.array([0, 0, 1, 2]))
    np.testing.assert_allclose(searcher.dense_scores(np.array([0, 1.0]))[0], [1.0, 0.8, 0.0])
    with pytest.raises(ValueError):
        HybridSearcher(bm25, DEFAULT, ExactIndex(vectors), chunk_doc=np.array([0, 1, 0, 2]))  # out of order
    with pytest.raises(ValueError):
        HybridSearcher(bm25, DEFAULT, ExactIndex(vectors), chunk_doc=np.array([0, 0, 0, 2]))  # doc 1 has none
    with pytest.raises(ValueError):
        HybridSearcher(bm25, DEFAULT, ExactIndex(vectors))  # four vectors, three documents


def test_config_validation_and_labels():
    with pytest.raises(ValueError):
        FusionConfig("borda")
    with pytest.raises(ValueError):
        FusionConfig("rrf", depth=0)
    with pytest.raises(ValueError):
        FusionConfig("convex", weight=1.5)
    with pytest.raises(ValueError):
        FusionConfig("convex", norm="softmax")
    assert FusionConfig("rrf").label() == "RRF k=60, dense weight 0.5, depth 1000"
    assert FusionConfig("convex", depth=100, weight=0.7, norm="tmm").label() == "convex tmm, alpha 0.7, depth 100"


def test_candidate_set_rejects_mismatched_scores():
    with pytest.raises(ValueError):
        candidate_set(np.zeros(3), np.zeros(4), 2)
    assert isinstance(candidate_set(np.zeros(3), np.zeros(3), 2), CandidateSet)
