import numpy as np
import pytest

from memret.dense import ExactIndex, HNSWIndex, best_chunk_per_document, top_k_rows


def unit(rows, seed=0, dim=16):
    v = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_top_k_breaks_ties_towards_the_lower_index():
    scores = np.array([[0.2, 0.9, 0.5, 0.9, 0.5, 0.5]], dtype=np.float32)
    ids, sc = top_k_rows(scores, 3)
    assert ids.tolist() == [[1, 3, 2]]
    assert sc[0].tolist() == pytest.approx([0.9, 0.9, 0.5])


def test_top_k_keeps_every_tie_at_the_boundary_before_choosing():
    scores = np.array([[0.5] * 10 + [0.7]], dtype=np.float32)
    ids, _ = top_k_rows(scores, 4)
    assert ids.tolist() == [[10, 0, 1, 2]]


def test_top_k_pads_when_there_are_fewer_than_k_columns():
    ids, sc = top_k_rows(np.array([[0.1, 0.3]], dtype=np.float32), 4)
    assert ids.tolist() == [[1, 0, -1, -1]]
    assert np.isneginf(sc[0, 2:]).all()


def test_exact_search_matches_sorting_every_score():
    vectors, queries = unit(500), unit(20, seed=1)
    ids, sc = ExactIndex(vectors).search(queries, 10, block=7)
    full = queries @ vectors.T
    for r in range(len(queries)):
        order = sorted(range(500), key=lambda j: (-full[r, j], j))[:10]
        assert ids[r].tolist() == order
        np.testing.assert_allclose(sc[r], full[r, order], rtol=1e-6)


def test_exact_search_agrees_with_faiss_flat():
    faiss = pytest.importorskip("faiss")
    vectors, queries = unit(1000, dim=32), unit(30, seed=2, dim=32)
    flat = faiss.IndexFlatIP(32)
    flat.add(vectors)
    f_scores, f_ids = flat.search(queries, 10)
    ids, sc = ExactIndex(vectors).search(queries, 10)
    assert (ids == f_ids).all()  # random data has no exact ties
    np.testing.assert_allclose(sc, f_scores, rtol=1e-5, atol=1e-6)


def test_search_documents_scores_each_document_by_its_best_chunk():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    chunk_doc = np.array([0, 0, 1, 2])
    ids, sc = ExactIndex(vectors).search_documents(np.array([[0, 1]], dtype=np.float32), chunk_doc, 3)
    assert ids.tolist() == [[0, 1, 2]]
    assert sc[0].tolist() == pytest.approx([1.0, 0.8, 0.6])


@pytest.mark.parametrize("chunk_doc", [[0, 1, 0], [1, 1, 2], [0, 0, 2], [0, 1]])
def test_search_documents_rejects_bad_chunk_maps(chunk_doc):
    with pytest.raises(ValueError):
        ExactIndex(unit(3)).search_documents(unit(1), np.array(chunk_doc), 1)


def test_search_documents_equals_chunk_search_then_best_chunk():
    vectors, queries = unit(300), unit(10, seed=3)
    chunk_doc = np.repeat(np.arange(100), 3)
    index = ExactIndex(vectors)
    doc_ids, _ = index.search_documents(queries, chunk_doc, 5)
    chunk_ids, chunk_sc = index.search(queries, 300)
    via_chunks, _ = best_chunk_per_document(chunk_ids, chunk_sc, chunk_doc, 5)
    assert (doc_ids == via_chunks).all()


def test_best_chunk_per_document_pads_when_hits_run_out():
    ids, _ = best_chunk_per_document(np.array([[4, 5, -1]]), np.array([[0.9, 0.8, -np.inf]]),
                                     np.array([0, 0, 1, 1, 2, 2]), 3)
    assert ids.tolist() == [[2, -1, -1]]


def test_hnsw_finds_the_exact_neighbours_when_it_explores_enough():
    pytest.importorskip("faiss")
    vectors, queries = unit(2000, dim=32), unit(50, seed=4, dim=32)
    exact, _ = ExactIndex(vectors).search(queries, 10)
    approx, _ = HNSWIndex(vectors, m=16, ef_construction=100).search(queries, 10, ef_search=256)
    recall = np.mean([len(set(a) & set(e)) / 10 for a, e in zip(approx, exact)])
    assert recall > 0.99


def test_hnsw_is_built_reproducibly():
    pytest.importorskip("faiss")
    vectors, queries = unit(1000, dim=16), unit(20, seed=5, dim=16)
    a, _ = HNSWIndex(vectors, m=8, ef_construction=40).search(queries, 10, ef_search=16)
    b, _ = HNSWIndex(vectors, m=8, ef_construction=40).search(queries, 10, ef_search=16)
    assert (a == b).all()
