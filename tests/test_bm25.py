import math

import numpy as np
import pytest

from memret.bm25 import BM25, InvertedIndex, lucene_length

DOCS = [["a", "b"], ["a", "a", "c"], ["b", "c", "c", "d"], ["e"]]


def reference_score(docs, query, k1, b):
    """BM25 written as directly from the formula as possible, for comparison."""
    nonempty = [d for d in docs if d]
    n = len(nonempty)
    avg = sum(len(d) for d in nonempty) / n
    out = []
    for doc in docs:
        score = 0.0
        for term in set(query):
            tf = doc.count(term)
            if tf == 0:
                continue
            df = sum(term in d for d in docs)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += query.count(term) * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))
        out.append(score)
    return out


@pytest.mark.parametrize("query", [["a"], ["c"], ["a", "c"], ["c", "c", "d"], ["a", "b", "c", "d", "e"]])
@pytest.mark.parametrize("k1,b", [(0.9, 0.4), (1.2, 0.75), (0.0, 0.5), (2.0, 0.0), (1.5, 1.0)])
def test_scores_match_the_formula(query, k1, b):
    bm25 = BM25(InvertedIndex(DOCS), k1=k1, b=b)
    assert np.allclose(bm25.scores(query), reference_score(DOCS, query, k1, b))


def test_postings_are_grouped_by_term_in_document_order():
    index = InvertedIndex(DOCS)
    assert index.postings_doc[index.postings("c")].tolist() == [1, 2]
    assert index.postings_tf[index.postings("c")].tolist() == [1, 2]
    assert index.df[index.vocab["a"]] == 2
    assert index.postings("zzz") is None


def test_idf_is_positive_even_for_a_term_in_every_document():
    bm25 = BM25(InvertedIndex([["x", "y"], ["x"], ["x", "z"]]))
    assert bm25.idf[bm25.index.vocab["x"]] > 0


def test_without_length_normalisation_length_does_not_matter():
    index = InvertedIndex([["a", "b"], ["a", "b", "c", "d", "e", "f"]])
    s = BM25(index, b=0.0).scores(["a"])
    assert s[0] == pytest.approx(s[1])
    s = BM25(index, b=0.75).scores(["a"])
    assert s[0] > s[1]


def test_term_frequency_saturates():
    index = InvertedIndex([["a"] * tf + ["x"] * (100 - tf) for tf in (1, 2, 10, 100)] + [["y"]])
    bm25 = BM25(index, k1=1.2, b=0.0)
    s = bm25.scores(["a"])[:4]
    assert all(np.diff(s) > 0)
    assert s[-1] < bm25.idf[index.vocab["a"]] * (1.2 + 1)  # the ceiling


def test_empty_documents_do_not_count():
    with_empty = BM25(InvertedIndex([["a"], [], ["b"], []]))
    without = BM25(InvertedIndex([["a"], ["b"]]))
    assert with_empty.index.num_docs == 2
    assert with_empty.scores(["a"])[[0, 2]] == pytest.approx(without.scores(["a"]))


def test_unknown_and_empty_queries_return_nothing():
    bm25 = BM25(InvertedIndex(DOCS))
    assert bm25.search(["zzz"]) == []
    assert bm25.search([]) == []


def test_ties_go_to_the_earlier_document_and_only_matches_are_returned():
    bm25 = BM25(InvertedIndex([["x"], ["a"], ["a"], ["a"], ["y"]]))
    assert [h.doc for h in bm25.search(["a"], k=10)] == [1, 2, 3]
    assert [h.doc for h in bm25.search(["a"], k=2)] == [1, 2]


def test_search_is_sorted_and_cut_at_k():
    rng = np.random.default_rng(0)
    docs = [list(rng.choice(list("abcdefgh"), size=rng.integers(1, 20))) for _ in range(500)]
    bm25 = BM25(InvertedIndex(docs))
    hits = bm25.search(["a", "c", "c"], k=50)
    assert len(hits) == 50
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    full = bm25.scores(["a", "c", "c"])
    assert scores[0] == pytest.approx(full.max())
    assert scores[-1] == pytest.approx(np.sort(full)[-50])


def test_lucene_length_encoding():
    assert [lucene_length(n) for n in (0, 10, 23, 24)] == [0, 10, 23, 24]
    assert [lucene_length(n) for n in (100, 137, 250, 1000)] == [96, 136, 248, 984]
    values = [lucene_length(n) for n in range(5000)]
    assert all(v <= n for n, v in enumerate(values))
    assert values == sorted(values)  # never reorders documents by length


def test_lucene_lengths_change_scores_only_for_long_documents():
    index = InvertedIndex([["a"] * 5, ["a"] + ["b"] * 136, ["c"]])
    exact, lucene = BM25(index).scores(["a"]), BM25(index, lengths="lucene").scores(["a"])
    assert exact[0] == pytest.approx(lucene[0])
    assert exact[1] != pytest.approx(lucene[1])


def test_rejects_bad_parameters():
    index = InvertedIndex(DOCS)
    with pytest.raises(ValueError):
        BM25(index, k1=-1)
    with pytest.raises(ValueError):
        BM25(index, b=1.5)
    with pytest.raises(ValueError):
        BM25(index, lengths="approximate")
