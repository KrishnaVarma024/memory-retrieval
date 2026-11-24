"""An inverted index scored with Okapi BM25, written from scratch.

The index is stored the way search engines store it: for every term, the list
of documents containing it (its postings) and how often it occurs in each.
Here the postings of all terms sit end to end in two flat arrays, with an
offsets array marking where each term's run starts (compressed sparse rows).
Scoring a query then touches only the postings of the query's terms, never
the whole collection.

BM25 for a query q and document d:

    score(q, d) = sum over terms t in q of  qtf(t) * idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avglen))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

tf is how often t occurs in d, df(t) how many documents contain t, N the number
of non-empty documents, and qtf(t) how often t occurs in the query. This is
Lucene's variant: its idf is never negative, unlike the textbook form, so a
term in more than half the documents still counts a little instead of
penalising every document that contains it. Lucene drops the constant
(k1 + 1) factor, which scales every score equally and so changes no ranking.

One more Lucene detail matters for matching its numbers: it stores each
document's length in a single byte, exactly up to 23 and with four significant
bits above that, so a 137-term document is scored as if it had 136 terms.
BM25(lengths="lucene") reproduces that; the default uses exact lengths.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    doc: int  # position of the document in the indexed collection
    score: float


class InvertedIndex:
    """Term -> (documents, term frequencies), plus document lengths."""

    def __init__(self, analyzed_docs: Iterable[Sequence[str]]):
        vocab: dict[str, int] = {}
        term_ids: list[int] = []
        doc_ids: list[int] = []
        freqs: list[int] = []
        lengths: list[int] = []
        for doc, terms in enumerate(analyzed_docs):
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                term_ids.append(vocab.setdefault(term, len(vocab)))
                doc_ids.append(doc)
                freqs.append(tf)

        # Group the (term, doc, tf) triples by term. A stable sort keeps each
        # term's documents in ascending order, which the tie-break relies on.
        terms = np.asarray(term_ids, dtype=np.int64)
        order = np.argsort(terms, kind="stable")
        self.postings_doc = np.asarray(doc_ids, dtype=np.int32)[order]
        self.postings_tf = np.asarray(freqs, dtype=np.float64)[order]
        df = np.bincount(terms, minlength=len(vocab))
        self.offsets = np.concatenate(([0], np.cumsum(df)))
        self.vocab = vocab
        self.df = df
        self.doc_len = np.asarray(lengths, dtype=np.float64)

    @property
    def num_docs(self) -> int:
        """Documents with at least one term. Empty ones can never match and, as in Lucene, are not counted."""
        return int(np.count_nonzero(self.doc_len))

    @property
    def total_terms(self) -> int:
        return int(self.doc_len.sum())

    def postings(self, term: str) -> slice | None:
        t = self.vocab.get(term)
        return None if t is None else slice(self.offsets[t], self.offsets[t + 1])


def _long_to_int4(n: int) -> int:
    # Lucene's SmallFloat.longToInt4: keep the 4 most significant bits (the top
    # one implicit) and store the shift above them.
    bits = n.bit_length()
    if bits < 4:
        return n
    shift = bits - 4
    return ((n >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(code: int) -> int:
    shift = (code >> 3) - 1
    return code & 0x07 if shift == -1 else ((code & 0x07) | 0x08) << shift


_FREE = 255 - _long_to_int4(2**31 - 1)  # 24 byte values are spent on exact small lengths


def lucene_length(n: int) -> int:
    """The length Lucene scores a document of n terms with, after its one-byte encoding."""
    if n < 0:
        raise ValueError(f"length must be >= 0, got {n}")
    return n if n < _FREE else _FREE + _int4_to_long(_long_to_int4(n - _FREE))


class BM25:
    """Scores queries against an InvertedIndex. Changing k1 or b needs no re-indexing."""

    def __init__(self, index: InvertedIndex, k1: float = 0.9, b: float = 0.4, lengths: str = "exact"):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        if lengths not in ("exact", "lucene"):
            raise ValueError(f"lengths must be 'exact' or 'lucene', got {lengths!r}")
        self.index, self.k1, self.b, self.lengths = index, k1, b, lengths
        n = index.num_docs
        self.idf = np.log1p((n - index.df + 0.5) / (index.df + 0.5))
        avg_len = index.total_terms / n if n else 0.0
        # Everything except qtf depends only on the document and the term, so it is
        # computed once per posting here rather than once per query.
        doc_len = index.doc_len
        if lengths == "lucene":  # the average stays exact, as in Lucene
            doc_len = np.array([lucene_length(int(x)) for x in doc_len], dtype=np.float64)
        tf = index.postings_tf
        norm = k1 * (1 - b + b * doc_len[index.postings_doc] / avg_len) if n else 0.0
        term_of_posting = np.repeat(np.arange(len(index.df)), index.df)
        self.weights = self.idf[term_of_posting] * tf * (k1 + 1) / (tf + norm)

    def scores(self, query_terms: Sequence[str]) -> np.ndarray:
        """BM25 score of every document; documents sharing no term with the query score 0."""
        scores = np.zeros(len(self.index.doc_len))
        for term, qtf in Counter(query_terms).items():
            span = self.index.postings(term)
            if span is not None:
                # A term's postings name each document at most once, so this
                # fancy-indexed += never loses an update.
                scores[self.index.postings_doc[span]] += qtf * self.weights[span]
        return scores

    def search(self, query_terms: Sequence[str], k: int = 1000) -> list[Hit]:
        """The k best documents, best first. Ties go to the earlier document, as in Lucene."""
        scores = self.scores(query_terms)
        matched = np.flatnonzero(scores > 0)
        if len(matched) > k:
            # argpartition finds the k largest in linear time, but cuts ties at the
            # boundary arbitrarily; keep every document tied with the k-th score.
            kth = np.partition(scores[matched], len(matched) - k)[len(matched) - k]
            matched = matched[scores[matched] >= kth]
        # lexsort sorts by the last key first: score descending, then document position.
        order = np.lexsort((matched, -scores[matched]))[:k]
        return [Hit(int(d), float(scores[d])) for d in matched[order]]
