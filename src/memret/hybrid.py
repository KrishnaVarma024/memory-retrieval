"""Hybrid retrieval: BM25 and dense search, fused into one ranking.

For each query, both retrievers score every document: BM25 from its inverted
index, dense search with one matrix product. Each contributes its `depth` best
documents as candidates. Because both searches are exact, every candidate's
score under *both* retrievers is known, even for a document only one of them
returned, so a score-based fusion never has to guess a missing score. Fusion
then reorders the candidates (fusion.py).

    query ─┬─ analyze ─ BM25 scores ─── top depth ─┐
           │                                        ├─ union ─ CandidateSet ─ fuse(config) ─ top k
           └─ vector ── dense scores ── top depth ─┘

Candidates and fusion are kept apart so an evaluation can collect each query's
candidates once and then try hundreds of fusion settings on them cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .dense import ExactIndex
from .fusion import NORMALISATIONS, combine, order, rrf
from .text import Analyzer

RETRIEVERS = ("bm25", "dense")
THEORETICAL_MIN = (0.0, -1.0)  # BM25 is never negative; cosine similarity is at least -1
METHODS = ("bm25", "dense", "rrf", "convex", "raw_sum")


@dataclass(frozen=True)
class FusionConfig:
    """How to turn a query's candidates into one ranking.

    method: "bm25" or "dense" (one retriever alone), "rrf", "convex", or
    "raw_sum" (BM25 and cosine added unnormalised, kept to show why
    normalisation matters). weight is dense's share: RRF weighs the two
    retrievers 1 - weight and weight, and convex uses it as alpha. Only the
    ratio matters for RRF, so 0.5 means equal weights.
    """

    method: str
    depth: int = 1000
    k: float = 60.0
    weight: float = 0.5
    norm: str = "minmax"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if not 0 <= self.weight <= 1:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        if self.norm not in NORMALISATIONS:
            raise ValueError(f"unknown normalisation {self.norm!r}; choose from {NORMALISATIONS}")

    def label(self) -> str:
        if self.method == "rrf":
            return f"RRF k={self.k:g}, dense weight {self.weight:g}, depth {self.depth}"
        if self.method == "convex":
            return f"convex {self.norm}, alpha {self.weight:g}, depth {self.depth}"
        return f"{self.method}, depth {self.depth}"


# The one setting Phase 3 chose on development queries for every dataset
# (reports/phase3/hybrid.md). alpha belongs to bge-small's range of cosines:
# re-tune it if the embedding model changes.
SHIPPED = FusionConfig("convex", depth=100, weight=0.95, norm="tmm")


@dataclass(frozen=True)
class CandidateSet:
    """One query's candidates: the union of each retriever's top `depth`.

    docs are document positions in ascending order. scores[0] and scores[1]
    are every candidate's BM25 and dense scores. ranks are 1-based positions
    in each retriever's own ranking, np.inf where the candidate is outside
    that retriever's top `depth`.
    """

    docs: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray
    depth: int

    def truncate(self, depth: int) -> CandidateSet:
        """The candidates a smaller depth would have produced, without searching again."""
        if depth > self.depth:
            raise ValueError(f"can't deepen candidates from {self.depth} to {depth}")
        if depth == self.depth:
            return self
        ranks = np.where(self.ranks <= depth, self.ranks, np.inf)
        keep = np.isfinite(ranks).any(axis=0)
        return CandidateSet(self.docs[keep], self.scores[:, keep], ranks[:, keep], depth)


def top_depth(scores: np.ndarray, depth: int, *, positive_only: bool = False) -> np.ndarray:
    """Positions of the `depth` best documents, best first, ties to the lower position.

    The same rule as BM25.search and dense.top_k_rows. positive_only keeps
    BM25's convention that a document sharing no term with the query isn't returned.
    """
    pool = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(pool) > depth:
        values = scores[pool]
        kth = np.partition(values, len(pool) - depth)[len(pool) - depth]
        pool = pool[values >= kth]  # keep everything tied with the depth-th score
    return pool[np.lexsort((pool, -scores[pool]))][:depth]


def candidate_set(bm25_scores: np.ndarray, dense_scores: np.ndarray, depth: int) -> CandidateSet:
    """Candidates for one query from each retriever's score of every document."""
    if bm25_scores.shape != dense_scores.shape or bm25_scores.ndim != 1:
        raise ValueError(f"need one score per document from each retriever, got {bm25_scores.shape} "
                         f"and {dense_scores.shape}")
    lists = (top_depth(bm25_scores, depth, positive_only=True), top_depth(dense_scores, depth))
    docs = np.union1d(*lists).astype(np.int64)
    ranks = np.full((2, len(docs)), np.inf)
    for r, ranked in enumerate(lists):
        ranks[r, np.searchsorted(docs, ranked)] = np.arange(1, len(ranked) + 1)
    scores = np.vstack([bm25_scores[docs], dense_scores[docs]]).astype(np.float64)
    return CandidateSet(docs, scores, ranks, depth)


def fuse(candidates: CandidateSet, config: FusionConfig, k: int) -> np.ndarray:
    """The k best document positions for one query under one fusion setting."""
    c = candidates.truncate(config.depth)
    if config.method in RETRIEVERS:
        r = RETRIEVERS.index(config.method)
        mine = np.isfinite(c.ranks[r])
        return c.docs[mine][np.argsort(c.ranks[r][mine], kind="stable")][:k]
    if config.method == "rrf":
        fused = rrf(c.ranks, config.k, (1 - config.weight, config.weight))
    elif config.method == "convex":
        fused = combine(c.scores, (1 - config.weight, config.weight), config.norm, THEORETICAL_MIN)
    else:  # raw_sum
        fused = combine(c.scores, (1.0, 1.0), "none")
    return order(c.docs, fused, k)


class HybridSearcher:
    """BM25 and dense search over one collection, fused per query.

    chunk_doc maps each stored vector to its document's position, for chunked
    collections; a document then scores as its best chunk. Omit it when there
    is one vector per document.
    """

    def __init__(self, bm25: BM25, analyzer: Analyzer, dense: ExactIndex, chunk_doc: np.ndarray | None = None):
        self.bm25, self.analyzer, self.dense = bm25, analyzer, dense
        self.n_docs = len(bm25.index.doc_len)
        if chunk_doc is None:
            if len(dense) != self.n_docs:
                raise ValueError(f"{len(dense)} vectors for {self.n_docs} documents; pass chunk_doc")
            self._starts = None
        else:
            chunk_doc = np.asarray(chunk_doc)
            if len(chunk_doc) != len(dense) or np.any(np.diff(chunk_doc) < 0):
                raise ValueError("chunk_doc needs one entry per vector, in document order")
            if not np.array_equal(np.unique(chunk_doc), np.arange(self.n_docs)):
                raise ValueError("every document needs at least one vector")
            self._starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])

    def dense_scores(self, query_vectors: np.ndarray) -> np.ndarray:
        """Every document's dense score for each query: shape (queries, documents)."""
        s = self.dense.scores(np.atleast_2d(query_vectors))
        return s if self._starts is None else np.maximum.reduceat(s, self._starts, axis=1)

    def candidates(self, query_text: str, query_vector: np.ndarray, depth: int = 1000) -> CandidateSet:
        bm25_scores = self.bm25.scores(self.analyzer(query_text))
        return candidate_set(bm25_scores, self.dense_scores(query_vector)[0], depth)

    def search(self, query_text: str, query_vector: np.ndarray, config: FusionConfig, k: int = 10) -> np.ndarray:
        """The k best document positions for one query."""
        return fuse(self.candidates(query_text, query_vector, config.depth), config, k)
