"""Finding the stored vectors nearest a query vector.

All vectors are unit length, so the dot product of two vectors is their cosine
similarity: 1 for the same direction, 0 for unrelated. Search means "the k
stored vectors with the largest dot product with the query".

ExactIndex does it the obvious way, written here from scratch: one matrix
product against every stored vector, then the top k. Its cost grows linearly
with the number of vectors, but it is never wrong, so it is the reference that
approximate search is measured against.

HNSWIndex uses FAISS's HNSW graph (Malkov and Yashunin, 2016). Each vector is
linked to a few dozen near neighbours, on several layers of decreasing
density. A search starts at the top layer, walks greedily towards the query,
drops a layer, and repeats; on the bottom layer it keeps a list of the
`ef_search` best candidates seen so far. Larger ef_search explores more of the
graph: slower, but closer to exact. Phase 2's job is to measure that trade-off.

Documents longer than the model's limit are split into chunks (chunking.py). A
document's score is then the score of its best chunk, often called MaxP.
"""

from __future__ import annotations

import numpy as np


def top_k_rows(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row of scores, the k best column indices and their scores.

    Best first; equal scores go to the lower index, the same rule as BM25.search,
    so results never depend on how a partition happened to order ties.
    Rows with fewer than k columns are padded with index -1 and score -inf.
    """
    m, n = scores.shape
    ids = np.full((m, k), -1, dtype=np.int64)
    out = np.full((m, k), -np.inf, dtype=np.float32)
    if n == 0 or k == 0:
        return ids, out
    take = min(k, n)
    for r in range(m):
        row = scores[r]
        if n > take:
            cand = np.argpartition(-row, take - 1)[:take]
            kth = row[cand].min()
            cand = np.flatnonzero(row >= kth)  # keep everything tied with the k-th score
        else:
            cand = np.arange(n)
        order = np.lexsort((cand, -row[cand]))[:take]
        ids[r, :take] = cand[order]
        out[r, :take] = row[cand[order]]
    return ids, out


class ExactIndex:
    """Brute-force inner-product search, from scratch."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.vectors)

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """Similarity of every query to every stored vector: shape (queries, vectors)."""
        return np.asarray(queries, dtype=np.float32) @ self.vectors.T

    def search(self, queries: np.ndarray, k: int, *, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Top k stored vectors per query: (ids, scores), each of shape (queries, k).

        Queries are processed in blocks so the score matrix stays small:
        256 queries against 60,000 vectors is 61 MB of float32.
        """
        queries = np.atleast_2d(queries)
        ids = np.empty((len(queries), k), dtype=np.int64)
        sc = np.empty((len(queries), k), dtype=np.float32)
        for s in range(0, len(queries), block):
            ids[s:s + block], sc[s:s + block] = top_k_rows(self.scores(queries[s:s + block]), k)
        return ids, sc

    def search_documents(self, queries: np.ndarray, chunk_doc: np.ndarray, k: int, *,
                         block: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Top k documents per query, each scored by its best chunk.

        chunk_doc gives each stored vector's document and must be non-decreasing
        (chunks stored in document order, every document present), so the best
        chunk per document is one np.maximum.reduceat over contiguous runs.
        """
        chunk_doc = np.asarray(chunk_doc)
        if len(chunk_doc) != len(self) or np.any(np.diff(chunk_doc) < 0):
            raise ValueError("chunk_doc must give one document per vector, in non-decreasing order")
        starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
        if len(starts) and (chunk_doc[0] != 0 or chunk_doc[-1] != len(starts) - 1):
            raise ValueError("every document from 0 to the last must have at least one chunk")
        queries = np.atleast_2d(queries)
        ids = np.empty((len(queries), k), dtype=np.int64)
        sc = np.empty((len(queries), k), dtype=np.float32)
        for s in range(0, len(queries), block):
            doc_scores = np.maximum.reduceat(self.scores(queries[s:s + block]), starts, axis=1)
            ids[s:s + block], sc[s:s + block] = top_k_rows(doc_scores, k)
        return ids, sc


class HNSWIndex:
    """FAISS's HNSW graph over inner product. Built single-threaded so the graph is reproducible."""

    def __init__(self, vectors: np.ndarray, *, m: int = 32, ef_construction: int = 200, threads: int = 1):
        import faiss

        self._faiss = faiss
        faiss.omp_set_num_threads(threads)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(vectors)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int, ef_search: int) -> tuple[np.ndarray, np.ndarray]:
        """Top k per query. The candidate list holds max(ef_search, k): it can't return more than it explored."""
        self.index.hnsw.efSearch = max(ef_search, k)
        scores, ids = self.index.search(np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32), k)
        return ids.astype(np.int64), scores


def best_chunk_per_document(chunk_ids: np.ndarray, chunk_scores: np.ndarray, chunk_doc: np.ndarray,
                            k: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn ranked chunk hits into ranked documents: a document's first (best) chunk decides its place.

    Rows may come back with fewer than k documents if the chunk hits cover
    fewer; those slots hold -1. Ask the index for more chunks than documents.
    """
    m = len(chunk_ids)
    ids = np.full((m, k), -1, dtype=np.int64)
    sc = np.full((m, k), -np.inf, dtype=np.float32)
    for r in range(m):
        seen: set[int] = set()
        for c, s in zip(chunk_ids[r], chunk_scores[r]):
            if c < 0:
                break
            d = int(chunk_doc[c])
            if d not in seen:
                ids[r, len(seen)], sc[r, len(seen)] = d, s
                seen.add(d)
                if len(seen) == k:
                    break
    return ids, sc
