# Add dense retrieval with a content-addressed embedding cache, and measure exact against HNSW search

## Summary

Adds embedding-based retrieval next to BM25: a pinned bge-small-en-v1.5
embedder, an embedding cache keyed by content, a chunker, exact vector search
written in NumPy, and FAISS HNSW. The pipeline reproduces the model's published
nDCG@10 on three BEIR datasets to within 0.0007, then answers two questions:
whether dense beats a tuned BM25 here (clearly on FiQA, not significantly on
SciFact or NFCorpus), and when exact search stops meeting the latency budget
(at about 950,000 vectors on this machine).

## Why

Phase 3 will fuse keyword and vector results, and Phase 4 will rerank them. Both
need a dense retriever whose quality is known, an honest baseline to compare it
with, and an embedding step that doesn't cost an hour every time it runs.

## What changed

| File | What it does | Worth a close look |
| --- | --- | --- |
| `embed.py` | Tokenise, encode, [CLS] pooling, L2 normalisation; model pinned to a commit; length-sorted batches | That pooling and the query prefix match the model card |
| `cache.py` | Vectors keyed by SHA-256 of (model fingerprint, text); `.npz` shards written atomically | `get_many`: duplicates within one call are computed once |
| `chunking.py` | Whole documents if they fit 510 tokens, else sentence-aligned windows with overlap | The loop in `chunk_spans` always moves forward |
| `dense.py` | Exact search with BM25's tie rule; FAISS HNSW; best chunk per document | `search_documents` relies on chunks being in document order and checks it |
| `evaluate_dense.py` | Dense vs tuned BM25 with paired tests; vs published scores; chunking chosen on dev queries | BM25 settings read from Phase 1's tuning reports |
| `ann.py` | Recall-versus-latency sweep over ef_search; scaling to 400,000 vectors | Timing: one thread, warm-up pass, three repeats for scaling |
| `bench_embed.py` | Cold and warm throughput; hit rates counted from the texts | |
| `pyproject.toml` | A `dense` extra: torch (CPU build on Linux), transformers, faiss-cpu | Phase 1 still installs without them |

About 1,165 lines of code and 311 of tests. `reports/`, `docs/` and `uv.lock`
are generated or prose.

## Results

| Dataset | nDCG@10, tuned BM25 | nDCG@10, dense | Published | Dense minus BM25 (95% interval) |
| --- | --- | --- | --- | --- |
| SciFact | 0.6830 | 0.7127 | 0.7127 | +0.030 (−0.001 to +0.061) |
| NFCorpus | 0.3259 | 0.3438 | 0.3431 | +0.018 (−0.003 to +0.038) |
| FiQA-2018 | 0.2499 | 0.4035 | 0.4034 | +0.154 (+0.130 to +0.177) |

On FiQA, 57,638 vectors, one thread: exact search 3.8 ms p95; HNSW at
ef_search 40 returns 95.6% of the exact top 10 in 0.21 ms and costs 0.008
nDCG@10. Exact search crosses the 50 ms budget at about 950,000 vectors.

Embedding runs at about 4,200 tokens/s on four CPU cores; cached vectors come
back about 26,000 times faster, and a chunked run reuses 83–91% of a document
run's vectors.

Tables: `reports/phase2/dense.md`, `ann.md`, `cache.md`. Write-up:
`docs/design/02-dense.md`.

## Decisions a reviewer may question

- **Comparing with tuned BM25.** Against default BM25, dense looked
  significantly better on all three datasets; against the settings Phase 1
  tuned on development queries, only on FiQA. The tuned comparison is the honest one.
- **Exact search as the default.** HNSW is 18 times faster here, but exact
  search is already 13 times under budget at this size and never misses.
- **Whole documents, not chunks.** Chunking made no measurable difference on
  development queries of any dataset, so it isn't the default. The chunker is
  kept for the long texts later phases will store.
- **Part of FiQA embedded on another machine.** To save about half an hour, two
  machines embedded different halves and the content-keyed cache merged them.
  The results match the model card.
- **Scaling on padded vectors.** Real data beyond 57,638 vectors would need
  another corpus; exact search's cost doesn't depend on the values.

## Alternatives considered

- **sentence-transformers instead of calling the model directly.** Fewer
  lines, but pooling, truncation and normalisation would be hidden defaults, and
  those are exactly what the parity check tests.
- **hnswlib, or HNSW from scratch.** FAISS gives exact and approximate indexes
  behind one interface. HNSW from scratch remains a stretch goal.
- **Caching by document id.** Simpler keys, but a changed text or a new chunking
  setting would silently reuse stale vectors.

## Test plan

```
uv sync --extra dense
uv run pytest -q                                                     # 109 passed
uv run python -m memret.evaluate_dense --dataset all --mode both
uv run python -m memret.evaluate_dense --dataset all --choose-chunking
uv run python -m memret.ann --dataset fiqa
uv run python -m memret.bench_embed
```

- Exact search checked against sorting every score and against FAISS's exact index.
- The embedder checked offline with a tiny random BERT: unit length, order
  restored after length sorting, vector equals the normalised [CLS] output.
- The cache checked across restarts, with writes split over several shards, and against a
  model with a different fingerprint.
- Interrupted embedding runs resumed from their last shard (done for real while
  embedding the three corpora).

## Limitations

- CPU only; timings come from a 4-CPU Linux VM on the author's MacBook.
- The cache loads fully into memory at start-up and never deletes; Phase 5 moves it into SQLite.
- One embedding model and one chunk size.
