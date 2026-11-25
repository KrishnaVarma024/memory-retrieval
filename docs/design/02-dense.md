# Phase 2: dense retrieval

## Goal

Add retrieval by meaning. BM25 only scores documents that share words with the
query, so "how to lower my pulse" never finds "reducing heart rate". An
embedding model maps both to nearby vectors. The phase has three questions to
answer with measurements, not assumptions: does our embedding pipeline produce
the model's published quality; is dense retrieval actually better than a tuned
BM25 on our data; and when does exact vector search stop being fast enough.

## What was built

| Part | File | Notes |
| --- | --- | --- |
| Embedder | `embed.py` | `BAAI/bge-small-en-v1.5` pinned to commit `5c38ec7c`. Tokenise (WordPiece, 512-token limit), encode, take the [CLS] vector, normalise to unit length. Queries get the model's instruction prefix. Batches sorted by length. The model loads only when a vector is missing from the cache. |
| Cache | `cache.py` | Key = SHA-256 of (model, revision, pooling, maximum length, normalisation, exact text). In memory as a dictionary; on disk as `.npz` shards, each written to a temporary file, synced and renamed. |
| Chunker | `chunking.py` | Documents that fit 510 tokens stay whole. Longer ones split into windows that end at the last sentence boundary in their second half, never past the cap, overlapping by up to 64 tokens. Offsets come from the tokenizer, so a chunk is an exact slice of its document. |
| Search | `dense.py` | `ExactIndex`: dot products in NumPy and the same top-k rule as BM25 (ties to the earlier document). `HNSWIndex`: FAISS `IndexHNSWFlat`, built on one thread so the graph is reproducible. A document's score is its best chunk's. |
| Commands | `evaluate_dense.py`, `ann.py`, `bench_embed.py` | Quality against BM25 and the published scores; exact against HNSW; embedding cost and cache behaviour. |

## Matching the published numbers

The model card reports MTEB results for each dataset. MTEB embeds each document
whole, truncated at 512 tokens, with the instruction prefix on queries: our
"document" mode.

| Dataset | nDCG@10 ours | Published | R@100 ours | Published |
| --- | --- | --- | --- | --- |
| SciFact | 0.71271 | 0.71275 | 0.94167 | 0.94167 |
| NFCorpus | 0.34379 | 0.34305 | 0.31113 | 0.31135 |
| FiQA-2018 | 0.40347 | 0.40343 | 0.69629 | 0.69552 |

SciFact and FiQA agree to within 0.0001 nDCG@10. NFCorpus is 0.0007 above the
card. We did not chase that gap: it is a tenth of the difference any later
phase will care about, and plausible sources include floating-point differences
across hardware and library versions, and how the two evaluations break ties.
The point of the check is that a pooling, prefix or normalisation mistake would
cost several points, and there is none.

One convention matters here as it did for BM25: a retrieved document whose id
equals the query id is dropped. FiQA has 55 such collisions.

## Dense against BM25

Phase 1 promised that later phases would compare against each dataset's tuned
BM25, not the default. That changes the conclusion.

| Dataset | Dense minus tuned BM25, nDCG@10 | 95% interval | p | Queries dense wins | BM25 wins |
| --- | --- | --- | --- | --- | --- |
| SciFact | +0.0298 | −0.0012 to +0.0605 | 0.06 | 73 | 51 |
| NFCorpus | +0.0179 | −0.0025 to +0.0380 | 0.08 | 119 | 93 |
| FiQA-2018 | +0.1535 | +0.1302 to +0.1767 | < 0.001 | 312 | 87 |

Against default BM25 (k1 = 0.9, b = 0.4) the SciFact and NFCorpus gains were
significant. Against tuned BM25 they are not. Only on FiQA, where a forum
question and its answer rarely share wording, is dense clearly better.

Two more observations shape the next phases:

- **Recall@100 improves where nDCG@10 doesn't.** NFCorpus gains 0.065
  recall@100 (interval +0.046 to +0.086) and FiQA 0.147. In Phase 4 a reranker
  sees only the first stage's top candidates, so recall is the number that
  limits it.
- **Each retriever wins queries the other loses.** Even on FiQA, BM25 is better
  on 87 of 648 queries. That is the case for combining them, which is Phase 3.

## Chunking

The model reads 510 tokens of content. 9% of SciFact and NFCorpus documents and
4% of FiQA's are longer, so their ends are invisible in document mode.

Chunking was decided on development queries before the test numbers were read:

| Dataset | Split | Chunks minus documents, nDCG@10 | 95% interval |
| --- | --- | --- | --- |
| SciFact | train (809) | −0.0015 | −0.0048 to +0.0017 |
| NFCorpus | dev (324) | −0.0005 | −0.0021 to +0.0010 |
| FiQA-2018 | dev (500) | +0.0027 | −0.0025 to +0.0091 |

No measurable effect on any dataset, so whole documents stay the default. That
fits the data: few documents are long, and in abstracts and forum answers the
start usually says what the document is about. The chunker stays, because the
memory framework will store long files and chat transcripts where this won't hold.

## Exact against approximate search

On FiQA, 57,638 vectors, one thread, one query at a time:

| Search | Recall@10 of exact | nDCG@10 | p95 latency |
| --- | --- | --- | --- |
| Exact, ours | 1 | 0.4035 | 3.80 ms |
| Exact, FAISS IndexFlatIP | 1 | 0.4035 | 5.50 ms |
| HNSW, ef_search 10 | 0.821 | 0.3710 | 0.088 ms |
| HNSW, ef_search 40 | 0.956 | 0.3957 | 0.214 ms |
| HNSW, ef_search 160 | 0.992 | 0.4005 | 0.599 ms |
| HNSW, ef_search 640 | 0.999 | 0.4035 | 2.040 ms |

HNSW (M = 32, efConstruction = 200) took 27 s to build on one thread. Our
exact search is faster than FAISS's here, probably because ours finds the top k
with a partial sort while FAISS keeps a heap per query. Both are exact: their
top 101 is identical for 632 of 648 queries, and the rest differ only where two
scores are equal or within about a millionth, which each rounds and breaks
differently. nDCG@10 is the same to four decimals.

**The crossover.** Padding the collection with perturbed copies of real vectors,
exact search's p95 grows by about 5.1 ms per 100,000 vectors: 6.3 ms at 100,000,
12.4 ms at 200,000, 21.3 ms at 400,000. A straight line through those points
crosses the Phase 0 budget of 50 ms at about 950,000 vectors. HNSW at
ef_search 40 stays under 0.4 ms up to 200,000.

So the Phase 0 decision holds: exact search first, HNSW when a collection nears
a million vectors or when latency budgets tighten, for example once a reranker
spends most of the time.

## The cache

| Measurement | Value |
| --- | --- |
| Cold throughput, 4 CPU cores | 25.3 texts/s, 4,189 tokens/s |
| Warm throughput | about 656,000 texts/s, 26,000 times faster |
| Duplicate documents (a cold run's only hits) | SciFact 0, NFCorpus 40 (1.1%), FiQA 39 (0.07%) |
| Chunk run after a document run | 83.4%, 83.1% and 91.2% of chunks already cached |
| Full cold FiQA run, estimated | 38 minutes |

Sorting each batch by length before padding raised throughput from 7.3 to 23.2
documents per second in an early test, because padding tokens cost as much as
real ones.

The cache is keyed by content, not by document id. That is what made two things
work: switching to chunks reused every document that stayed whole, and
embeddings computed on two different machines merged into one cache with no
coordination.

## The four questions

**What breaks at 100 times the data?** Embedding first: at 4,000 tokens per
second, 5.7 million FiQA-sized documents take about 65 hours on one machine.
That needs a GPU or parallel workers, which the content-addressed cache already
supports. Then memory: 5.7 million 384-dimension float32 vectors are 8.8 GB,
more than this machine's RAM, so vectors would be memory-mapped or compressed
(float16 halves them; product quantisation cuts them 16 to 32 times at some
recall cost). Exact search would take around 300 ms, far past the budget, so
HNSW or an inverted-file index becomes necessary. The cache loads every shard
into a dictionary at start-up, which also stops scaling; Phase 5 moves it into SQLite.

**What happens on failure?** A killed embedding run loses at most the shard it
was writing; everything finished is on disk and the next run skips it. A
damaged shard fails its zip checksum when read. A model change can't serve old
vectors, because the fingerprint is part of every key and every folder name.

**How do we know it is correct?** The pipeline reproduces the published nDCG@10
on three datasets. Exact search is tested against sorting every score and
against FAISS's exact index. Best-chunk aggregation is tested against chunk
search followed by aggregation. The embedder's pooling, normalisation, ordering
and truncation are tested offline with a tiny random BERT, so the tests need no
download. 109 tests in all.

**What would we change?** Phase 5 should move the cache into SQLite with
reference counts, so vectors can be deleted when their chunks are. And if a
real corpus has long documents, test chunk sizes below 510 tokens on
development queries; we only tried one size.

## Limits

- Timings come from a 4-CPU Linux VM on an Apple-silicon MacBook, CPU only.
  Embedding on the Mac's own GPU would be much faster.
- Part of FiQA's embeddings were computed on a second (x86) machine and merged
  through the cache. The evaluation matches the model card, so any numerical
  difference between the machines is below what matters here.
- Scaling beyond 57,638 vectors used perturbed copies of real vectors. Exact
  search's cost doesn't depend on the values; HNSW's recall and speed on real
  data of that size may differ.
- One model. Larger embedding models score higher on these datasets and cost
  more to run; comparing them is not this phase's question.
