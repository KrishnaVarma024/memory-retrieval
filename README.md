# memory-retrieval

A retrieval and long-term memory engine for AI agents, built in seven phases:
keyword search, vector search, rank fusion, reranking, a crash-safe store, and
memories that stay correct as facts change. Every result in this README comes
from a command in the repository.

**Status: Phase 3 of 7.** An evaluation harness and BM25 written from scratch
(Phase 1); dense retrieval with an embedding model, a content-addressed
embedding cache, a chunker, exact vector search and HNSW (Phase 2); and hybrid
retrieval that fuses the two, with its worst losses read by hand (Phase 3).
Each is checked against a published reference or the previous phase's numbers
before anything is built on top.

## Phase 3 results

BM25 (tuned per dataset in Phase 1) and dense retrieval (Phase 2) fused into one
ranking. 206 fusion settings were scored on development queries; one setting
for all datasets was chosen there and tested once: **convex combination of
theoretical min-max-normalised scores, dense weight 0.95, top 100 from each
retriever**.

| Dataset | nDCG@10, BM25 | Dense | RRF, k=60 | Hybrid | Hybrid minus dense (95% interval) | Oracle |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | 0.6830 | 0.7127 | 0.7319 | 0.7414 | +0.029 (+0.010 to +0.048), Holm p = 0.006 | 0.7751 |
| NFCorpus | 0.3259 | 0.3438 | 0.3643 | 0.3716 | +0.028 (+0.016 to +0.041), Holm p < 0.001 | 0.3909 |
| FiQA-2018 | 0.2499 | 0.4035 | 0.3696 | 0.4184 | +0.015 (+0.004 to +0.026), Holm p = 0.009 | 0.4365 |

The oracle picks, per query, whichever single retriever did better.

1. **The hybrid beats both retrievers on all three datasets.** All six
   comparisons against a single retriever hold after Holm's correction for
   multiple tests. It closes 45–59% of the gap between dense and the oracle.
2. **On a fourth dataset nobody tuned on, it matches the better retriever.**
   ArguAna was held out from every choice. There the shipped setting scored
   0.6063 against dense's 0.6061 (+0.0002, interval −0.005 to +0.006) and beat
   untuned BM25 by 0.20. Across four datasets the hybrid was never worse than
   the better single retriever, and better on three.
3. **How you fuse matters more than whether you fuse.** Adding raw scores
   reproduces BM25 (0.6844 against 0.6830 on SciFact), because BM25 scores run
   into the tens while cosines sit near 0.5 to 0.8. Textbook reciprocal rank
   fusion falls below dense alone on FiQA and on ArguAna, by giving equal votes
   to a much weaker retriever.
4. **The settings agree on how much say dense gets.** Each dataset's best
   setting used a different normalisation and weight. Measured where it
   matters, among pairs in the fused top 20 that BM25 and dense order
   differently, every winner sides with dense in 54–60% of the disputes.
5. **One setting for every dataset is enough.** Tuning per dataset was identical
   on FiQA, +0.002 (within noise) on NFCorpus and −0.008 on SciFact, so a new
   user's data, which has no judgements, loses nothing measurable here.
6. **The hybrid's worst losses come from both directions.** Of 30 read by hand:
   11 are exact terms only BM25 sees (gene names, study names, verbatim claims),
   7 are paraphrases only dense bridges, 7 are near-ties, and 5 are unjudged or
   doubtfully judged documents. No single weight fixes both of the first two;
   a reranker that reads query and document together might (Phase 4).
7. **It costs about a millisecond.** On FiQA's 57,638 documents, fusion adds
   about 1 ms; both retrievers plus fusion take 3 to 5 ms at the 95th
   percentile across three runs, excluding query embedding.

Full tables: [reports/phase3/hybrid.md](reports/phase3/hybrid.md); the held-out
dataset: [reports/phase3/transfer_arguana.md](reports/phase3/transfer_arguana.md); the worst
losses with their documents: [SciFact](reports/phase3/losers_scifact.md),
[NFCorpus](reports/phase3/losers_nfcorpus.md), [FiQA](reports/phase3/losers_fiqa.md).
Write-up: [docs/design/03-hybrid.md](docs/design/03-hybrid.md).

## Phase 2 results

[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (33M parameters,
384 dimensions), pinned to one commit of its weights, on the same test queries.
BM25 here uses the k1 and b each dataset got from Phase 1's tuning.

| Dataset | nDCG@10, BM25 | nDCG@10, dense | nDCG@10, published | Dense minus BM25 (95% interval) | R@100, BM25 | R@100, dense |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | 0.6830 | 0.7127 | 0.7127 | +0.030 (−0.001 to +0.061), p = 0.06 | 0.9282 | 0.9417 |
| NFCorpus | 0.3259 | 0.3438 | 0.3431 | +0.018 (−0.003 to +0.038), p = 0.08 | 0.2461 | 0.3111 |
| FiQA-2018 | 0.2499 | 0.4035 | 0.4034 | +0.154 (+0.130 to +0.177), p < 0.001 | 0.5489 | 0.6963 |

1. **Our embedding pipeline reproduces the model's published numbers.** nDCG@10
   is within 0.0001 of the model card on SciFact and FiQA and 0.0007 on NFCorpus,
   so tokenisation, pooling and normalisation are right before anything is compared.
2. **Dense beats tuned BM25 clearly on one dataset of three.** On FiQA's forum
   questions, where wording rarely matches the answer's, it gains 0.154 nDCG@10.
   On SciFact and NFCorpus the gain is 0.018 to 0.030 and the intervals include
   zero. Recall@100 improves significantly on NFCorpus (+0.065) and FiQA (+0.147),
   which matters for Phase 4, where a reranker only sees the first stage's top 100.
3. **Exact search is fast enough far beyond our data.** Searching 57,638 vectors
   exactly takes 3.8 ms at the 95th percentile on one thread. Measured up to
   400,000 vectors (21 ms), it would cross the 50 ms budget at about 950,000.
4. **HNSW is about 18 times faster, for a small, measurable loss.** At
   ef_search = 40 it returns 95.6% of the exact top 10 in 0.21 ms (p95), costing
   0.008 nDCG@10; at ef_search = 640 it matches exact search's nDCG@10.
5. **Chunking long documents didn't help here**, on development queries of all
   three datasets (differences of −0.0015 to +0.0027, every interval spanning
   zero), because 91–96% of these documents fit the model's 512 tokens whole.
   Whole documents stay the default; the chunker is kept for longer text.
6. **The cache makes embedding a one-time cost.** Cold, the model embeds about
   4,200 tokens per second on four CPU cores, so FiQA takes 38 minutes. A cached
   vector comes back about 26,000 times faster, and switching to chunks reuses
   83–91% of the vectors a document run already computed.

Full tables: [reports/phase2/dense.md](reports/phase2/dense.md),
[reports/phase2/ann.md](reports/phase2/ann.md) and
[reports/phase2/cache.md](reports/phase2/cache.md). Write-up:
[docs/design/02-dense.md](docs/design/02-dense.md).

## Phase 1 results

BM25 (k1 = 0.9, b = 0.4) on the test queries of three [BEIR](https://arxiv.org/abs/2104.08663)
datasets, against [Anserini](https://github.com/castorini/anserini)'s published
BM25 "flat" baseline, which runs on Lucene.

| Dataset | Documents | Test queries | nDCG@10, ours | nDCG@10, Anserini | R@100, ours | R@100, Anserini |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | 5,183 | 300 | 0.6776 | 0.6789 | 0.9253 | 0.9253 |
| NFCorpus | 3,633 | 323 | 0.3205 | 0.3218 | 0.2456 | 0.2457 |
| FiQA-2018 | 57,600 | 648 | 0.2354 | 0.2361 | 0.5418 | 0.5395 |

1. **The from-scratch BM25 matches Lucene to within 0.0013 nDCG@10.** Most of the
   gap is one storage detail: Lucene keeps each document's length in a single
   byte, so long documents are scored with a rounded length. Emulating that
   reproduces SciFact exactly (0.6789) and leaves gaps of 0.0004 and 0.0001 on
   the others. The rest is tokenisation: our regular-expression tokenizer
   produces 0.01% to 0.07% fewer terms than Lucene's.
2. **Stemming matters and stopword removal doesn't.** On development queries,
   turning off Porter stemming costs 0.020 to 0.030 nDCG@10 on every dataset,
   with 95% intervals excluding zero. Keeping stopwords changes nothing
   measurable, because BM25's idf already gives common words almost no weight.
3. **Tuning k1 and b helps on two of three datasets.** Chosen on development
   queries and checked once on test: +0.0054 on NFCorpus (p = 0.003), +0.0145 on
   FiQA (p = 0.003), and +0.0053 on SciFact, within noise (interval −0.0034 to +0.0140).
4. **It is fast enough to stay out of the way.** Median query latency is about
   1 ms on FiQA's 57,600 documents, single-threaded, including tokenisation.

Full tables: [reports/phase1/bm25.md](reports/phase1/bm25.md) and
[reports/phase1/tuning.md](reports/phase1/tuning.md). Write-up:
[docs/design/01-eval-bm25.md](docs/design/01-eval-bm25.md).

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dense                         # Phase 1 alone needs only `uv sync`
uv run pytest -q                              # 147 tests, a few seconds, no network
uv run python -m memret.evaluate --dataset all   # Phase 1: downloads 26 MB once; about 15 s after that
uv run python -m memret.tune --dataset all       # Phase 1: k1/b grid and analyzer ablation; about 45 s

uv run python -m memret.evaluate_dense --dataset all --mode both   # Phase 2: embeds once (see below), then a few minutes
uv run python -m memret.evaluate_dense --dataset all --choose-chunking
uv run python -m memret.ann --dataset fiqa       # recall-versus-latency curve and scaling; about 4 min
uv run python -m memret.bench_embed              # embedding throughput and cache hit rates; about 2 min

uv run python -m memret.evaluate_hybrid --dataset all   # Phase 3: needs Phase 1's tuning reports and Phase 2's vectors; under a minute
uv run python -m memret.evaluate_transfer --dataset arguana   # Phase 3, held out: embeds ArguAna on the first run, about 10 minutes
```

Data is fetched from the BeIR datasets on the Hugging Face Hub at pinned
commits, and each file's SHA-256 is checked before use. The embedding model is
fetched the same way, pinned to one commit of its weights.

On macOS, torch and faiss-cpu each bundle their own OpenMP runtime, and a
process that loads both crashes as soon as FAISS starts worker threads. No
command loads both. The test suite does, so on macOS it allows the duplicate
runtime and runs single-threaded (see `tests/conftest.py`).

The first Phase 2 run embeds about 76,000 texts: documents, chunks and queries. On four CPU
cores that takes about an hour; on an Apple-silicon Mac it is faster. Vectors
go to a cache under `data/cache/`, written in shards, so an interrupted run
resumes where it stopped and every later run skips the work.

## How Phase 1 works

```
BEIR files ──► Analyzer ──► Inverted index ──► BM25 scorer ──► top k ──► metrics per query ──► paired tests
pinned,        tokenise,     term → (doc, tf)    weights         ties go    nDCG@10, MRR@10,      bootstrap interval,
SHA-256        lowercase,    in flat arrays      precomputed     to the     R@100, R@1000         randomization p
checked        33 stopwords, (CSR layout)        per posting     earlier
               Porter stem                                       document
```

- **Analyzer** (`text.py`): copies the pipeline Anserini uses, so the comparison
  above tests BM25 rather than two different tokenizers.
- **Inverted index** (`bm25.py`): every term's postings sit end to end in two
  arrays with an offsets array, built by sorting (term, document, frequency)
  triples. A query touches only its own terms' postings.
- **BM25** (`bm25.py`): everything except the query term count depends only on
  the document and the term, so it is computed once per posting at build time.
  A query is then a few vectorised additions. Changing k1 or b recomputes
  weights without re-indexing.
- **Metrics** (`metrics.py`): trec_eval's definitions, one value per query.
  Queries a system returns nothing for count as zero instead of disappearing.
- **Statistics** (`stats.py`): a paired bootstrap interval and a paired
  randomization test, both resampling queries.

## How Phase 2 works

```
document ──► Chunker ──────────► Embedder ─────────────► Cache ──────────────► Exact or HNSW search ──► best chunk per document
             whole if it fits    tokenise, BERT encoder,  key = SHA-256 of      dot products of unit     a document scores as
             in 510 tokens,      [CLS] vector, unit       (model, revision,     vectors; HNSW walks a    its best chunk
             else windows at     length; queries get      pooling, text);       neighbour graph
             sentence breaks     an instruction prefix    shards on disk
```

- **Embedder** (`embed.py`): tokenisation, the encoder, [CLS] pooling and
  normalisation are each a visible line, not a library call. Batches are sorted
  by length, which made embedding three times faster by cutting padding.
- **Cache** (`cache.py`): content-addressed, so the same text under the same
  model is never embedded twice, whichever command asks. Shards are written to
  a temporary file and renamed, so a killed run leaves complete shards only.
- **Chunker** (`chunking.py`): documents that fit stay whole; longer ones split
  at sentence boundaries with a hard cap of 510 tokens and a 64-token overlap.
- **Search** (`dense.py`): exact search written in NumPy and checked against
  FAISS's exact index; HNSW from FAISS. Ties go to the earlier document, as in BM25.

## How Phase 3 works

```
query ─┬─ BM25 scores, every document ─── top 100 ─┐
       │                                          ├─ union ─► candidates with both scores ─► normalise ─► 0.05·BM25 + 0.95·dense ─► top k
       └─ dense scores, every document ── top 100 ─┘
```

- **Candidates** (`hybrid.py`): both searches are exact, so every candidate's
  BM25 and dense scores are both known, even for a document only one retriever
  returned. No missing score has to be guessed.
- **Fusion** (`fusion.py`): reciprocal rank fusion, Σ w / (k + rank), and convex
  combination of normalised scores. Theoretical min-max divides by the distance
  from each score's floor (0 for BM25, −1 for cosine).
- **Choosing the setting** (`evaluate_hybrid.py`): candidates are collected once
  per query, then all 206 settings are scored on development queries as array
  operations. Test queries are evaluated once, after the choice.
- **Checking it travels** (`evaluate_transfer.py`): the chosen setting is named
  once, as `hybrid.SHIPPED`, and a test fails if it disagrees with the choice in
  the committed report. It runs unchanged on a dataset held out from every choice.

## Design decisions

| Decision | Why | Alternative considered |
| --- | --- | --- |
| Match a published baseline before improving anything | A metric bug or a tokenizer difference would silently shift every later comparison | Trust our own numbers |
| Lucene's idf, ln(1 + (N − df + 0.5) / (df + 0.5)) | Never negative, so very common terms still count slightly | Textbook idf, negative for terms in over half the documents |
| Exact document lengths by default; Lucene's one-byte lengths as an option | Exact is correct; the option exists only to explain the gap | Copy Lucene's rounding everywhere |
| Drop a result whose id equals the query id | BEIR and Anserini do it; FiQA has 55 test queries whose ids collide with unrelated document ids | Keep them, and fail to match the reference |
| Tune on development queries only | Test queries are used once, to check the choice | Pick the best setting on test, which overstates the gain |
| Report per-query values and paired tests | Differences of 0.005 nDCG are common; a mean alone can't tell signal from noise | Compare means |
| Reproduce the embedding model's published scores first | The same reason as for BM25: an error in pooling or normalisation would shift every later comparison | Trust the library defaults |
| Pin the model to a commit of its weights | A model updated upstream would silently change every vector | Load "latest" |
| Cache keyed by (model, revision, pooling, text) | A vector can never be served for a different model, and identical text is never embedded twice | Cache by document id, which breaks when text changes |
| Exact search by default | 3.8 ms p95 at 57,638 vectors, and it never misses a neighbour; HNSW only pays off near a million vectors | HNSW from the start |
| Whole documents, not chunks, by default | Chunking made no measurable difference on development queries | Always chunk |
| Compare against tuned BM25, not default BM25 | A weak baseline overstates the new method's gain | Default k1 and b |
| Fuse the union of each retriever's top list, rescoring every candidate with both | Both searches are exact, so no score is missing | Fuse each top list's own scores and impute the gaps |
| One fusion setting for every dataset | A user's data has no judgements to tune on; per-dataset tuning didn't beat it on test | Tune per collection |
| Read the ten worst losses per dataset, not only the averages | The averages say the hybrid wins; the losses say where and why it doesn't | Report means only |

## Repository layout

```
src/memret/
    datasets.py    pinned BEIR downloads, checksum checks, loading
    text.py        analyzer: tokenizer, stopwords, Porter stemming
    bm25.py        inverted index and BM25 scoring
    metrics.py     nDCG, MRR, recall, per query
    stats.py       paired bootstrap interval, paired randomization test
    experiment.py  indexing, querying and timing shared by the commands
    evaluate.py    command: test-set evaluation against Anserini
    tune.py        command: development-set tuning and analyzer ablation
    embed.py       embedding model: tokenise, encode, [CLS] pooling, normalise
    cache.py       content-addressed embedding cache, sharded on disk
    chunking.py    sentence-aware chunking with a hard token cap
    dense.py       exact search, FAISS HNSW, best chunk per document
    evaluate_dense.py   command: dense retrieval against BM25 and the published scores
    ann.py         command: exact against HNSW, recall-versus-latency, scaling
    bench_embed.py command: embedding throughput and cache hit rates
    fusion.py      rank fusion and score normalisation
    hybrid.py      candidates from both retrievers, fused per query
    evaluate_hybrid.py  command: fusion settings on dev, hybrid against both retrievers, losers, hard queries
    evaluate_transfer.py  command: the shipped setting on a held-out dataset
tests/             147 tests
reports/phase1/    results written by the Phase 1 commands
reports/phase2/    results written by the Phase 2 commands
reports/phase3/    results written by the Phase 3 commands
docs/design/       one write-up per phase
docs/pr/           pull request descriptions
```

## Limitations

- Indexes live in memory and are rebuilt on every run; only embeddings are cached. Persistence comes in Phase 5.
- English only. The tokenizer approximates Lucene's Unicode word-break rules with a regular expression.
- Timings come from a 4-CPU Linux virtual machine on an Apple-silicon MacBook.
  They show the order of magnitude, not a tuned benchmark. Part of FiQA's
  embeddings were computed on a second, x86 machine to save time; the cache
  merges them because it is keyed by text, and the results match the model card.
- Scaling beyond 57,638 vectors was measured on padded copies of real vectors, not on real documents.
- The fusion setting was chosen on development queries of the same three datasets it is reported on.
  One held-out dataset (ArguAna) is a first check that it travels, not proof; there it matched dense
  rather than beating it.
- Latency varies between runs on the same VM: the Phase 3 total's p95 came out between 3.2 and 4.9 ms
  over three runs. Single numbers in the reports are one run.

## Roadmap

| Phase | Adds | Status |
| --- | --- | --- |
| 1 | Evaluation harness, BM25 from scratch | Done |
| 2 | Embeddings, embedding cache, exact and HNSW vector search | Done |
| 3 | Rank fusion of keyword and vector results, failure analysis | Done |
| 4 | Cross-encoder reranking under a latency budget | Next |
| 5 | SQLite store, transactional outbox, deletes, crash tests | |
| 6 | Agent memory: fact extraction, deduplication, facts that change over time | |
| 7 | Service: API, MCP interface, degraded modes, load tests, provable deletion | |
