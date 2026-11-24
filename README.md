# memory-retrieval

A retrieval and long-term memory engine for AI agents, built in seven phases:
keyword search, vector search, rank fusion, reranking, a crash-safe store, and
memories that stay correct as facts change. Every result in this README comes
from a command in the repository.

**Status: Phase 1 of 7.** An evaluation harness and BM25 written from scratch,
checked against the published Lucene baseline before anything is built on top.

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
uv sync
uv run pytest -q                              # 69 tests, under a second, no network
uv run python -m memret.evaluate --dataset all   # downloads 26 MB once; about 15 s after that
uv run python -m memret.tune --dataset all       # k1/b grid and analyzer ablation; about 45 s
```

Data is fetched from the BeIR datasets on the Hugging Face Hub at pinned
commits, and each file's SHA-256 is checked before use.

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

## Design decisions

| Decision | Why | Alternative considered |
| --- | --- | --- |
| Match a published baseline before improving anything | A metric bug or a tokenizer difference would silently shift every later comparison | Trust our own numbers |
| Lucene's idf, ln(1 + (N − df + 0.5) / (df + 0.5)) | Never negative, so very common terms still count slightly | Textbook idf, negative for terms in over half the documents |
| Exact document lengths by default; Lucene's one-byte lengths as an option | Exact is correct; the option exists only to explain the gap | Copy Lucene's rounding everywhere |
| Drop a result whose id equals the query id | BEIR and Anserini do it; FiQA has 55 test queries whose ids collide with unrelated document ids | Keep them, and fail to match the reference |
| Tune on development queries only | Test queries are used once, to check the choice | Pick the best setting on test, which overstates the gain |
| Report per-query values and paired tests | Differences of 0.005 nDCG are common; a mean alone can't tell signal from noise | Compare means |

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
tests/             69 tests
reports/phase1/    results written by the two commands
docs/design/       one write-up per phase
docs/pr/           pull request descriptions
```

## Limitations

- The index lives in memory and is rebuilt on every run. Persistence comes in Phase 5.
- English only. The tokenizer approximates Lucene's Unicode word-break rules with a regular expression.
- Timings come from a 4-CPU Linux virtual machine on an Apple-silicon MacBook, single-threaded.
  They show the order of magnitude, not a tuned benchmark.
- Keyword search alone misses paraphrases. That is what Phase 2 addresses.

## Roadmap

| Phase | Adds | Status |
| --- | --- | --- |
| 1 | Evaluation harness, BM25 from scratch | Done |
| 2 | Embeddings, embedding cache, exact and HNSW vector search | Next |
| 3 | Rank fusion of keyword and vector results, failure analysis | |
| 4 | Cross-encoder reranking under a latency budget | |
| 5 | SQLite store, transactional outbox, deletes, crash tests | |
| 6 | Agent memory: fact extraction, deduplication, facts that change over time | |
| 7 | Service: API, MCP interface, degraded modes, load tests, provable deletion | |
