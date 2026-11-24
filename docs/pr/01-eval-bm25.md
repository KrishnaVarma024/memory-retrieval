# Phase 1: evaluation harness and BM25 from scratch

## Problem

Every later phase of this project (dense retrieval, fusion, reranking, agent
memory) will be judged by how much it improves ranking quality. That judgement
is only as good as the harness making it, so the harness comes first, and it
has to be checked against something outside this repository before it is
trusted.

## What this PR adds

- `memret.datasets`: SciFact, NFCorpus and FiQA-2018 from BEIR, downloaded at
  pinned Hugging Face commits with SHA-256 verification and atomic writes.
- `memret.text`: an analyzer mirroring Anserini's (Lucene StandardTokenizer
  approximation, possessive removal, lower-casing, 33 stopwords, Porter stemming).
- `memret.bm25`: an inverted index in compressed-sparse-row arrays and Lucene's
  BM25 variant, with per-posting weights precomputed so a query is a handful of
  vectorised additions.
- `memret.metrics` and `memret.stats`: per-query nDCG@10, MRR@10, R@100 and
  R@1000 with trec_eval's definitions; paired bootstrap intervals and a paired
  randomization test.
- `memret.evaluate`: test-set evaluation against Anserini's published baseline.
- `memret.tune`: k1/b grid and analyzer ablation on development queries, then a
  single tuned-versus-default comparison on test.

## Results

| Dataset | nDCG@10 ours | Anserini | R@100 ours | Anserini |
| --- | --- | --- | --- | --- |
| SciFact | 0.6776 | 0.6789 | 0.9253 | 0.9253 |
| NFCorpus | 0.3205 | 0.3218 | 0.2456 | 0.2457 |
| FiQA-2018 | 0.2354 | 0.2361 | 0.5418 | 0.5395 |

The gap is explained, not just small. Lucene stores document lengths in one
byte; emulating that gives 0.6789 on SciFact, identical to Anserini, and
0.3214 and 0.2360 on the others. What remains matches a tokenizer that emits
0.01% to 0.07% fewer terms than Lucene's.

On development queries, removing stemming costs 0.020 to 0.030 nDCG@10 (all
intervals exclude zero); keeping stopwords changes nothing measurable. Tuned k1
and b improve test nDCG@10 by 0.0054 on NFCorpus and 0.0145 on FiQA (both
p = 0.003); SciFact's +0.0053 is within noise.

## Design decisions and trade-offs

- **Match a reference before improving.** Copying Anserini's analyzer and
  parameters costs some freedom now, but turns "our BM25 works" into a
  four-decimal check instead of a belief.
- **Precomputed per-posting weights.** Uses one float per posting of extra memory
  (30 MB for FiQA) to make queries cheap. Changing k1 or b recomputes weights,
  never the index.
- **Exact lengths by default, Lucene's as an option.** The option exists to
  explain the reference gap; it is not the system's behaviour.
- **Dropping results whose id equals the query id.** A BEIR convention that
  matters in FiQA, where 55 test query ids collide with document ids.
- **Ties broken by document position.** Deterministic, and the same as Lucene.
  trec_eval breaks ties differently (by document id, descending), which is one
  reason we evaluate our own ranking rather than re-sorting by score.

## Alternatives considered

- Using `bm25s` or Pyserini instead of writing BM25: faster to get numbers,
  but nothing learned about inverted indexes, and no way to explain a gap.
- Postings as Python dicts of lists: simpler to read, roughly an order of
  magnitude more memory and slower queries.
- Tuning on test queries: would have reported a larger gain that is not real.

## Testing

69 tests, under a second, no network:

- BM25 against a slow formula-level reference on a hand-built corpus, across
  five (k1, b) settings and five queries; idf positivity; length normalisation;
  term-frequency saturation; empty documents; tie-breaking; top-k cut-off;
  Lucene length encoding values.
- Metrics against hand-computed nDCG, recall and MRR, an oracle ranking, and
  unanswered queries counting as zero.
- Statistics on identical systems, a consistently better system and noise.
- Data loading on a fixture in BEIR's layout, including duplicate ids,
  malformed qrels, and a checksum mismatch leaving no partial file.

Reproduce with the commands in the README.

## Known limitations

- In-memory, single-threaded, rebuilt every run.
- Timings measured in a Linux VM on an Apple-silicon MacBook; indicative only.
- Multiple comparisons in the tuning report are not corrected for.

## Next

Phase 2: chunking, embeddings with a cache, and exact then approximate (HNSW)
vector search, evaluated with this harness against these BM25 numbers.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Jyjdgap7o32HDKVcXjMdP5
