# Add BM25 retrieval and an evaluation harness that reproduces Anserini on BEIR

## Summary

Adds a from-scratch BM25 retriever and the harness that will judge every later
retrieval change in this repo. On three BEIR datasets it lands within 0.0013
nDCG@10 of Anserini's published Lucene baseline, and the gap is accounted for.

## Why

Phases 2 to 4 will be accepted or rejected on differences of about 0.01
nDCG@10, so the harness has to be checked against an outside reference first.

## What changed

| File | What it does | Worth a close look |
| --- | --- | --- |
| `bm25.py` | Inverted index in CSR arrays; Lucene's BM25 with per-posting weights precomputed | The top-k tie handling in `search` |
| `metrics.py` | Per-query nDCG@10, MRR@10, R@100, R@1000 (trec_eval definitions) | Unanswered queries count as 0, not dropped |
| `stats.py` | Paired bootstrap interval and paired randomization test over queries | Seeded; chunked to bound memory |
| `text.py` | Analyzer mirroring Anserini's: tokenize, possessives, lowercase, 33 stopwords, Porter | Regex approximation of Lucene's tokenizer |
| `datasets.py` | BEIR downloads pinned to Hub commits, SHA-256 checked, written atomically | |
| `evaluate.py`, `tune.py` | Test-set evaluation vs Anserini; k1/b grid and analyzer ablation on dev queries | Test queries used once per dataset |

889 lines of code and 397 of tests. The commits are split by module, so it
reviews most easily commit by commit in the order above. `reports/`, `docs/`
and `uv.lock` are generated or prose.

## Results

| Dataset | nDCG@10 | with Lucene's length encoding | Anserini |
| --- | --- | --- | --- |
| SciFact | 0.6776 | 0.6789 | 0.6789 |
| NFCorpus | 0.3205 | 0.3214 | 0.3218 |
| FiQA-2018 | 0.2354 | 0.2360 | 0.2361 |

Lucene stores document length in one byte (exact below 24, four significant
bits above), so long documents are scored with a rounded length. Emulating that
closes the gap on SciFact; the rest matches our tokenizer emitting 0.01–0.07%
fewer terms. Exact lengths stay the default.

Tuning and analyzer ablations are in `reports/phase1/tuning.md`; the
write-up is `docs/design/01-eval-bm25.md`.

## Decisions a reviewer may question

- **Dropping results whose id equals the query id.** BEIR and Anserini both do
  it. In FiQA, 55 test query ids collide with unrelated document ids, so
  without it we would not be comparing like with like.
- **Precomputed weights.** About 30 MB extra on FiQA for queries that are a
  few vectorised additions.
- **SciFact's training queries as its dev set.** It ships no dev split. They
  are disjoint from test.
- **Ties go to the earlier document**, as in Lucene. trec_eval would break
  them by document id, so we score our own ranking rather than re-sorting.

## Alternatives considered

- **Using `bm25s` or Pyserini instead of writing BM25.** Faster to get numbers,
  but there would be no way to explain a gap against the reference, and nothing
  to build the later phases' indexes on.
- **Postings as Python dicts of lists.** Easier to read; roughly an order of
  magnitude more memory and slower queries than flat arrays.
- **Choosing k1 and b on test queries.** Would have reported a gain that is
  partly selection noise. Test queries are used once.

## Test plan

```
uv run pytest -q                                # 69 passed
uv run python -m memret.evaluate --dataset all  # prints the table above
uv run python -m memret.tune --dataset all      # writes reports/phase1/tuning.md
```

- BM25 checked against a formula-level reference implementation on a hand-built
  corpus, over five (k1, b) settings and five queries.
- Metrics checked against hand-computed values and an oracle ranking.
- Ran from a fresh clone with an empty data folder: files downloaded, checksums
  passed, and every metric matched; only timing fields differed.

## Limitations

- In memory and single-threaded; the index is rebuilt each run (persistence is Phase 5).
- Timings come from a Linux VM on the author's MacBook and are indicative only.
- No correction for multiple comparisons across the nine dev/test tests.

