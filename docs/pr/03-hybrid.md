# Add hybrid retrieval: rank and score fusion of BM25 and dense, chosen on development queries, with a read failure analysis

## Summary

Fuses Phase 1's BM25 and Phase 2's dense retrieval into one ranking. 206 fusion
settings (reciprocal rank fusion and convex combination under three score
normalisations, at two candidate depths) are scored on development queries; one
global setting is chosen and tested once on three BEIR datasets. The hybrid
beats both single retrievers on all three, every comparison surviving Holm's
correction. The ten worst losses per dataset are written out and were read and
labelled; the queries the hybrid gets entirely wrong form a hard-query set for
Phase 4.

## Why

The two retrievers fail on different queries: picking the better one per query
after the fact would add 0.03 to 0.06 nDCG@10 over dense alone. Phase 4's
reranker needs the best first stage we can build, and the framework needs a
fusion setting it can ship without relevance judgements.

## What changed

| File | What it does | Worth a close look |
| --- | --- | --- |
| `fusion.py` | RRF, min-max / theoretical min-max / z-score normalisation, weighted combination, final ordering | Ties go to the lower document position, the same rule as both retrievers |
| `hybrid.py` | `CandidateSet` (both retrievers' scores and ranks for the union of their top lists), `FusionConfig`, `HybridSearcher` | `candidate_set` relies on both searches being exact, so every candidate has both scores |
| `evaluate_hybrid.py` | Dev grid, global and per-dataset choice, paired tests with Holm correction, losers, hard queries, latency, effective dense share | Test queries are only evaluated after both choices are fixed |
| `tests/` | 33 new tests: formulas by hand, single-retriever settings reproduce each search, truncation equals a shallower search, Holm by hand | |

About 780 lines of code and 295 of tests. `reports/` and `docs/` are generated or prose.

## Results

Hybrid: convex combination, theoretical min-max, α = 0.95, depth 100.

| Dataset | nDCG@10 BM25 | Dense | Hybrid | Hybrid minus dense (95% interval) | Holm p | Oracle |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | 0.6830 | 0.7127 | 0.7414 | +0.029 (+0.010 to +0.048) | 0.006 | 0.7751 |
| NFCorpus | 0.3259 | 0.3438 | 0.3716 | +0.028 (+0.016 to +0.041) | < 0.001 | 0.3909 |
| FiQA-2018 | 0.2499 | 0.4035 | 0.4184 | +0.015 (+0.004 to +0.026) | 0.009 | 0.4365 |

- Closes 45–59% of the gap between dense and the per-query oracle.
- Textbook RRF (k = 60, equal weights) is below dense alone on FiQA (0.3696).
  Summing raw scores reproduces BM25 (0.6844 against 0.6830 on SciFact).
- Tuning per dataset didn't help: identical on FiQA, +0.002 (n.s.) on NFCorpus,
  −0.008 (significant) on SciFact.
- Every winning setting gives dense about 0.73 of the ordering once each
  normalisation's spread is accounted for, whatever its nominal α.
- Losers, 30 read: 11 exact terms only BM25 sees, 7 vocabulary mismatch only
  dense bridges, 7 near-ties, 5 judgement or query artefacts.
- 180 hard queries are ranking failures (Phase 4's target), 143 recall failures.
- FiQA, one query at a time: 3.2 ms p95 for both retrievers plus fusion,
  excluding query embedding.
- Reports are byte-identical on x86 and ARM, apart from timings.

Tables: `reports/phase3/hybrid.md`; losers: `reports/phase3/losers_*.md`;
write-up: `docs/design/03-hybrid.md`.

## Decisions a reviewer may question

- **One global setting, not one per dataset.** A new user's data has no
  judgements to tune on, so the global setting is what ships, and per-dataset
  tuning didn't beat it on test.
- **tmm over min-max.** Tied on development queries (0.5159 against 0.5150);
  the rule took the argmax. α = 0.95 is specific to bge-small's cosine range
  and must be re-tuned with a new model. The write-up proposes a robustness
  tie-breaker for future selections.
- **Depth 100 for the hybrid.** Chosen on development queries with the rest of
  the setting. Its R@1000 is capped by the candidate count and isn't compared.
- **Missing BM25 tuning file is an error.** Comparing against default BM25
  would flatter the hybrid; Phase 2's helper fell back silently and is left
  unchanged.

## Alternatives considered

- **Fusion inside each retriever's top k only, with missing scores imputed.**
  Needed with approximate indexes; unnecessary here because both searches are
  exact, and imputation adds a setting with no right answer.
- **Learned fusion (a small model over both scores and query features).** Needs
  training queries per domain, which the framework won't have. Phase 4's
  reranker is the learned stage.
- **Wider RRF weight grid.** All datasets chose the top weight, 0.7; extending
  it to 0.95 on development queries left the best RRF average below convex
  combination's.

## Test plan

```
uv sync --extra dense
uv run pytest -q                                          # 143 passed
uv run python -m memret.evaluate_hybrid --dataset all     # under a minute with Phase 2's cache
```

- Single-retriever settings checked to reproduce `BM25.search` and dense
  search; the report's BM25 and dense rows reproduce Phases 1 and 2 to four decimals.
- RRF, each normalisation and Holm's correction checked against hand-computed values.
- A query with no BM25 match falls back to dense's ranking.
- Ran on the x86 container and the ARM VM; outputs compared byte for byte.

## Limitations

- The global setting was chosen on the same three datasets it is reported on
  (development queries to choose, test to report). Transfer to new data is assumed.
- One reader labelled the losers.
- Timings from a 4-CPU Linux VM on an Apple-silicon MacBook.
