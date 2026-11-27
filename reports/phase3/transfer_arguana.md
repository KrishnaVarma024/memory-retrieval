# Phase 3, held out: arguana

1,406 test queries over 8,674 documents. Nothing was tuned on this dataset: BM25 uses Phase 1's defaults (k1 0.9, b 0.4) and the hybrid is the shipped setting, **convex tmm, alpha 0.95, depth 100**, chosen on the three Phase 3 datasets.

| System | nDCG@10 | R@100 |
| --- | --- | --- |
| BM25, default k1 and b | 0.4046 | 0.9331 |
| Dense | 0.6061 | 0.9879 |
| RRF, k=60 | 0.5408 | 0.9872 |
| Hybrid | 0.6063 | 0.9900 |

| Comparison | nDCG@10 difference | 95% interval | Holm p | R@100 difference | 95% interval |
| --- | --- | --- | --- | --- | --- |
| Hybrid vs BM25, default k1 and b | +0.2017 | +0.1844 to +0.2190 | < 0.001 | +0.0569 | +0.0448 to +0.0697 |
| Hybrid vs Dense | +0.0002 | -0.0054 to +0.0059 | 0.934 | +0.0021 | +0.0000 to +0.0050 |

Dense's share of the shipped setting's ordering (see `hybrid.md` for both measures):

| Dataset | Pairwise, top 20 | Proxy |
| --- | --- | --- |
| arguana (held out) | 0.79 | 0.87 |
| scifact | 0.58 | 0.74 |
| nfcorpus | 0.54 | 0.67 |
| fiqa | 0.60 | 0.73 |

1,298 of the queries are themselves documents in the collection. Each system's top 10 contains the query's own document on this many of them before it is dropped: BM25, default k1 and b 1,298, Dense 1,298, RRF, k=60 1,298, Hybrid 1,298. The own document is unjudged, so it is dropped from every ranking, as in Phases 1 to 3 and BEIR's evaluation.
