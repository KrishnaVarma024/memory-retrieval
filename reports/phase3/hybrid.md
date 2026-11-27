# Phase 3: hybrid retrieval

Test queries. **BM25** uses each dataset's tuned k1 and b from Phase 1; **Dense** is bge-small-en-v1.5 on whole documents from Phase 2. Fusion reorders the union of each retriever's top 100 or top 1,000 (the depth is chosen with the rest of the setting). **Hybrid** is one setting for every dataset, chosen on development queries by average nDCG@10: **convex tmm, alpha 0.95, depth 100**. **Hybrid, tuned per dataset** uses each dataset's own best development setting. **RRF, k=60** is the untuned textbook setting. **Raw score sum** adds BM25 and cosine without normalising. **Oracle** picks, per query, whichever single retriever did better: a ceiling for any method that chooses between them.

## nDCG@10

| Dataset | BM25 | Dense | RRF, k=60 | Hybrid | Hybrid, tuned per dataset | Raw score sum | Oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 0.6830 | 0.7127 | 0.7319 | 0.7414 | 0.7331 | 0.6844 | 0.7751 |
| nfcorpus | 0.3259 | 0.3438 | 0.3643 | 0.3716 | 0.3735 | 0.3370 | 0.3909 |
| fiqa | 0.2499 | 0.4035 | 0.3696 | 0.4184 | 0.4184 | 0.2518 | 0.4365 |

## Recall@100

| Dataset | BM25 | Dense | RRF, k=60 | Hybrid | Hybrid, tuned per dataset | Raw score sum |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | 0.9282 | 0.9417 | 0.9683 | 0.9717 | 0.9683 | 0.9282 |
| nfcorpus | 0.2461 | 0.3110 | 0.3148 | 0.3190 | 0.3175 | 0.2875 |
| fiqa | 0.5489 | 0.6963 | 0.6984 | 0.7070 | 0.7070 | 0.5533 |

## Paired comparisons

Differences in mean over the same test queries, with 95% bootstrap intervals and two-sided randomization p-values. The six hybrid-against-single-retriever nDCG@10 tests are one family, so their p-values are also given after Holm's correction.

| Dataset | Comparison | nDCG@10 difference | 95% interval | p | Holm p | R@100 difference | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | Hybrid vs BM25 | +0.0584 | +0.0350 to +0.0818 | < 0.001 | < 0.001 | +0.0434 | +0.0220 to +0.0677 |
| scifact | Hybrid vs Dense | +0.0286 | +0.0096 to +0.0482 | 0.003 | 0.006 | +0.0300 | +0.0100 to +0.0533 |
| scifact | Hybrid vs RRF, k=60 | +0.0095 | -0.0051 to +0.0240 | 0.200 |  | +0.0033 | +0.0000 to +0.0100 |
| scifact | Hybrid, tuned per dataset vs Hybrid | -0.0082 | -0.0137 to -0.0037 | < 0.001 |  | -0.0033 | -0.0100 to +0.0000 |
| nfcorpus | Hybrid vs BM25 | +0.0457 | +0.0315 to +0.0605 | < 0.001 | < 0.001 | +0.0728 | +0.0555 to +0.0911 |
| nfcorpus | Hybrid vs Dense | +0.0278 | +0.0158 to +0.0409 | < 0.001 | < 0.001 | +0.0080 | -0.0002 to +0.0162 |
| nfcorpus | Hybrid vs RRF, k=60 | +0.0072 | +0.0005 to +0.0140 | 0.035 |  | +0.0041 | -0.0032 to +0.0131 |
| nfcorpus | Hybrid, tuned per dataset vs Hybrid | +0.0020 | -0.0010 to +0.0051 | 0.201 |  | -0.0015 | -0.0068 to +0.0034 |
| fiqa | Hybrid vs BM25 | +0.1684 | +0.1485 to +0.1883 | < 0.001 | < 0.001 | +0.1581 | +0.1350 to +0.1817 |
| fiqa | Hybrid vs Dense | +0.0149 | +0.0038 to +0.0264 | 0.009 | 0.009 | +0.0107 | -0.0017 to +0.0236 |
| fiqa | Hybrid vs RRF, k=60 | +0.0488 | +0.0371 to +0.0604 | < 0.001 |  | +0.0086 | -0.0023 to +0.0193 |
| fiqa | Hybrid, tuned per dataset vs Hybrid | +0.0000 | +0.0000 to +0.0000 | 1.000 |  | +0.0000 | +0.0000 to +0.0000 |

## Settings chosen on development queries

| Dataset | Split | Best RRF | Dev nDCG@10 | Best convex | Dev nDCG@10 | Global setting here |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | train (809) | RRF k=5, dense weight 0.7, depth 100 | 0.7658 | convex minmax, alpha 0.75, depth 100 | 0.7752 | 0.7735 |
| nfcorpus | dev (324) | RRF k=20, dense weight 0.7, depth 100 | 0.3417 | convex zscore, alpha 0.75, depth 1000 | 0.3481 | 0.3475 |
| fiqa | dev (500) | RRF k=5, dense weight 0.7, depth 100 | 0.4199 | convex tmm, alpha 0.95, depth 100 | 0.4268 | 0.4268 |

Alpha is dense's weight on normalised scores, not its share of the ranking: the normalisations leave the two retrievers with different spreads. Two measures put the settings on one scale, on test queries. **Pairwise**: among pairs in the fused top 20 that BM25 and dense order differently, the share the fusion orders as dense does; this is who decides the ranking users see. **Proxy**: alpha times dense's spread over all candidates, over the total (median over queries); cheaper, and it overstates dense's say, because the spread over all candidates is dominated by weak ones.

| Dataset | Hybrid: pairwise | Hybrid: proxy | Per-dataset setting: pairwise | Per-dataset setting: proxy |
| --- | --- | --- | --- | --- |
| scifact | 0.58 | 0.74 | 0.59 | 0.75 |
| nfcorpus | 0.54 | 0.67 | 0.54 | 0.75 |
| fiqa | 0.60 | 0.73 | 0.60 | 0.73 |

## Sensitivity on development queries

Dev nDCG@10 at depth 100, the best value in each row in bold. For convex combination, against alpha (0 is BM25 alone, 1 dense alone):

Convex minmax

| Dataset | 0 | 0.2 | 0.4 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 0.6990 | 0.7244 | 0.7498 | 0.7668 | 0.7700 | **0.7752** | 0.7727 | 0.7692 | 0.7605 | 0.7551 | 0.7469 |
| nfcorpus | 0.2926 | 0.3204 | 0.3339 | 0.3420 | 0.3425 | **0.3439** | 0.3428 | 0.3393 | 0.3363 | 0.3318 | 0.3278 |
| fiqa | 0.2708 | 0.3190 | 0.3638 | 0.4046 | 0.4185 | **0.4259** | 0.4259 | 0.4244 | 0.4221 | 0.4186 | 0.4094 |

Convex tmm

| Dataset | 0 | 0.2 | 0.4 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 0.6990 | 0.7032 | 0.7112 | 0.7230 | 0.7327 | 0.7395 | 0.7460 | 0.7554 | 0.7656 | **0.7735** | 0.7469 |
| nfcorpus | 0.2926 | 0.3079 | 0.3128 | 0.3185 | 0.3226 | 0.3271 | 0.3309 | 0.3361 | 0.3412 | **0.3475** | 0.3278 |
| fiqa | 0.2708 | 0.2804 | 0.2962 | 0.3193 | 0.3368 | 0.3447 | 0.3634 | 0.3827 | 0.4001 | **0.4268** | 0.4094 |

Convex zscore

| Dataset | 0 | 0.2 | 0.4 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 0.6990 | 0.7273 | 0.7543 | 0.7682 | 0.7741 | **0.7742** | 0.7710 | 0.7674 | 0.7613 | 0.7559 | 0.7469 |
| nfcorpus | 0.2926 | 0.3231 | 0.3360 | 0.3424 | 0.3430 | **0.3436** | 0.3412 | 0.3383 | 0.3347 | 0.3315 | 0.3278 |
| fiqa | 0.2708 | 0.3250 | 0.3724 | 0.4072 | 0.4216 | **0.4262** | 0.4237 | 0.4231 | 0.4214 | 0.4174 | 0.4094 |

RRF with equal weights, against k:

| Dataset | k=1 | k=5 | k=10 | k=20 | k=40 | k=60 | k=100 | k=200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | **0.7625** | 0.7611 | 0.7576 | 0.7542 | 0.7516 | 0.7511 | 0.7500 | 0.7493 |
| nfcorpus | 0.3319 | 0.3367 | 0.3382 | **0.3400** | 0.3383 | 0.3378 | 0.3370 | 0.3370 |
| fiqa | 0.3901 | **0.3911** | 0.3878 | 0.3843 | 0.3789 | 0.3767 | 0.3743 | 0.3753 |

## Query by query

Against the better of the two single retrievers on each query (nDCG@10). Hard queries: the hybrid puts nothing relevant in its top 10. A recall failure has nothing relevant in the top 100 either, so no reranker can fix it; a ranking failure has, so Phase 4's reranker might.

| Dataset | Queries | Hybrid above | Equal | Below | Hard: ranking | Hard: recall | Hard, though one retriever found something |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 300 | 20 | 220 | 60 | 27 | 8 | 7 |
| nfcorpus | 323 | 58 | 145 | 120 | 39 | 48 | 11 |
| fiqa | 648 | 127 | 350 | 171 | 114 | 87 | 34 |

The ten worst losses per dataset, with the documents involved: [scifact](losers_scifact.md), [nfcorpus](losers_nfcorpus.md), [fiqa](losers_fiqa.md). Hard queries: [hard_queries.json](hard_queries.json).

## Cost

FiQA, 200 test queries × 3, one at a time, convex tmm, alpha 0.95, depth 100. Excludes embedding the query (vectors come from the cache).

| Stage | p50 | p95 |
| --- | --- | --- |
| BM25: analyze and score | 0.36 ms | 0.78 ms |
| Dense: score every document | 1.51 ms | 2.04 ms |
| Candidates and fusion | 1.07 ms | 1.55 ms |
| Total | 3.03 ms | 4.08 ms |

Machine: Linux-6.8.0-149-generic-aarch64-with-glibc2.35, Python 3.12.15, NumPy 2.5.5, 4 CPUs, NumPy's default threads.
