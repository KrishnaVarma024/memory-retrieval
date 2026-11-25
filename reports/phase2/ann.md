# Phase 2: exact against approximate search on fiqa

57,638 document vectors (384 dimensions), 648 test queries, one thread, one query at a time. Recall is the share of the exact top k that the approximate search also returns.

## Exact search

| Implementation | p50 | p95 | nDCG@10 |
| --- | --- | --- | --- |
| Ours (NumPy) | 2.65 ms | 3.80 ms | 0.4035 |
| FAISS IndexFlatIP | 4.02 ms | 5.50 ms | 0.4035 |

Both are exact. Their top 101 is identical for 632 of 648 queries; the rest differ only where two scores are equal or within about a millionth, which the two implementations round and break differently.

## HNSW (M = 32, efConstruction = 200, built in 27 s on one thread)

Latency is for a top-10 search. Recall@100 needs a top-100 search, which explores at least 100 candidates whatever ef_search says, so it is shown only from ef_search 160.

| ef_search | Recall@10 | Recall@100 | nDCG@10 | p50 | p95 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.8207 |  | 0.3710 | 0.054 ms | 0.088 ms |
| 20 | 0.9015 |  | 0.3868 | 0.074 ms | 0.103 ms |
| 40 | 0.9557 |  | 0.3957 | 0.131 ms | 0.214 ms |
| 80 | 0.9818 |  | 0.3971 | 0.227 ms | 0.314 ms |
| 160 | 0.9923 | 0.9714 | 0.4005 | 0.421 ms | 0.599 ms |
| 320 | 0.9974 | 0.9911 | 0.4020 | 0.785 ms | 1.090 ms |
| 640 | 0.9991 | 0.9977 | 0.4035 | 1.551 ms | 2.040 ms |

## As the collection grows

Padded with perturbed copies of real vectors (Gaussian noise, sd 0.03 per dimension, renormalised). 200 queries per size, each timed 3 times. HNSW uses the smallest ef_search that reached recall@10 of at least 0.95 above; it is measured up to 200,000 vectors.

| Vectors | Exact p50 | Exact p95 | HNSW p95 | HNSW recall@10 | HNSW build (all cores) |
| --- | --- | --- | --- | --- | --- |
| 57,638 | 2.7 ms | 3.7 ms | 0.18 ms | 0.9570 | 10 s |
| 100,000 | 4.4 ms | 6.3 ms | 0.21 ms | 0.9615 | 19 s |
| 200,000 | 9.7 ms | 12.4 ms | 0.31 ms | 0.9575 | 49 s |
| 400,000 | 18.2 ms | 21.3 ms |  | | |

A straight-line fit gives exact search about 5.1 ms of p95 latency per 100,000 vectors, crossing the 50 ms budget at about 952,410 vectors.

Machine: Linux-6.8.0-149-generic-aarch64-with-glibc2.35, Python 3.12.15, NumPy 2.5.5, 4 CPUs.
