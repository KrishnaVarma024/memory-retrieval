# Phase 2: embedding cost and the cache

Model `BAAI/bge-small-en-v1.5`, CPU only, 4 threads, batches of 32 sorted by length.

## Throughput

The same 1,000 FiQA documents (165,352 tokens including [CLS] and [SEP]), embedded into an empty cache and then again from it.

| Run | Seconds | Texts per second | Tokens per second |
| --- | --- | --- | --- |
| Cold (computed) | 39.5 | 25.3 | 4,189 |
| Warm (cached) | 0.002 | 655,971 | |

A cache hit is about 25,895 times faster than computing. Loading the cache from disk in a new process took 0.002 s for these vectors.

## Hit rates and what a full run costs

Counted from the texts, so they hold on any machine. A cold document run's only hits are duplicate documents. A chunk run after a document run hits on every document short enough to stay whole. Times are estimates from the throughput above.

| Dataset | Documents | Duplicate documents | Chunks | Chunk hit rate after a document run | Cold document run | Chunk run after it |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | 5,183 | 0 (0.00%) | 5,666 | 83.4% | 7 min | 1.2 min |
| nfcorpus | 3,633 | 40 (1.10%) | 3,983 | 83.1% | 5 min | 0.9 min |
| fiqa | 57,638 | 39 (0.07%) | 60,589 | 91.2% | 38 min | 7.8 min |

Machine: Linux-6.8.0-149-generic-aarch64-with-glibc2.35, Python 3.12.15, NumPy 2.5.5, 4 CPUs.
