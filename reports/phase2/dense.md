# Phase 2: dense retrieval with bge-small-en-v1.5

Test queries. Model `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405e`, [CLS] pooling, unit-length vectors, exact inner-product search. **Document** embeds each document whole, truncated at 512 tokens, as the model's published evaluation does. **Chunks** splits documents longer than 510 tokens into overlapping windows (64-token overlap) and scores a document by its best chunk. **BM25** uses the k1 and b each dataset got from Phase 1's tuning (scifact 1.2 and 0.75; nfcorpus 1.5 and 0.5; fiqa 0.9 and 0.75). **Published** is the model card's MTEB result ([source](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)).

| Dataset | Queries | Metric | BM25 | Dense, document | Dense, chunks | Published |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | 300 | nDCG@10 | 0.6830 | 0.7127 | 0.7128 | 0.7127 |
| scifact | 300 | R@100 | 0.9282 | 0.9417 | 0.9417 | 0.9417 |
| scifact | 300 | R@1000 | 0.9767 | 0.9967 | 0.9967 |  |
| scifact | 300 | MRR@10 | 0.6466 | 0.6822 | 0.6823 |  |
| nfcorpus | 323 | nDCG@10 | 0.3259 | 0.3438 | 0.3434 | 0.3431 |
| nfcorpus | 323 | R@100 | 0.2461 | 0.3111 | 0.3117 | 0.3114 |
| nfcorpus | 323 | R@1000 | 0.3699 | 0.6154 | 0.6151 |  |
| nfcorpus | 323 | MRR@10 | 0.5319 | 0.5302 | 0.5314 |  |
| fiqa | 648 | nDCG@10 | 0.2499 | 0.4035 | 0.4014 | 0.4034 |
| fiqa | 648 | R@100 | 0.5489 | 0.6963 | 0.6896 | 0.6955 |
| fiqa | 648 | R@1000 | 0.7577 | 0.8909 | 0.8899 |  |
| fiqa | 648 | MRR@10 | 0.3114 | 0.4879 | 0.4844 |  |

## Dense against BM25

Paired over the same test queries; 95% bootstrap interval and two-sided randomization p.

| Dataset | Mode | nDCG@10 difference | 95% interval | p | Queries dense wins | Queries BM25 wins |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | document | +0.0298 | -0.0012 to +0.0605 | 0.0611 | 73 | 51 |
| scifact | chunks | +0.0298 | -0.0014 to +0.0607 | 0.0615 | 73 | 50 |
| nfcorpus | document | +0.0179 | -0.0025 to +0.0380 | 0.0775 | 119 | 93 |
| nfcorpus | chunks | +0.0176 | -0.0027 to +0.0377 | 0.0828 | 121 | 92 |
| fiqa | document | +0.1535 | +0.1302 to +0.1767 | 0.0001 | 312 | 87 |
| fiqa | chunks | +0.1515 | +0.1280 to +0.1746 | 0.0001 | 310 | 87 |

## Corpus and search cost

| Dataset | Documents | Passages, chunks mode | Exact search p50 | Exact search p95 |
| --- | --- | --- | --- | --- |
| scifact | 5,183 | 5,666 | 0.34 ms | 0.45 ms |
| nfcorpus | 3,633 | 3,983 | 0.27 ms | 0.40 ms |
| fiqa | 57,638 | 60,589 | 2.84 ms | 3.69 ms |

## Chunking, decided on development queries

Chunks minus whole documents, nDCG@10, paired over each dataset's development queries. The default mode is chosen here, before looking at the test columns above.

| Dataset | Split | Queries | Difference | 95% interval | p |
| --- | --- | --- | --- | --- | --- |
| scifact | train | 809 | -0.0015 | -0.0048 to +0.0017 | 0.377 |
| nfcorpus | dev | 324 | -0.0005 | -0.0021 to +0.0010 | 0.519 |
| fiqa | dev | 500 | +0.0027 | -0.0025 to +0.0091 | 0.382 |

Search timings: Linux-6.8.0-149-generic-aarch64-with-glibc2.35, Python 3.12.15, NumPy 2.5.5, 4 CPUs, one query at a time with NumPy's default threads; single-threaded latency and approximate search are in [ann.md](ann.md). Embedding cost and cache behaviour are in [cache.md](cache.md).
