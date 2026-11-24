# Phase 1: evaluation harness and BM25 from scratch

## Goal

Build the thing that judges every later phase, and a first retriever to judge.
The test of the harness is external: run BM25 through it and land on the
numbers Anserini publishes for the same datasets. If our numbers differ, either
the metrics or the retriever is wrong, and it is better to find out now than
after three phases have been compared with a broken ruler.

## What was built

| Part | File | Notes |
| --- | --- | --- |
| Data | `datasets.py` | SciFact, NFCorpus and FiQA-2018 from the BeIR datasets on the Hugging Face Hub, pinned to commits and SHA-256 checked. Downloads go to a temporary file and are renamed only after the checksum matches. |
| Analyzer | `text.py` | Regular-expression approximation of Lucene's StandardTokenizer, possessive removal, lower-casing, Lucene's 33 English stopwords, Porter stemming (PyStemmer). |
| Index | `bm25.py` | Postings in compressed-sparse-row form: one array of document numbers, one of term frequencies, one of offsets per term. Built by a stable sort of (term, document, frequency) triples. |
| Scoring | `bm25.py` | Lucene's BM25 variant. Per-posting weights are computed at construction; a query sums the weights of its terms' postings, multiplied by how often each term appears in the query. |
| Metrics | `metrics.py` | nDCG@10 with graded gains, MRR@10, recall at 100 and 1000. One value per query. |
| Statistics | `stats.py` | 95% paired bootstrap interval (10,000 resamples) and two-sided paired randomization test (10,000 sign flips). |
| Commands | `evaluate.py`, `tune.py` | Test-set evaluation against the reference; development-set tuning and ablation, with one final test comparison. |

## Matching Anserini

Anserini's BEIR "flat" baseline indexes title and text as one field with
Lucene's default English analyzer and scores with BM25, k1 = 0.9, b = 0.4. Its
configuration files record the expected scores and the index statistics,
including the total number of terms after analysis. That last number is a
direct check on the tokenizer.

| Dataset | nDCG@10 ours | with Lucene lengths | Anserini | Total terms ours | Anserini |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.6776 | 0.6789 | 0.6789 | 838,002 | 838,128 |
| NFCorpus | 0.3205 | 0.3214 | 0.3218 | 637,037 | 637,485 |
| FiQA-2018 | 0.2354 | 0.2360 | 0.2361 | 5,288,049 | 5,288,635 |

**The first gap, 0.0007 to 0.0013, was document lengths.** Lucene stores each
document's length in one byte: values below 24 exactly, larger ones with four
significant bits. A 137-term document is scored as 136 terms, a 1,000-term one
as 984. With that encoding emulated (`BM25(lengths="lucene")`), SciFact matches
to four decimals. The default stays exact, because exact is correct; the option
exists to explain the difference.

**The remaining gap is tokenisation.** Our tokenizer yields 0.01% to 0.07% fewer
terms than Lucene's, which follows the full Unicode word-break rules. The
leftover differences, 0.0004 on NFCorpus and 0.0001 on FiQA, are of that size.

Two conventions had to match as well. FiQA contains 38 empty documents, which
Lucene doesn't count, so N and the average length use non-empty documents only.
And both BEIR and Anserini drop a retrieved document whose id equals the query
id. In FiQA, 55 test query ids collide with unrelated document ids, so without
this rule our numbers would differ for a reason that has nothing to do with
ranking.

## What the tuning and ablation showed

All choices were made on development queries (SciFact's training queries,
since it has no development split). Test queries were used once per dataset.

- **Stemming:** without it, development nDCG@10 drops by 0.030 (SciFact), 0.022
  (NFCorpus) and 0.020 (FiQA), every interval excluding zero. Queries and
  documents use different inflections of the same words often enough to matter.
- **Stopwords:** keeping them changes nDCG@10 by less than 0.003, every interval
  spanning zero. BM25's idf already gives words in most documents almost no
  weight, so removing them mainly saves index space.
- **k1 and b:** the best development settings were (1.2, 0.75) for SciFact,
  (1.5, 0.5) for NFCorpus and (0.9, 0.75) for FiQA. On test, tuning gained
  +0.0054 on NFCorpus (p = 0.003) and +0.0145 on FiQA (p = 0.003). SciFact's
  +0.0053 is within noise. FiQA's documents are forum answers of very uneven
  length, which is consistent with it preferring stronger length normalisation.

Later phases use each dataset's tuned setting as their BM25 baseline, and say so.

## The four questions

**What breaks at 100 times the data?** Indexing holds every (term, document,
frequency) triple in Python lists before sorting: FiQA's 3.7 million postings
are fine, 370 million would not be. The fix is building per-block indexes and
merging them, the way Lucene writes segments. Query time grows with the length
of the query terms' postings; a common term's postings grow linearly with the
corpus, which is when dynamic pruning such as WAND or MaxScore becomes worth it.

**What happens on failure?** A download that fails or is corrupted leaves no
file under the real name, and the next run fetches it again. A changed upstream
file fails the checksum loudly instead of silently changing results.

**How do we know it is correct?** BM25 is checked against a slow,
straight-from-the-formula implementation on hand-built corpora for five
parameter settings; the metrics against hand-computed values and an oracle
ranking; and the whole pipeline against an independent implementation
(Lucene) on three datasets, to four decimal places on one of them.

**What would we change?** Replace the regular-expression tokenizer with a full
implementation of Unicode word breaking if exact Lucene parity ever mattered;
it currently costs at most 0.0004 nDCG@10.

## Limits

- Single-threaded and in memory; nothing persists between runs.
- Timings were measured in a 4-CPU Linux virtual machine on an Apple-silicon
  MacBook. They are indicative, and they vary from run to run.
- Three datasets, all English. Multiple comparisons (three tuning tests, six
  ablation tests) were not corrected for; the significant results all have p of
  0.007 or below.
