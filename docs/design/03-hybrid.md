# Phase 3: hybrid retrieval

## Goal

Phase 2 ended with two retrievers that each win queries the other loses: on
FiQA, where dense is better by 0.154 nDCG@10, BM25 still wins 87 of 648 queries.
Picking the better one per query after the fact (the oracle) would score
0.4365 there, against dense's 0.4035. This phase combines them and answers four
questions with measurements: does a fused ranking beat *both* single retrievers,
with intervals, on all three datasets; which fusion method, and why; what
setting can the framework ship to a user whose data has no relevance
judgements; and where does the hybrid still fail, read query by query.

## What was built

| Part | File | Notes |
| --- | --- | --- |
| Fusion functions | `fusion.py` | Reciprocal rank fusion, score normalisation (min-max, theoretical min-max, z-score, none), weighted combination, and the final ordering with ties to the lower document position. Pure functions on arrays, no retrieval code. |
| Candidates and fusion | `hybrid.py` | `CandidateSet`: one query's candidates with both retrievers' scores and ranks. `FusionConfig`: a fusion setting. `HybridSearcher`: BM25 plus dense search over one collection, fused per query; chunked collections score a document as its best chunk. |
| Command | `evaluate_hybrid.py` | Scores 206 fusion settings on development queries, chooses one per dataset and one global, tests once, writes paired comparisons, the ten worst losses per dataset, a hard-query set, and latency. |

## Candidates first, fusion second

```
query ─┬─ analyze ─ BM25 scores of every document ─── top depth ─┐
       │                                                          ├─ union ─ CandidateSet ─ fuse(config) ─ top k
       └─ vector ── dense scores of every document ── top depth ─┘
```

Both searches are exact: BM25 scores every document that shares a term, and
dense search is one matrix product over all vectors. So for every candidate,
including one that only a single retriever returned, *both* scores are known.
Score-based fusion never has to guess a missing score, which is the usual
problem when one side is an approximate index that only reports its own top k.

Keeping candidates apart from fusion is what makes the evaluation cheap: each
query's candidates are collected once at depth 1,000, a depth-100 candidate set
is derived from it without searching again (`CandidateSet.truncate`), and the
206 settings are then pure array operations. All three datasets, all settings,
development and test, run in under a minute.

## Fusion methods

**Reciprocal rank fusion (RRF)**, Cormack, Clarke and Büttcher (2009):
score(d) = Σ w_r / (k + rank_r(d)), ranks counted from 1, a document missing
from a retriever's list contributing nothing. It uses ranks only, so the two
retrievers' scales never meet. k controls how fast credit decays down a list:
k = 1 gives the first place 1/2 and the tenth 1/11; k = 60 gives 1/61 and 1/70,
nearly flat.

**Convex combination**: score(d) = α·norm(dense) + (1 − α)·norm(BM25). The
scores must first be brought to a common scale:

| Normalisation | Formula, over one query's candidates | What it keeps |
| --- | --- | --- |
| min-max | (s − min) / (max − min) | Only the order and relative gaps; the top is always 1 and the bottom 0 |
| theoretical min-max (tmm) | (s − floor) / (max − floor), floor 0 for BM25 and −1 for cosine | How far each score is from the worst possible score |
| z-score | (s − mean) / std | Gaps in units of the spread |
| none | s | Kept to show what goes wrong without normalising |

Each setting also chooses the depth (100 or 1,000 per retriever). The grid:
RRF with k in {1, 5, 10, 20, 40, 60, 100, 200} and dense weight in
{0.3, …, 0.7}; convex with each of three normalisations and α from 0 to 1 in
steps of 0.05; both depths. 206 settings, with the textbook RRF (k = 60, equal
weights, depth 1,000) first so it wins any exact tie.

## Protocol

Settings are scored on development queries (SciFact has none, so its train
split stands in, as in Phases 1 and 2) by nDCG@10. Two are chosen before any
test query is evaluated: each dataset's best, and one global setting with the
best average over the three datasets. The global one is "the hybrid", because
it is what the framework can ship: a new user's memories come with no
judgements to tune on. Test queries are then scored once. The six
hybrid-against-single-retriever nDCG@10 comparisons are treated as one family
and their p-values Holm-corrected.

## Results

The global setting is **convex combination, theoretical min-max, α = 0.95,
depth 100**.

| Dataset | BM25 | Dense | RRF k=60 | Hybrid | Per dataset | Raw sum | Oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact | 0.6830 | 0.7127 | 0.7319 | **0.7414** | 0.7331 | 0.6844 | 0.7751 |
| NFCorpus | 0.3259 | 0.3438 | 0.3643 | **0.3716** | 0.3735 | 0.3370 | 0.3909 |
| FiQA-2018 | 0.2499 | 0.4035 | 0.3696 | **0.4184** | 0.4184 | 0.2518 | 0.4365 |

| Dataset | Hybrid minus BM25 (95% interval) | Hybrid minus dense (95% interval) | Holm p, vs dense |
| --- | --- | --- | --- |
| SciFact | +0.058 (+0.035 to +0.082) | +0.029 (+0.010 to +0.048) | 0.006 |
| NFCorpus | +0.046 (+0.032 to +0.061) | +0.028 (+0.016 to +0.041) | < 0.001 |
| FiQA-2018 | +0.168 (+0.149 to +0.188) | +0.015 (+0.004 to +0.026) | 0.009 |

The BM25 and dense columns reproduce Phase 1's tuned BM25 and Phase 2's dense
numbers to four decimals, so the fusion code doesn't change either retriever.
Every number above is byte-identical between an x86 Linux container and the
ARM Linux VM on the MacBook.

1. **The hybrid beats both retrievers on all three datasets**, and all six
   comparisons survive Holm's correction. It closes 45–59% of the gap between
   dense and the oracle (SciFact 0.029 of 0.062, NFCorpus 0.028 of 0.047, FiQA
   0.015 of 0.033).
2. **Adding raw scores is BM25 with extra steps.** A query's top BM25 score is
   typically 13 to 31 (median by dataset); cosines sit between about 0.5 and
   0.8. Summed, BM25 decides almost every order: 0.6844 against BM25's 0.6830 on SciFact. Normalisation is not a
   detail.
3. **Textbook RRF loses to dense alone on FiQA** (0.3696 against 0.4035). Equal
   votes for a retriever that is 0.15 nDCG@10 worse drag the better one down.
   Tuned RRF does better (dense weight 0.7, k = 5 on FiQA's development
   queries), but the best RRF average, 0.5085, stays below convex
   combination's 0.5159. All three datasets picked dense weight 0.7, the top of
   the grid, so a post-hoc check on development queries extended it to 0.95:
   no RRF setting beat 0.5085.
4. **Tuning per dataset bought nothing.** On FiQA it chose the global setting;
   on NFCorpus it was +0.002 (interval −0.001 to +0.005); on SciFact it was
   *worse* on test, −0.008 (interval −0.014 to −0.004). With 206 settings and
   a few hundred development queries, the per-dataset winner partly fits noise.
5. **Recall@100 barely improves over dense on two datasets**: +0.030 on SciFact,
   but +0.008 on NFCorpus and +0.011 on FiQA, both intervals including zero.
   The hybrid mostly reorders what dense already found. That matters for
   Phase 4, which can only rerank what the first stage returns.

### What α means

α = 0.95 reads as "almost all dense". It isn't: α multiplies *normalised*
scores, and the normalisations leave the two retrievers with different spreads.
Theoretical min-max maps cosines of 0.5 to 0.8 into a narrow band near 1 (a
standard deviation of about 0.02 to 0.03 across a query's candidates), while
BM25 spreads over 0 to 1 (about 0.16 to 0.19). Two measures put the settings on
one scale; both are in `reports/phase3/hybrid.md`, on test queries:

- **Pairwise share**, the direct one: among pairs of documents in the fused top
  20 that BM25 and dense order differently, the share the fusion orders as
  dense does. It measures who decides the ranking users see.
- **Proxy**: α·sd_dense / (α·sd_dense + (1 − α)·sd_BM25), with spreads taken
  over all candidates. Cheaper, and it overstates dense's say.

| Dataset | Hybrid: pairwise | Hybrid: proxy | Per-dataset winner: pairwise | Proxy | Winner |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.58 | 0.74 | 0.59 | 0.75 | min-max, α 0.75 |
| NFCorpus | 0.54 | 0.67 | 0.54 | 0.75 | z-score, α 0.75 |
| FiQA-2018 | 0.60 | 0.73 | 0.60 | 0.73 | tmm, α 0.95 |

Three datasets, three different normalisations, nearly one answer: where the
two retrievers disagree near the top, dense should win a little over half the
disputes. On every dataset, its own winner and the shipped setting agree to
within 0.01.

**Correction.** The first version of this write-up read the proxy as the share
and said dense drives "about three quarters of the ordering". The proxy is
about 15 points too high: the spread over all candidates is dominated by weak
candidates nobody sees, while near the top cosines bunch and BM25 decides more
of the close calls. That the settings agree survives; the number didn't.

This makes α = 0.95 specific to bge-small. A model whose cosines bunch
differently changes the spread and therefore what α means; ArguAna, below,
shows it happening. If the embedding model changes, α must be re-tuned, and
checking that the pairwise share stays near 0.55 to 0.6 is a quick sanity test.

### tmm against min-max

On development queries the two are tied: 0.5159 for tmm at α 0.95 against
0.5150 for min-max at α 0.75. Min-max also has the safer curve, within about 0.01
of its best from α 0.7 to 0.9, while tmm's peaks sharply at 0.95 with dense alone at 1.0 next to it.
The selection rule took the argmax, tmm. A post-hoc look at test (labelled as
such; it did not change the choice) shows tmm ahead on SciFact (+0.008,
interval +0.004 to +0.014) and NFCorpus (+0.009, +0.003 to +0.016), and level
on FiQA.

A plausible reason is that tmm adapts per query: when dense's cosines are
bunched, dense's normalised spread shrinks and BM25 gets more say. That would
help if bunched cosines meant dense was unsure. Measured on test, the
per-query effective share barely tracks which retriever was better: rank
correlations of −0.05, +0.15 and +0.04. So that explanation is not supported,
and the test advantage stays unexplained. The protocol has no robustness
tie-breaker; next time, settings within noise on development queries should be
decided by the flatter curve, decided before test is run.

## Held out: ArguAna

The three datasets above chose the setting and then reported it, so they can't
show whether it travels. ArguAna was held out from every choice: 8,674
arguments, 1,406 test queries and no development split, which is a new user's
situation exactly. Nothing was tuned on it: BM25 keeps Phase 1's defaults (k1
0.9, b 0.4), and the hybrid is `hybrid.SHIPPED` unchanged
(`evaluate_transfer.py`, `reports/phase3/transfer_arguana.md`).

| System | nDCG@10 | R@100 |
| --- | --- | --- |
| BM25, default k1 and b | 0.4046 | 0.9331 |
| Dense | 0.6061 | 0.9879 |
| RRF, k=60 | 0.5408 | 0.9872 |
| Hybrid | 0.6063 | 0.9900 |

- **Half the claim travels.** Hybrid minus dense is +0.0002 (interval −0.0054
  to +0.0059, Holm p 0.93); hybrid minus BM25 is +0.2017 (+0.1844 to +0.2190).
  The hybrid matches the better retriever and costs nothing, but it doesn't beat
  both. Across four datasets: never worse than the better single retriever,
  better on three.
- **The share says why.** On ArguAna the shipped α gives dense a pairwise share
  of 0.79, against 0.54 to 0.60 on the tuning datasets. Its cosines spread more
  relative to BM25, so the same α hands dense more of the close calls, and the
  hybrid behaves almost like dense alone. BM25 is weak here anyway: queries are
  whole arguments and the relevant document is the best counter-argument, same
  topic and opposite stance, so word overlap can't single it out.
- **Textbook RRF falls below dense again** (0.5408 against 0.6061): two of four
  datasets now show equal-weight RRF losing to its better input.
- **1,298 of the 1,406 queries are themselves documents** in the collection, and
  every system ranks the query's own document in its top 10. It is unjudged, so
  it's dropped from every ranking, the convention since Phase 1 and BEIR's.

## Reading the losers

For each dataset, the ten test queries where the hybrid falls furthest below
the better single retriever are written out with the relevant document, each
system's top three and the rank of each system's first relevant document
(`reports/phase3/losers_*.md`). I read all 30 and gave each one primary cause:

| Cause | SciFact | NFCorpus | FiQA | What happens |
| --- | --- | --- | --- | --- |
| A. Exact term only BM25 sees | 7 | 3 | 1 | A rare word, gene name, study name or verbatim phrase links query and document. Dense ranks it low; with dense winning most close calls, the hybrid only half follows BM25. |
| B. Vocabulary mismatch only dense bridges | 1 | 2 | 4 | The answer uses different words ("BMI and mortality" for "obesity decreases life quality"). BM25 scores it zero or near it, and documents with literal overlap push it down. |
| C. Near-tie, one or two judged documents | 2 | 3 | 2 | The relevant document moves one to three places. With one judged document, rank 1 to rank 3 halves nDCG@10. |
| D. Judgement or query artefact | 0 | 2 | 3 | Unjudged documents that look relevant, very broad queries with hundreds of judged documents, or an ambiguous query. |

| Dataset | Query | Cause | Note |
| --- | --- | --- | --- |
| SciFact | 1363, 128, 1362 | A | All three share one judged document, which mentions "arterioles, venules" once in passing in an angiogenesis study. BM25 finds the rare words; dense ranks it 162nd to 484th. The judgement itself is doubtful. |
| SciFact | 237, 684 | A | "clpC", a gene name. Dense prefers general Clp-protein papers. |
| SciFact | 674, 845 | A | The document contains the claim nearly verbatim ("LDL cholesterol has a causal role in the development of cardiovascular disease"). |
| SciFact | 870 | B | "Obesity decreases life quality" against a BMI-mortality meta-analysis; no shared terms. Dense rank 3, hybrid 28. |
| SciFact | 598, 127 | C | Relevant document 1st in dense, 2nd and 7th in BM25; the hybrid puts it 4th and 2nd. |
| NFCorpus | PLAIN-1721, PLAIN-1320, PLAIN-2620 | A | "NIH-AARP", "Physicians' Health Study II", "phytates": names and rare terms dense blurs into nearby topics. |
| NFCorpus | PLAIN-3231, PLAIN-2250 | B | "Meat & multiple myeloma": BM25 matches only the disease. "Tongue worm" against a swallowed-fish-bone case: no shared words. |
| NFCorpus | PLAIN-3085, PLAIN-441, PLAIN-44 | C | Small shuffles; in PLAIN-441 all three systems put a relevant document first. |
| NFCorpus | PLAIN-531, PLAIN-1409 | D | "alternative medicine" (308 judged) and "industrial toxins" (253 judged): BM25's literal matches are unjudged, so the comparison is mostly about judgement coverage. |
| FiQA | 6410 | A | "reconstituted" appears in a two-sentence answer; dense doesn't have it in its top 1,000, the hybrid ranks it 113th. |
| FiQA | 6890, 1920, 2076, 3932 | B | Dense finds the answer at rank 1 to 4; BM25 matches "clarification", "Canada", "income tax", "interest rate" in generic posts. |
| FiQA | 4447, 10558 | C | One-to-three-place moves. |
| FiQA | 3837, 9961 | D | One judged answer each; the documents ranked above it answer the same question and are unjudged. 9961's dense top hit begins "You can move money from a 403b to a 401k plan". |
| FiQA | 8959 | D | "Opening-day price" means an IPO; dense reads it as a daily opening price. |

Three conclusions:

- **The losses come from both directions.** Category A is the price of a
  dense-heavy weighting; category B happens despite it, because a zero BM25
  score is the floor of the normalised scale. No single global weight fixes
  both. That needs either a per-query weight or a model that reads the query
  and document together, which is Phase 4's reranker.
- **SciFact's losses are lexical, FiQA's are semantic**, matching where each
  single retriever is strong.
- **About a sixth of the "losses" are not retrieval errors** (category D):
  unjudged relevant documents and doubtful judgements. Sparse judgements make
  every system look worse than it is, and no change to retrieval fixes that.

## Hard queries

A hard query is one where the hybrid puts nothing relevant in its top 10. A
*recall* failure has nothing relevant in the top 100 either, so a reranker of
the top 100 cannot fix it. A *ranking* failure has something in the top 100.

| Dataset | Test queries | Hard: ranking | Hard: recall | Hard, though one single retriever scored above zero |
| --- | --- | --- | --- | --- |
| SciFact | 300 | 27 | 8 | 7 |
| NFCorpus | 323 | 39 | 48 | 11 |
| FiQA-2018 | 648 | 114 | 87 | 34 |

The 180 ranking failures are the set Phase 4's reranker will be judged on
(`reports/phase3/hard_queries.json`). NFCorpus has more recall failures than
ranking failures: its relevant documents are often linked to the query through
a nutrition website's citations rather than through their text, and no
first-stage method in this project reaches them.

## Cost

FiQA, 57,638 documents, one query at a time, 200 test queries three times
after a warm-up, on the 4-CPU Linux VM on the MacBook. The same code timed in
three separate runs:

| Stage | p50, three runs | p95, three runs |
| --- | --- | --- |
| BM25: analyze and score | 0.32 to 0.40 ms | 0.68 to 0.93 ms |
| Dense: score every document | 1.39 to 1.54 ms | 1.51 to 2.33 ms |
| Candidates and fusion | 0.97 to 1.09 ms | 1.25 to 2.00 ms |
| Total | 2.70 to 3.12 ms | 3.23 to 4.91 ms |

`reports/phase3/hybrid.md` holds the last run. The first version of this
write-up gave one run as if it were the number; the spread at p95 is as large
as the fusion step itself, so read these as an order of magnitude.

Query embedding is excluded: vectors come from the cache. Fusion adds about a
millisecond, which includes selecting each retriever's top 100 from 57,638
scores. The whole query is under 10% of the Phase 0 budget of 50 ms, which
leaves the budget to the embedding and the Phase 4 reranker.

## The four questions

**What breaks at 100 times the data?** The design leans on exact search: every
candidate's two scores come free because both retrievers score every document.
At 5.7 million documents, dense needs an approximate index (Phase 2 estimated
about 300 ms exact) and BM25 needs pruning, and then neither returns scores
outside its own top k. The fix keeps the design: score the candidates the other
retriever brought, directly. Dot products for a few hundred candidate vectors
cost microseconds; BM25 for named documents needs a forward index (document to
terms) or postings lookups. `CandidateSet` would be built from those calls
instead of from two full score arrays.

**What happens on failure?** A query with no BM25 match still gets dense's
ranking (tested). If the embedding model is unavailable, BM25 alone runs as a
degraded mode; that switch belongs to Phase 7's service layer. A corrupt or
missing tuning file stops the evaluation with a message rather than silently
comparing against default BM25, which would flatter the hybrid.

**How do we know it is correct?** RRF and every normalisation are tested
against hand-computed values; each single-retriever setting reproduces
`BM25.search` and dense search exactly; truncating a depth-1,000 candidate set
equals searching at depth 100; the report's single-retriever rows reproduce
Phases 1 and 2. Outputs are identical on x86 and ARM. 147 tests.

**What would we change?** Add a robustness tie-breaker to the selection rule
before running it. Try a per-query weight, for example from how many query
terms are rare, since categories A and B pull in opposite directions. And
hold out more datasets: ArguAna is one, and there the hybrid only matched
dense.

## Limits

- Three English datasets with sparse judgements chose the setting and report
  it (development queries for the choice, test queries for the report). One
  held-out dataset, ArguAna, is a first check of transfer, not proof.
- One reader labelled the 30 losers. The labels are judgement calls; the
  reports let anyone check them.
- The convex grid steps α by 0.05. Under tmm that is coarse: α 0.9, 0.95 and
  1.0 are effective dense shares of about 0.57, 0.74 and 1.
- Timings come from a 4-CPU Linux VM on an Apple-silicon MacBook.

## Reading

- Cormack, Clarke and Büttcher, "Reciprocal rank fusion outperforms Condorcet
  and individual rank learning methods", SIGIR 2009. The RRF formula and k = 60.
- Bruch, Gai and Ingber, "An analysis of fusion functions for hybrid
  retrieval", ACM TOIS 2023, [arXiv:2210.11934](https://arxiv.org/abs/2210.11934).
  Argues a tuned convex combination generally beats RRF, that RRF is sensitive
  to k, and that α needs few labelled queries to tune. Our results agree on
  the three tuning datasets.
