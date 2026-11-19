# 04 · IR metrics: nDCG@10, Recall@10, MRR, Coverage@10

**Code:** `src/metrics.py` · **Toy:** 07 · **Tests:** `tests/test_metrics.py`

## 1. Intuition

Each metric answers a different user question about one ranked list:
- **Recall@10**: *"Of everything relevant, how much made the top 10?"* For RAG this matters most, because the LLM reads all 10.
- **MRR@10**: *"How far down is the first good answer?"* (1 if at rank 1, ½ at rank 2, 0 if not in the top 10)
- **nDCG@10**: *"Is the good stuff near the top?"* It rewards each relevant doc with a log-discount by rank, normalised by the best possible ordering.
- **Coverage@10** (ours, Part C): *"For a two-part question, did we find something for BOTH parts?"*

## 2. The math (single query; average over queries)

- Recall@k = |top-k ∩ R| / |R|
- MRR@k = 1/rank₁ (rank of first relevant doc, if ≤ k, else 0)
- DCG@k = Σᵢ₌₁ᵏ (2^{relᵢ} − 1)/log₂(i+1); nDCG@k = DCG@k / IDCG@k
- Coverage@k = 𝟙[top-k ∩ R_a ≠ ∅ ∧ top-k ∩ R_b ≠ ∅]

**Worked example** (toy 07, k = 5): relevant = {g1, g2}; ranked = [x, x, x, g1, g2].
Recall = 2/2 = 1.0, MRR = 1/4 = 0.25, DCG = 1/log₂5 + 1/log₂6 = 0.431 + 0.387 = 0.818,
IDCG = 1 + 1/log₂3 = 1.631 → nDCG = **0.501**. Compare ranked = [g1, x, x, x, x]: Recall 0.5,
MRR 1.0, nDCG 0.613. Higher nDCG and MRR, *lower* recall. Metrics disagree by design.

## 3. Diagram

```
rank:     1     2     3     4     5
gain:    [g]   [ ]   [ ]   [g]   [ ]
discount 1/1  1/1.58 1/2  1/2.32 1/2.58     DCG = 1 + 0.431
```

## 4. How this repo implements it

- Pure Python, ~60 lines, each function with a doctest.
- **IDCG uses all judged-relevant docs** (a common bug is to use only the retrieved ones, which inflates nDCG).
- Duplicates are removed before scoring. Queries without relevant docs are skipped and `n` is reported.
- Stratified reporting in `src/eval.py`: Recall@10 by query length and by "gold passage is among the
  10% longest docs". Always with group sizes, because 13 short queries are noise.

## 5. Run the toy

`python examples/07_metrics_by_hand.py`: three queries, every number computed by hand and by `src/metrics.py`.

## 6. Observe it in the real system

From `results/bench.json` (hybrid-linear-minilm, dev):

| stratum | Recall@10 | n |
|---|---|---|
| short query (< 5 words) | 0.385 | 13 |
| medium (5–15) | 0.491 | 403 |
| long (> 15) | 0.510 | 84 |
| gold in 10% longest docs | 0.464 | 185 |
| gold not in longest | 0.508 | 315 |

Reading: short queries are hardest (little context for either retriever), though n = 13 is too small to trust.
Long gold passages are harder (truncation at 256 tokens + BM25 length normalisation both penalise them).

## 7. Pitfalls and interview questions

- **Comparing nDCG across datasets** is meaningless: it depends on how many relevant docs each query has.
- **Unjudged ≠ irrelevant**: FiQA's sparse qrels undercount true positives, so every system looks worse than it is.
- **Averaging percentages of different-sized groups** (Simpson's paradox): report n.
- *Q: Which metric would you optimise for RAG?* Recall@k at the context size you feed the LLM, plus
  latency. Precision matters less because the LLM can ignore junk (up to a point).

## 8. Exercises

1. (Concept) Can MRR be 1 while Recall@10 is 0.25? Construct the example.
2. (Predict) If every query had exactly 1 relevant doc, how would nDCG@10 relate to MRR@10?
3. (Do) Add `precision_at_k` to `src/metrics.py` with a doctest and a unit test.
4. (Do) `results/runs/*.json` store only the top-10 per query. Change `src/eval.py` to also store the top-100,
   then answer: is BM25's recall problem in *ranking* (gold is at rank 11–100) or in *retrieval* (gold not in the top-100 at all)?

Solutions: [solutions/04_metrics.md](solutions/04_metrics.md)

## 9. Further reading

- Järvelin & Kekäläinen, *Cumulated gain-based evaluation of IR techniques* (2002)
- Manning et al., *IR book* ch. 8 · `pytrec_eval` / `ranx` for production use
