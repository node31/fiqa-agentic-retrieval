# Walkthrough: reading the Part A results

How to read `results/bench.json` and `results/ablation_*.json` like a reviewer would. Every number below
is copied from those files.

## 1. The ladder: BM25 → dense → hybrid

| | nDCG@10 | R@10 | MRR@10 | single-node p50 |
|---|---|---|---|---|
| BM25 | 0.264 | 0.314 | 0.322 | 1.2 ms |
| dense MiniLM | 0.388 | 0.467 | 0.453 | 18.8 ms |
| hybrid linear α=0.7 | 0.419 | 0.491 | 0.487 | 20.7 ms |

- **BM25 → dense: +0.124 nDCG.** FiQA questions are paraphrase-heavy ("prepay" vs "pay off early").
  Cost: 16× latency, almost all of it the query encoder.
- **dense → hybrid: +0.031 nDCG for +2 ms.** BM25 still contributes exact-term evidence (tickers, form
  numbers, product names) that dense blurs. That's cheap quality.

## 2. The fusion curve: a lesson in tuning

```
nDCG@10  linear: .287 (α=.1)  .336 (.3)  .402 (.5)  .418 (.6)  .419 (.7)  .406 (.8)  .401 (.9)
         RRF:    .373 (α=.5)  .387 (.6)  .395 (.7)  .404 (.8)  .406 (.9)
         dense alone .388
```
- The optimum is broad (0.6–0.7), so we're not overfitting a single point.
- Plain RRF (α = .5) is *below* dense: fusing a weak system with equal weight hurts.
- Even the best weighted RRF (0.406) loses to linear (0.419): score magnitudes carry information that ranks discard.

## 3. Stratified recall: who suffers?

| stratum | R@10 | n | read it as |
|---|---|---|---|
| short query (< 5 words) | 0.385 | 13 | hardest but tiny n: don't over-read |
| medium (5–15) | 0.491 | 403 | the bulk |
| long (> 15) | 0.510 | 84 | more context helps both retrievers |
| gold among 10% longest docs | 0.464 | 185 | long answers are harder (truncation at 256 tokens; BM25 length norm) |
| gold not longest | 0.508 | 315 | |

Fix ideas for long docs: split passages into overlapping chunks and score max-over-chunks; raise
`max_seq_length` for the cache (costs index time); lower BM25 b.

## 4. Reranking: a negative result worth knowing

| rerank depth | nDCG@10 | R@10 | p50 |
|---|---|---|---|
| none | 0.419 | 0.491 | 20.7 ms |
| top-10 | 0.406 | 0.491 | 339 ms |
| top-20 | 0.396 | 0.483 | 621 ms |
| top-50 | see `results/ablation_rerank_minilm.json` | | |

At k=10 Recall@10 is unchanged *by construction* (the same 10 docs, reordered), yet nDCG drops: the
cross-encoder orders them **worse** than the fused scores. Deeper k lets it pull in docs the hybrid ranked
lower, and recall falls too. A model trained on MS MARCO web search doesn't transfer well to long forum answers.

## 5. Latency: what is in 20 ms?

Single-node hybrid (`scripts/microbench.py`): 8.2 ms MiniLM query encode + 5.3 ms faiss over 57K vectors +
1.0 ms BM25 + fusion and Python glue for the rest.
To go faster: ONNX/int8 encoder, batching concurrent queries, or a smaller encoder; in a single node an
ANN index would save a few ms of the 5.3 ms scan. Sharding already cuts the scan to 1.4 ms per shard.

## Questions to test yourself
1. Why is Recall@10 identical for "none" and "rerank top-10"?
2. If you had to cut p50 to 10 ms, which component would you attack first, and what would you lose?
3. Which stratum would chunking help most? Predict the direction of change for each row.
