# How I read the Part A results

These are the numbers from `results/bench.json` and `results/ablation_*.json`, and what I take from each.

## BM25, then dense, then hybrid

| | nDCG@10 | R@10 | MRR@10 | single-node p50 |
|---|---|---|---|---|
| BM25 | 0.264 | 0.314 | 0.322 | 1.2 ms |
| dense MiniLM | 0.388 | 0.467 | 0.453 | 18.8 ms |
| hybrid linear α=0.7 | 0.419 | 0.491 | 0.487 | 20.7 ms |

Going from BM25 to dense adds 0.124 nDCG. FiQA questions are full of paraphrase ("prepay" vs "pay off early"), which
exact-term matching misses. It costs 16× the latency, and almost all of that is the query encoder.

Going from dense to hybrid adds another 0.031 nDCG for about 2 ms. BM25 still brings exact-term evidence (tickers, form
numbers, product names) that dense blurs, so I kept it.

## The fusion curve

```
nDCG@10  linear: .287 (α=.1)  .336 (.3)  .402 (.5)  .418 (.6)  .419 (.7)  .406 (.8)  .401 (.9)
         RRF:    .373 (α=.5)  .387 (.6)  .395 (.7)  .404 (.8)  .406 (.9)
         dense alone .388
```

The optimum is broad (0.6 to 0.7), so α = 0.7 is not a lucky single point. Plain RRF (α = .5) scores below dense alone:
giving a weaker system equal weight pulls the stronger one down. Even the best weighted RRF (0.406) loses to linear
(0.419), because score magnitudes carry information that ranks throw away. That is why I chose linear fusion.

## Stratified recall

| stratum | R@10 | n | how I read it |
|---|---|---|---|
| short query (< 5 words) | 0.385 | 13 | hardest, but n is too small to lean on |
| medium (5 to 15) | 0.491 | 403 | the bulk of the questions |
| long (> 15) | 0.510 | 84 | more context helps both retrievers |
| gold among 10% longest docs | 0.464 | 185 | long answers are harder (truncation at 256 tokens, BM25 length normalisation) |
| gold not longest | 0.508 | 315 | |

For long docs I would split passages into overlapping chunks and score each doc by its best chunk, raise
`max_seq_length` for the cache (more index time), or lower BM25's b.

## Reranking made it worse

| rerank depth | nDCG@10 | R@10 | p50 |
|---|---|---|---|
| none | 0.419 | 0.491 | 20.7 ms |
| top-10 | 0.406 | 0.491 | 339 ms |
| top-20 | 0.396 | 0.483 | 621 ms |
| top-50 | 0.377 | 0.463 | 1246 ms |

At k = 10, Recall@10 cannot change: it is the same 10 docs, reordered. nDCG still drops, so the cross-encoder orders them
worse than the fused scores do. At deeper k it pulls in docs the hybrid ranked lower, and recall falls too. My reading:
a model trained on MS MARCO web search does not transfer well to long forum answers. I ship without a reranker.

## What is in 20 ms

Single-node hybrid (`scripts/microbench.py`): 8.2 ms MiniLM query encode, 5.3 ms faiss over 57K vectors, 1.0 ms BM25,
and fusion plus Python glue for the rest. To go faster I would try an ONNX/int8 encoder, batching concurrent queries,
or a smaller encoder. On a single node an ANN index would save only a few ms of the 5.3 ms scan, and sharding already
cuts the scan to 1.4 ms per shard.
