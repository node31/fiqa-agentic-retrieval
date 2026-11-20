# Walkthrough: reading the Part B results

## 1. Correctness first: sharded == single node

| strategy | single node | 3 shards | Δ |
|---|---|---|---|
| BM25 nDCG@10 | 0.2642 | 0.2642 | 0.0% |
| dense nDCG@10 | 0.3881 | 0.3881 | 0.0% |
| hybrid nDCG@10 | 0.4190 | 0.4190 | 0.0% |

Exact equality is a strong signal: any bug in merging, global stats or fusion order would show up as a
small drift. Compare with **local IDF** (no DFS phase): 0.2653. That's slightly *different* (here, even slightly
higher, within noise). The DFS phase is what makes the sharded system *identical*, not just "close".

## 2. Latency: 1 shard vs 3 shards (hybrid, through the router)

| | p50 | p95 |
|---|---|---|
| single node, in-process (no HTTP) | 20.7 | 24.5 |
| router + 1 shard (HTTP) | 23.4 | 26.3 |
| router + 3 shards (HTTP) | 28.6 | 40.5 |

Decompose it:
- in-process → 1 shard over HTTP: **+2.7 ms**, the cost of 2 HTTP hops (router→shard→router) + JSON.
- 1 → 3 shards: **+5 ms p50, +14 ms p95**: the DFS round trip (skipped for 1 shard) + waiting for the slowest of 3.
- BM25 alone shows the overhead most clearly: 1.2 ms in-process vs 10.7 ms through 3 shards. 90% is coordination.

## 3. Memory per shard

722–723 MB peak RSS per shard for every strategy. The breakdown (toy 11 arithmetic):
torch + MiniLM weights + Python ≈ 650 MB; dense vectors 19.2K × 1.5 KB = 28 MB; BM25 CSC + CSR ≈ 2 × 1.26M
postings × 8 B ≈ 20 MB; text ≈ 15 MB. **The model dominates.** A shard without an embedding model (if the router
embedded documents too) would be ~150 MB. That's a real design option for memory-constrained shards.

## 4. Live updates and migration

- `tests/test_integration_service.py`: 4 tests over 3 real shard processes: idempotency, freshness (searchable
  right after the ack, < 1 s for write + 3 searches), delete visibility in both indexes, hard-kill + WAL recovery + partial results.
- `results/migration_demo.json`: 2,045 queries during migrate → cutover → rollback, **0 failed**. Index sizes
  per shard are equal across models (19,206 / 19,288 / 19,260 docs, i.e. the fresh 19,167 / 19,254 / 19,217 plus the
  116 docs written during the run). All 116 written docs are found by both models.

## Questions to test yourself
1. Where would you look first if sharded nDCG were 0.4185 instead of 0.4190?
2. The load test's p50 was ~140 ms, but the benchmark's 28.6 ms. Both are "true". Explain.
3. How would memory per shard change if you moved document embedding to the router? What would get worse?
