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

## 5. Two write modes under heavy load (`results/write_load.json`)

`scripts/write_load.py`: real corpus, 3 shards, 8 writers + 2 readers, 30 s, once per mode (ADR 0013).

| | sync (apply, then ack) | async (ack, then apply) |
|---|---|---|
| writes acked / s | 57.8 | **94.2** |
| ack ms p50 / p95 | 128 / 181 | **78 / 120** |
| visible lag ms p95 / max | **0 / 0** | 466 / 583 (promise < 1,000) |
| search ms p50 under load | **131** | 184 |

How to read it:
- Async takes embedding (~25 ms per doc) off the ack path, so writers get answers faster and push 1.6× more writes.
- Freshness is the price: a write becomes searchable ~0.2–0.6 s after its ack instead of immediately.
- Searches slow down in async **because more writes get through**, and the shards spend that CPU embedding them.
  Lock waits were ≈ 0, so it isn't contention.
- The first async version lost to sync (47 writes/s, lag up to 1.3 s). Embedding in batches doesn't help on a CPU,
  and a single applier thread was serial. Measuring it is how we found out.

## 6. One query's time, split (`timings_ms`)

Every `/search` response splits the router's time into parts that add up to `latency_ms`. Hybrid, 3 shards,
scenario 04 (real corpus, light load): stats round trip ~15 ms · query-encode overhang 0–15 ms · slowest shard
compute ~4–5 ms · network/queue ~8–12 ms · merge + fuse < 1 ms. The shards' own search work is the smallest part, so
§2's "+5 ms p50 for 3 shards" is round trips, not search. Details: [../scenarios.md](../scenarios.md) §4.1.

## Questions to test yourself
1. Where would you look first if sharded nDCG were 0.4185 instead of 0.4190?
2. The load test's p50 was ~140 ms, but the benchmark's 28.6 ms. Both are "true". Explain.
3. How would memory per shard change if you moved document embedding to the router? What would get worse?
4. Async made searches slower under load, yet lock waits were ≈ 0. Where did the time go, and how would you test
   your answer? (Hint: compare the two modes at the *same* write rate.)
