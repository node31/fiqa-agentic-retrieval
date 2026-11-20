# ADR 0013: Async apply as a second, switchable write mode

**Status:** accepted (alongside ADR 0007; sync stays the default). Revised after the first real-corpus load test.

## Context
ADR 0007 applies every write before acking it. That gives the strongest freshness guarantee (the first
search after the ack finds the write), but every write pays for its own embedding on the client's path
while holding the shard's write lock for the WAL fsync and the apply. The common alternative in ingestion
pipelines (Kafka-fed indexers, Milvus's log broker, Elasticsearch's translog + refresh) is:
**log → ack → apply in the background**. We want to compare the two **under heavy write load**, without
losing the sync implementation, and keep a **1 s freshness** promise in async mode.

## Decision
Keep both. `APPLY_MODE=sync|async` per shard process (`ShardStore(apply_mode=...)`), default `sync`.

Async write path (`src/service/async_apply.py`):
1. validate the payload (nothing may fail after the ack);
2. version check against the latest **accepted** version (queued writes count);
3. **back-pressure:** wait while (a) the oldest unapplied write is older than `max_lag_s` (0.5 s), or
   (b) the **predicted time until this write is applied** exceeds `max_lag_s`, or (c) `max_pending` writes
   are queued. The prediction counts **docs**, not batches: remaining time of the batch in flight +
   `ceil(docs ahead × models / embed_workers)` parallel rounds × measured seconds per round + a fixed
   per-batch cost. The estimate rises fast and falls slowly;
4. WAL append + fsync (text only; vectors don't exist yet) → enqueue → **ack** (`"visible": false`);
5. one applier thread: take ≤ `batch_max` (32) writes, embed them **in parallel, one doc per model call**
   (`embed_workers`, default 8, env `ASYNC_EMBED_WORKERS`), outside the lock; apply them in queue order
   under one short write lock.

Freshness guarantee in async mode: **searchable within 1 s**: 0.5 s predicted budget + 0.5 s headroom for
prediction error. Proven by `tests/test_integration_async.py`, measured under load by `scripts/write_load.py`.

## What the load test taught us (`scripts/write_load.py`, real corpus, 3 shards, 8 writers + 2 readers, 30 s)

| | sync | async v1 (one batched call, 1 thread) | **async v2 (shipped)** |
|---|---|---|---|
| writes acked / s | 57.8 | 47.1 | **94.2** |
| ack ms p50 / p95 | 128 / 181 | 85 / 649 | **78 / 120** |
| visible lag ms p95 / max | 0 / 0 | 1,071 / **1,306** ❌ | **466 / 583** ✓ |
| write errors | 0 | 1 | 0 |
| search ms p50 under load | **131** | 112 | 184 |

Source: `results/write_load.json` (v2 and sync); v1 numbers are from the first run, before the fix. v2 was run
3 times: 92.6 / 90.8 / 94.2 writes/s, max visible lag 460 / 542 / 583 ms.

**Why v1 failed.** The applier's time was ~100% embedding (lock wait ≤ 8 ms, apply ≤ 26 ms; per-batch embed
p95 360–600 ms, max 985 ms). Two wrong assumptions:
- *"Batching is async's big win."* False on CPU. Same 256 random FiQA passages, MiniLM, 4 torch threads:

  | docs per model call | 1 | 2 | 4 | 8 | 16 | 32 | 256 |
  |---|---|---|---|---|---|---|---|
  | ms per doc | 25.6 | 26.1 | 27.6 | 31.5 | 30.9 | 32.3 | 24.7 |

  Small batches are padded to their longest passage, so they cost *more* per doc; a huge call (sorted by length
  internally) saves only ~4%. Sync's concurrent requests embed **in parallel**; v1's single applier thread
  embedded **serially**. So v1 was slower than sync.
- *"Predicted wait = batches ahead × average batch time."* Batch cost scales with docs per batch (3 to 18 under
  load); a big batch after small ones overshot the estimate, so lag reached 1.3 s.

**Why search latency is higher in async.** Not a lock problem (lock waits ≈ 0). At 94 writes/s the shards spend
63% more CPU on embedding than at 58 writes/s, and searches share the same cores. At an *equal* write rate the
difference would be smaller (not measured).

## Alternatives considered
- **Async without back-pressure:** the queue grows without bound under load, so freshness degrades without
  bound. Rejected.
- **Age-only back-pressure** (first version): misses a fast burst. Every write in the burst is "young" when
  accepted, but the last one waits for every batch before it. Found by
  `test_async_backpressure_keeps_visible_lag_under_one_second`; fixed with a drain-time prediction.
- **Batch-count prediction** (v1): fails because batch cost scales with docs. Replaced by the doc/round model.
- **One batched embed call per batch** (v1): slower than one doc per call on CPU (table above). Replaced by
  parallel single-doc calls. On a GPU, batching would likely win; `embed_chunk` is a parameter for that case.
- **Store vectors in the async WAL:** impossible (they're computed after the ack). Consequence: replay
  re-embeds those records, so restart is slower after a burst.
- **Group commit** (one fsync for many writes): the natural next step for ack latency. Not done.

## Consequences
- Async acks no longer wait for embedding: 1.6× the write throughput of sync and faster acks, at the price of
  a visible lag of ~0.2 s typical, < 0.6 s worst observed (promise: < 1 s).
- Extra machinery that sync doesn't need: the accepted-version map, the queue, the applier thread + embed
  pool, the cost model, back-pressure, `flush` (`POST /admin/flush`), lag/batch-timing metrics in `/admin/status`.
- A write that fails to apply after its ack marks the shard unhealthy (crash-only); restart replays the WAL.
  A production system would add a dead-letter queue.
- Both modes now reject empty upsert text, and the router answers **422** at once (before, a shard's 400 was
  retried 3× and surfaced as a misleading "503 shards unavailable"; FiQA has 38 empty passages).
- Walkthrough with the numbers: ch07 §7.4 "Two write modes" (`docs/from_zero/07_part_b_distributed.md`).
