# ADR 0013: Async apply as a second, switchable write mode

**Status:** accepted (alongside ADR 0007; sync stays the default)

## Context
ADR 0007 applies every write before acking it. That gives the strongest freshness guarantee (the first
search after the ack finds the write), but every write pays for its own embedding on the client's path,
one model call per doc, while holding the shard's write lock for the WAL fsync and the apply. The
common alternative in ingestion pipelines (Kafka-fed indexers, Milvus's log broker,
Elasticsearch's translog + refresh) is: **log → ack → apply in the background**. We want to compare
the two **under heavy write load**, without losing the sync implementation.

## Decision
Keep both. `APPLY_MODE=sync|async` per shard process (`ShardStore(apply_mode=...)`), default `sync`.

Async write path (`src/service/async_apply.py`):
1. validate the payload (nothing may fail after the ack);
2. version check against the latest **accepted** version (queued writes count);
3. **back-pressure:** wait while (a) the oldest unapplied write is older than `max_lag_s` (0.5 s), or
   (b) the predicted wait (batches queued ahead × measured seconds per batch) exceeds `max_lag_s`,
   or (c) `max_pending` writes are queued;
4. WAL append + fsync (text only; vectors don't exist yet) → enqueue → **ack** (`"visible": false`);
5. one applier thread: take ≤ `batch_max` (32) writes, embed them in **one batched call per model**
   outside the lock, apply them in queue order under one short write lock.

Freshness guarantee in async mode: **searchable within 1 s**, enforced by step 3, proven by
`tests/test_integration_async.py` and measured under load by `scripts/write_load.py`.

## Alternatives considered
- **Async without back-pressure:** the queue grows without bound under load, so freshness degrades
  without bound. Rejected.
- **Age-only back-pressure** (our first version): misses a fast burst. Every write in the burst is
  "young" when accepted, but the last one waits for every batch before it. Found by
  `test_async_backpressure_keeps_visible_lag_under_one_second`; fixed with the drain-time prediction (3b).
- **Store vectors in the async WAL:** impossible (they're computed after the ack). Consequence: replay
  re-embeds those records, so restart is slower after a burst.
- **Group commit** (one fsync for many writes): the natural next step for throughput. Not done.

## Consequences
- Async acks no longer wait for embedding. The applier embeds in batches, which is much cheaper per doc, so
  write throughput can rise. Searches see a write within ~`max_lag_s` + one batch, not instantly.
- Extra machinery that sync doesn't need: the accepted-version map, the queue, the applier thread, back-pressure,
  `flush` (`POST /admin/flush`), and lag metrics in `/admin/status`.
- A write that fails to apply after its ack marks the shard unhealthy (crash-only); restart replays the WAL.
  A production system would add a dead-letter queue.
- Measured comparison: `results/write_load.json`, ch07 §7.4 "Two write modes".
