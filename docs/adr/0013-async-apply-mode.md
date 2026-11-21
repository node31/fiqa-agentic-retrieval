# ADR 0013: Async apply as a second, switchable write mode

Status: accepted, alongside [ADR 0007](0007-sync-apply-wal.md); sync stays the default. I revised it after the first real-corpus load test.

## Context
In sync mode ([ADR 0007](0007-sync-apply-wal.md)) every write pays for its own embedding on the client's path and holds the shard's write lock for the WAL fsync and the apply. Ingestion pipelines (Kafka-fed indexers, Milvus's log broker, the Elasticsearch translog and refresh) usually log, ack, then apply in the background. I wanted to compare the two under heavy write load, keep the sync implementation, and keep a 1 s freshness promise in async mode.

## Decision
I kept both. `APPLY_MODE=sync|async` is set per shard process (`ShardStore(apply_mode=...)`), default `sync`.

The async write path (`src/service/async_apply.py`):
1. Validate the payload, because nothing may fail after the ack.
2. Check the version against the latest accepted version (queued writes count).
3. Back-pressure: wait while the oldest unapplied write is older than `max_lag_s` (0.5 s), or the predicted time until this write is applied exceeds `max_lag_s`, or `max_pending` writes are queued. The prediction counts docs: the remaining time of the batch in flight, plus `ceil(docs ahead × models / embed_workers)` parallel rounds × measured seconds per round, plus a fixed per-batch cost. The estimate rises fast and falls slowly.
4. Append the text to the WAL and fsync (vectors don't exist yet), enqueue, ack with `"visible": false`.
5. One applier thread takes up to `batch_max` (32) writes, embeds them in parallel with one doc per model call (`embed_workers`, default 8, env `ASYNC_EMBED_WORKERS`) outside the lock, then applies them in queue order under one short write lock.

The async promise is searchable within 1 s: 0.5 s predicted budget plus 0.5 s headroom for prediction error. `tests/test_integration_async.py` proves it, and `scripts/write_load.py` measures it under load.

## Load test results
`scripts/write_load.py`, real corpus, 3 shards, 8 writers and 2 readers, 30 s:

| | sync | async v1 (one batched call, 1 thread) | async v2 (shipped) |
|---|---|---|---|
| writes acked / s | 57.8 | 47.1 | 94.2 |
| ack ms p50 / p95 | 128 / 181 | 85 / 649 | 78 / 120 |
| visible lag ms p95 / max | 0 / 0 | 1,071 / 1,306 (over 1 s) | 466 / 583 |
| write errors | 0 | 1 | 0 |
| search ms p50 under load | 131 | 112 | 184 |

Sync and v2 come from `results/write_load.json`. The v1 numbers are from the first run, before the fix. I ran v2 3 times: 92.6 / 90.8 / 94.2 writes/s, max visible lag 460 / 542 / 583 ms.

Why v1 failed: the applier spent almost all its time embedding (lock wait ≤ 8 ms, apply ≤ 26 ms, per-batch embed p95 360 to 600 ms, max 985 ms). I had made two wrong assumptions.

- I assumed batching was async's big win. On CPU it isn't. Same 256 random FiQA passages, MiniLM, 4 torch threads:

  | docs per model call | 1 | 2 | 4 | 8 | 16 | 32 | 256 |
  |---|---|---|---|---|---|---|---|
  | ms per doc | 25.6 | 26.1 | 27.6 | 31.5 | 30.9 | 32.3 | 24.7 |

  Small batches are padded to their longest passage, so they cost more per doc, and one huge call (sorted by length internally) saves only ~4%. Sync's concurrent requests embed in parallel; v1's single applier thread embedded serially, so v1 was slower than sync.
- I predicted the wait as batches ahead × average batch time. Batch cost scales with docs per batch (3 to 18 under load), so a big batch after small ones overshot the estimate and lag reached 1.3 s.

Search latency is higher in async because of CPU, not locks (lock waits ≈ 0). At 94 writes/s the shards spend 63% more CPU on embedding than at 58 writes/s, and searches share the same cores. At an equal write rate the gap would be smaller; I have not measured it.

## Alternatives considered
| Option | Why I rejected it |
|---|---|
| Async without back-pressure | The queue grows without bound under load, and so does staleness |
| Age-only back-pressure (my first version) | Misses a fast burst: every write is young when accepted, but the last one waits for every batch before it. `test_async_backpressure_keeps_visible_lag_under_one_second` caught it |
| Batch-count prediction (v1) | Batch cost scales with docs; replaced by the doc/round model |
| One batched embed call per batch (v1) | Slower than one doc per call on CPU (table above). On a GPU batching would likely win; `embed_chunk` is a parameter for that case |
| Vectors in the async WAL | Impossible, since they are computed after the ack. So replay re-embeds those records and restart is slower after a burst |
| Group commit (one fsync for many writes) | Not rejected, just not built. It is the natural next step for ack latency |

## Consequences
- Async acks don't wait for embedding: 1.6× the write throughput of sync and faster acks, for a visible lag of ~0.2 s typical and < 0.6 s worst observed (promise: < 1 s).
- Async needs machinery sync doesn't: the accepted-version map, the queue, the applier thread and embed pool, the cost model, back-pressure, `flush` (`POST /admin/flush`), and lag and batch-timing metrics in `/admin/status`.
- A write that fails to apply after its ack marks the shard unhealthy (crash-only), and a restart replays the WAL. A production system would add a dead-letter queue.
- Both modes now reject empty upsert text, and the router answers 422 at once. Before, a shard's 400 was retried 3 times and surfaced as a misleading "503 shards unavailable". FiQA has 38 empty passages.
- The from-zero walkthrough of both write modes is in [from_zero/07_part_b_distributed.md](../from_zero/07_part_b_distributed.md).
