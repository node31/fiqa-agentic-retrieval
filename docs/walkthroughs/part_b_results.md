# How I read the Part B results

## Sharded equals single node

| strategy | single node | 3 shards | Δ |
|---|---|---|---|
| BM25 nDCG@10 | 0.2642 | 0.2642 | 0.0% |
| dense nDCG@10 | 0.3881 | 0.3881 | 0.0% |
| hybrid nDCG@10 | 0.4190 | 0.4190 | 0.0% |

I checked correctness before speed. Exact equality is a strong signal, since any bug in merging, global statistics or
fusion order would show up as a small drift. With local IDF (no DFS phase) BM25 gives 0.2653: slightly different,
here even slightly higher, within noise. The DFS phase is what makes the sharded system identical to one node.

## Latency: 1 shard vs 3 shards (hybrid, through the router)

| | p50 (ms) | p95 (ms) |
|---|---|---|
| single node, in-process (no HTTP) | 20.7 | 24.5 |
| router + 1 shard (HTTP) | 23.4 | 26.3 |
| router + 3 shards (HTTP) | 28.6 | 40.5 |

In-process to 1 shard over HTTP adds 2.7 ms: two HTTP hops (router → shard → router) and JSON. Going from 1 to 3
shards adds 5 ms at p50 and 14 ms at p95, from the DFS round trip (skipped for 1 shard) and waiting for the slowest of
3. BM25 alone shows the overhead most clearly: 1.2 ms in-process vs 10.7 ms through 3 shards, so about 90% of it is
coordination.

## Memory per shard

Every strategy peaks at 722 to 723 MB RSS per shard. My estimate of where it goes: torch + MiniLM weights + Python
≈ 650 MB; dense vectors 19.2K × 1.5 KB = 28 MB; BM25 CSC + CSR ≈ 2 × 1.26M postings × 8 B ≈ 20 MB; text ≈ 15 MB. The
model dominates. If the router also embedded documents, a shard without an embedding model would need about 150 MB,
which is an option I would take for memory-constrained shards.

## Live updates and migration

`tests/test_integration_service.py` runs 4 tests over 3 real shard processes: idempotency, freshness (searchable right
after the ack, under 1 s for a write + 3 searches), delete visibility in both indexes, and hard-kill + WAL recovery +
partial results.

`results/migration_demo.json`: 2,045 queries during migrate → cutover → rollback, 0 failed. Index sizes per shard are
equal across models (19,206 / 19,288 / 19,260 docs, which is the fresh 19,167 / 19,254 / 19,217 plus the 116 docs
written during the run). Both models find all 116 written docs.

## Two write modes under heavy load (`results/write_load.json`)

`scripts/write_load.py`: real corpus, 3 shards, 8 writers + 2 readers, 30 s, once per mode (ADR 0013).

| | sync (apply, then ack) | async (ack, then apply) |
|---|---|---|
| writes acked / s | 57.8 | 94.2 |
| ack ms p50 / p95 | 128 / 181 | 78 / 120 |
| visible lag ms p95 / max | 0 / 0 | 466 / 583 (promise < 1,000) |
| search ms p50 under load | 131 | 184 |

Async takes embedding (~25 ms per doc) off the ack path, so writers get answers faster and push 1.6× more writes. The
price is freshness: a write becomes searchable about 0.2 to 0.6 s after its ack instead of immediately.

Searches are slower in async because more writes get through and the shards spend that CPU embedding them. Lock waits
were ≈ 0, so it is not contention.

My first async version lost to sync (47 writes/s, lag up to 1.3 s). Embedding in batches doesn't help on a CPU, and a
single applier thread was serial. I only found that out by measuring it.

## One query's time, split (`timings_ms`)

Every `/search` response splits the router's time into parts that add up to `latency_ms`. For hybrid on 3 shards in
scenario 04 (real corpus, light load): stats round trip ~15 ms, query-encode overhang 0 to 15 ms, slowest shard compute
~4 to 5 ms, network/queue ~8 to 12 ms, merge + fuse < 1 ms. The shards' own search work is the smallest part, so the
extra 5 ms at p50 for 3 shards is round trips, not search. Details in [../scenarios.md](../scenarios.md).
