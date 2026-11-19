# Solutions: sharding

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Let d be in the global top-D and live on shard s. Every doc ranked above d globally is also ranked above d on its own shard; fewer than D docs on shard s are above d, so d is within shard s's local top-D. (Requires identical scoring across shards, i.e. global stats.)

**2.** p95 grows (tail amplification across 30 processes + 30 HTTP calls per phase); per-shard memory is dominated by torch + the model (~650 MB), so 30 shards × ~700 MB ≈ 21 GB total: memory per *document* gets much worse.

**3.** See docs/solutions/drills/drill_01.md.

**4.** You lose exactness under writes: df changes after inserts/deletes are invisible for up to the TTL, and scores drift from single-node. With hash sharding the error is tiny; the gain is ~4 ms p50 (one fan-out). See drill 8.
