# Solutions: scaling

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** 768-d: flat = 100M × 3,072 B = 307 GB → ≥ 154 shards of 2 GB for vectors alone. PQ code size is set by m (bytes per vector), not by d: PQ-48 = 4.8 GB (≥ 3 shards, before BM25/text), PQ-96 = 9.6 GB.

**2.** If flat search runs at ~R vectors/ms per shard, half the 50 ms budget (25 ms) allows 25·R vectors. Measure R on your machine (see docs/15); with our measured R ≈ 11K/ms (`results/microbench.json`, 57K vectors in 5.3 ms) that's ≈ 275K vectors per shard. Above that, switch to ANN or add shards.

**3.** bge-small: 57,638 docs in 1,063 s ≈ 54 docs/s on 32 cores → 100M / 54 ≈ 1.85M s ≈ 21 days on this CPU. Use GPUs.
