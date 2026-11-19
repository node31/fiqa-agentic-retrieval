# Solutions: vector indexes

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Each shard's flat scan is independent; the exact global top-k is the merge of per-shard exact top-k (docs/08 rule 1). No approximation is introduced by partitioning.

**2.** HNSW latency grows roughly logarithmically (more layers/hops), i.e. barely changes; flat doubles.

**3.** Typically efSearch around 64–128 for M = 16–32 on random data; real embeddings reach 0.99 at lower ef. The answer is in your toy output.

**4.** 100M × 384 × 4 B = 153.6 GB flat; PQ-48: 100M × 48 B = 4.8 GB (+ ids and centroids).
