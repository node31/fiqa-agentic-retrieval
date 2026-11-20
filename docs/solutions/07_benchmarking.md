# Solutions: benchmarking

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** To keep the measured run honest: no dev query has been seen before it is timed, so no OS page cache, tokenizer cache or any future result cache could favour the dev set.

**2.** p50 rises a little (one more process per query, more coordination); p95 rises more: the router waits for the slowest of 10 shards (tail amplification), while per-shard compute shrinks only marginally at this corpus size.

**3.** Run it; typically 1 thread → ~2× slower encode; 16 threads → no better than 4 for single queries. Explain via work per batch vs thread overhead.

**4.** Add `'p99_ms': percentile(ms, 99)`. With 500 samples, nearest-rank p99 (`src/util.py`: rank ceil(0.99 × 500) = 495 of 500) is the 6th-slowest value: very noisy. You'd want several thousand samples, or report p99 with a bootstrap CI.
