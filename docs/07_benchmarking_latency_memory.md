# 07 · Benchmarking: latency percentiles, cold vs warm, memory

**Code:** `src/eval.py`, `scripts/bench.py`, `src/util.py` · **Toys:** 10, 11 · **Output:** `results/bench.json`

## 1. Intuition

A benchmark is an experiment. It needs a **protocol** (what is measured, from where, after what
warmup), **raw data** (not just averages), and **context** (hardware). Numbers without protocol
are anecdotes.

## 2. The mechanics

- **p50 / p95 / p99**: sort the latencies; pN = the value N% of requests are faster than. We use the
  nearest-rank method (`src/util.py::percentile`).
- **Cold**: the first 20 queries of a *fresh process*. **Warm**: all 500 dev queries after a 100-query
  warmup on **train** queries (so no dev query is seen before it's timed).
- **Peak RSS**: OS high-water mark of resident memory (Windows `peak_wset`, Linux `ru_maxrss`).

## 3. Diagram: what one timed query includes (service mode)

```
client ──HTTP──> router: tokenize + embed (8 ms)  ┐
                     ├─ DFS stats  ──> 3 shards ──┤ ~2-3 ms (parallel)
                     ├─ search     ──> 3 shards ──┤ ~3-5 ms (parallel, slowest wins)
                     └─ merge + fuse (<1 ms) ─────┘
client <──────────────── JSON ────────────────────  total p50 28.6 ms, p95 40.5 ms
```

## 4. How this repo implements it

- `scripts/bench.py` runs **each configuration in its own process** (so "cold" is real) and restarts the
  cluster per service configuration.
- Torch is pinned to **4 intra-op threads per process**. 4 processes × 32 threads on 32 cores would
  thrash. This was measured and decided once, then documented.
- Latency is measured **by the client, end to end**, so for service mode it includes HTTP to the router,
  which is what the 50 ms constraint refers to.
- Machine: AMD EPYC 7763 (16 physical / 32 logical cores visible), 128 GB RAM, Windows 11, Python 3.13.

## 5. Run the toys

```bash
python examples/10_latency_percentiles.py   # mean vs p50 vs p95; tail amplification with fan-out
python examples/11_memory_measurement.py    # RSS vs tracemalloc vs arithmetic
```

## 6. Observe it in the real system (`results/bench.json`, dev)

| config | warm p50 / p95 (ms) | cold p50 / p95 (ms) | peak RSS (MB) |
|---|---|---|---|
| bm25 (local) | 1.2 / 2.0 | 1.2 / 1.9 | 511 (process) |
| dense-minilm (local) | 18.8 / 20.8 | 19.1 / 21.2 | 806 |
| hybrid-linear-minilm (local) | 20.7 / 24.5 | 20.1 / 23.3 | 812 |
| **svc3 hybrid-linear-minilm** (router + 3 shards) | **28.6 / 40.5** | 29.5 / 33.0 | **723 per shard** |
| svc1 hybrid-linear-minilm (router + 1 shard) | 23.4 / 26.3 | 22.9 / 25.1 | 775 |

**Cold ≈ warm.** We expected a big cold penalty and found almost none, because we *removed* the cold costs
on purpose. The model is loaded and a warmup encode runs at startup (router `on_event("startup")`),
vectors are copied into RAM (not lazily mmapped), and the httpx client keeps connections alive.
What warmup hides (`results/microbench.json`): model load 1.5 s, and the first encode in a fresh
process costs 19 ms vs 8 ms afterwards (one-time kernel/allocator setup). Without the startup hook the
first user query would pay ~1.5 s.
The *warm* p95 is sometimes *higher* than cold p95: 500 samples see more tail events (GC,
OS scheduling) than 20 samples do. p95 of 20 samples is just the 19th value.

**1 shard is faster than 3 shards** (see DESIGN.md "counterintuitive finding"): per-shard compute at
this corpus size is ~1–3 ms, so splitting it saves little, while the 3-way fan-out adds the DFS round
trip and waits on the *slowest* of 3 (tail amplification, toy 10 step 2).

## 7. Pitfalls and interview questions

- **Averaging p95s** across runs is meaningless. Recompute from raw samples.
- **Benchmarking with a noisy neighbour**: our first dense run (p95 92 ms) overlapped the bge-small
  embedding job and was discarded. Always check the machine is idle.
- **Coordinated omission**: a closed-loop client (send next after previous returns) under-reports
  latency under overload. Our load test (`scripts/query_load.py`) is closed-loop. Fine for "zero failures",
  not for capacity planning.
- *Q: Why measure at the client?* Server-side timers miss queueing, serialization and network.

## 8. Exercises

1. (Concept) Why is warmup done on *train* queries, not dev?
2. (Predict) With 10 shards instead of 3, what happens to p50 and p95? Why?
3. (Do) Re-run `python -m src.eval --strategy dense --threads 1` and `--threads 16`. Explain.
4. (Do) Add p99 to `latency_summary`. Is 500 samples enough to trust it?

Solutions: [solutions/07_benchmarking.md](solutions/07_benchmarking.md)

## 9. Further reading

- Dean & Barroso, *The Tail at Scale* (CACM 2013): required reading for scatter-gather
- Gil Tene, *How NOT to Measure Latency* (talk): coordinated omission
- Brendan Gregg, *Systems Performance*, ch. 2 (methodology)
