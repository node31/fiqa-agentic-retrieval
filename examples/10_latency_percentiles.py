"""Toy 10: latency percentiles, cold vs warm, and measuring honestly.

Concept: p50/p95, tail latency, warmup effects, why averages lie.
Real code: src/eval.py (cold/warm protocol), src/util.py::percentile   Doc: docs/topics/07_benchmarking_latency_memory.md
"""
import random
import statistics
import time

from _common import banner, note, step, try_this

from src.util import percentile

banner("Toy 10: p50 vs p95 vs mean on a skewed latency distribution")
rng = random.Random(1)
lat = [rng.gauss(12, 1.5) for _ in range(950)] + [rng.uniform(40, 120) for _ in range(50)]  # 5% slow tail
step(1, f"1000 requests: 95% ~12 ms, 5% between 40 and 120 ms (GC pause, cache miss, a slow shard...)")
print(f"     mean = {statistics.mean(lat):.1f} ms   p50 = {percentile(lat, 50):.1f} ms   "
      f"p95 = {percentile(lat, 95):.1f} ms   p99 = {percentile(lat, 99):.1f} ms")
note(f"The mean ({statistics.mean(lat):.1f} ms) describes NO real request: "
     f"{sum(x < statistics.mean(lat) for x in lat) / len(lat):.0%} of requests are faster, the slow 5% are 3-10x slower. p50 = typical user, and p95/p99 = the unlucky ones. "
     "SLOs are written on percentiles.")
step(2, "Tail amplification under scatter-gather: the router waits for the SLOWEST of N shards")
p95_1 = None
for n in (1, 3, 10, 30):
    fanout = [max(rng.choice(lat) for _ in range(n)) for _ in range(5000)]
    p95_1 = p95_1 or percentile(fanout, 95)
    print(f"     {n:2d} shards -> p50 {percentile(fanout, 50):6.1f} ms   p95 {percentile(fanout, 95):6.1f} ms")
p_slow = sum(x >= 40 for x in lat) / len(lat)
note(f"With 30 shards, P(at least one slow shard) = 1 - (1 - {p_slow:.2f})^30 = {1 - (1 - p_slow) ** 30:.0%}, so most "
     f"queries hit one: p50 ({percentile(fanout, 50):.0f} ms) is now above the single-shard p95 "
     f"({percentile(lat, 95):.0f}-{p95_1:.0f} ms: p95 sits right on the edge of the 5% tail, so resampling moves it a lot). "
     "That is why the router needs timeouts (toy 14).")
step(3, "Cold vs warm: the first call pays one-off costs")
cache = {}


def handler(x):
    if "model" not in cache:  # lazy init: the classic cold-start cost
        time.sleep(0.05)
        cache["model"] = True
    return x * 2


times = []
for i in range(25):
    t0 = time.perf_counter()
    handler(i)
    times.append((time.perf_counter() - t0) * 1e3)
print(f"     first call {times[0]:.1f} ms, then p50 {percentile(times[1:], 50) * 1e3:.1f} us")
note("Real cold costs: lazy model load, PyTorch kernel selection on first forward pass, page faults on "
     "mmapped index files, new TCP connections, CPU caches. Our eval preloads models at startup, so the "
     "measured cold/warm delta is small. results/bench.json shows it.")
try_this("Why is p95 of only 20 samples (our cold measurement) a noisy number? Compute it 100 times on "
         "random subsets of `lat` and look at the spread.",
         "Change step 2 to use a 50 ms timeout per shard. What happens to p95 (and to completeness)?",
         "Remove the warmup in src/eval.py and re-run one config. Does warm p50 change?")
