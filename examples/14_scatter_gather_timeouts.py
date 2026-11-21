"""Toy 14: scatter-gather with asyncio, per-shard timeouts and partial results.

Concept: fan out concurrently, bound the wait, and TELL the caller what is missing.
Real code: src/service/router.py::Router._fan    Doc: docs/topics/08_sharding_scatter_gather.md
"""
import asyncio
import time

from _common import banner, note, step, try_this

banner("Toy 14: scatter-gather with a slow and a dead shard")


async def shard(i: int, delay: float, fail: bool = False):
    await asyncio.sleep(delay)
    if fail:
        raise ConnectionError(f"shard {i} refused connection")
    return [(f"s{i}-doc{j}", round(1 - 0.1 * j - 0.01 * i, 2)) for j in range(3)]


async def fan_out(timeout: float, policy: str):
    specs = [(0, 0.010), (1, 0.300), (2, 0.015, True)]  # normal, SLOW, DEAD

    async def one(spec):
        i = spec[0]
        try:
            return i, await asyncio.wait_for(shard(*spec), timeout)
        except Exception as e:  # noqa: BLE001
            return i, e

    t0 = time.perf_counter()
    res = await asyncio.gather(*(one(s) for s in specs))  # concurrent: total ~ max, not sum
    ok = {i: r for i, r in res if not isinstance(r, Exception)}
    missing = {i: type(r).__name__ for i, r in res if isinstance(r, Exception)}
    ms = (time.perf_counter() - t0) * 1e3
    if missing and policy == "fail":
        return f"HTTP 503 after {ms:.0f} ms, missing={missing}"
    merged = sorted((x for r in ok.values() for x in r), key=lambda x: -x[1])[:3]
    return f"{ms:.0f} ms, partial={bool(missing)}, missing={missing}, top3={merged}"


step(1, "Sequential calls would take 10 + 300 + 15 ms. asyncio.gather overlaps them.")
step(2, f"timeout=1.0s, policy=partial -> {asyncio.run(fan_out(1.0, 'partial'))}")
note("We waited for the slow shard (300 ms). Results are complete except for the dead one.")
step(3, f"timeout=0.05s, policy=partial -> {asyncio.run(fan_out(0.05, 'partial'))}")
note("The slow shard is cut off: latency is bounded by the timeout, and the response SAYS it is partial.")
step(4, f"timeout=0.05s, policy=fail    -> {asyncio.run(fan_out(0.05, 'fail'))}")
note("Search box: prefer partial (a slightly worse answer beats an error). Eval harness or compliance "
     "search: prefer fail (silently missing a third of the corpus is unacceptable). Our router supports "
     "both (FAILURE_POLICY) and defaults to partial + the `partial`/`missing_shards` fields.")
try_this("Add 'hedged requests': if a shard hasn't answered in p95 time, send a duplicate to a replica.",
         "What timeout would you pick given results/bench.json shard p99? Write down the trade-off.",
         "Make the dead shard fail SLOWLY (after 5 s) instead of fast. Why is that worse?")
