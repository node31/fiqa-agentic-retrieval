# Drill 7 solution: Hedged requests

```python
async def one(i, p):
    first = asyncio.create_task(self.shards[i].call(op, p, timeout))
    done, _ = await asyncio.wait({first}, timeout=self.hedge_after_s)   # e.g. single-shard p95
    if done:
        return i, first.result()
    second = asyncio.create_task(self.shards[i].call(op, p, timeout))  # hedge (a replica in production)
    done, pending = await asyncio.wait({first, second}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    for t in pending: t.cancel()
    ...
```
Only safe for **idempotent reads** (search/stats/docs). Never hedge writes without idempotency. Without
replicas, a hedge to the *same* slow process rarely helps. The point is the pattern (Dean & Barroso).
