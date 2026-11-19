"""Toy 15: hash partitioning. Modulo vs consistent hashing when the shard count changes.

Concept: how documents are assigned to shards, and how much data moves on resharding.
Real code: src/index/sharding.py::shard_of     Doc: docs/08_sharding_scatter_gather.md
"""
import bisect
import zlib
from collections import Counter

from _common import banner, note, step, try_this

from src.index.sharding import shard_of

banner("Toy 15: modulo hashing vs consistent hashing")
ids = [str(i) for i in range(57_638)]
step(1, f"Balance with crc32(id) % 3: {sorted(Counter(shard_of(d, 3) for d in ids).items())}")
note("Near-perfect balance with no lookup table. Any process can compute the owner.")
moved = sum(shard_of(d, 3) != shard_of(d, 4) for d in ids)
step(2, f"Grow 3 -> 4 shards with modulo: {moved / len(ids):.0%} of docs change shard")
note("Theory: a doc stays only if h % 3 == h % 4, i.e. 3 of every 12 hash values, so (N-1)/N = 75% move "
     "(N = 4 new shards): almost a full re-index.")


def ring(n, vnodes=100):
    pts = sorted((zlib.crc32(f"shard{s}-v{v}".encode()), s) for s in range(n) for v in range(vnodes))
    keys = [p for p, _ in pts]
    return lambda d: pts[bisect.bisect(keys, zlib.crc32(d.encode())) % len(pts)][1]


r3, r4 = ring(3), ring(4)
moved_c = sum(r3(d) != r4(d) for d in ids)
bal4 = Counter(r4(d) for d in ids)
step(3, f"Consistent hashing (100 virtual nodes/shard): {moved_c / len(ids):.0%} move; "
        f"balance {sorted(bal4.items())}")
note(f"Only the docs the new shard steals move: {moved_c / len(ids):.0%} here, which is exactly shard 3's share "
     f"({bal4[3] / len(ids):.0%}); the ideal is 1/N = 25%. Balance is much worse than modulo: the biggest shard holds "
     f"{max(bal4.values()) / min(bal4.values()):.2f}x the smallest. Virtual nodes only average out the arc lengths, "
     "and with 100 per shard the imbalance is still tens of percent: real systems use more vnodes or explicit "
     "rebalancing.")
step(4, "PITFALL: Python's hash() is salted per process")
print(f"     hash('12345') in this process = {hash('12345')}. Run again and it differs (PYTHONHASHSEED).")
note("If the router used hash() and the shards used hash(), they'd disagree on ownership, and writes would land "
     "on one shard while reads looked on another. crc32 is stable everywhere.")
try_this("Implement drill 1 in docs/DRILLS.md: add a 4th shard. Which component must re-index?",
         "Reduce vnodes to 3. How unbalanced does the ring get?",
         "Why does our router's fetch phase (texts by id) depend on shard_of matching the indexer?")
