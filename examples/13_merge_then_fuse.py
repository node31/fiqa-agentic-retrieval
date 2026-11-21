"""Toy 13: merge-then-fuse vs fuse-then-merge. The order of operations matters.

Concept: score normalisation is a GLOBAL operation. Doing it per shard changes the answer.
Real code: src/service/router.py (phase 3 merge, phase 4 fuse)   Doc: docs/08_sharding_scatter_gather.md
"""
from _common import banner, note, step, try_this

from src.retrieval.hybrid import linear, merge_shard_lists

banner("Toy 13: where should hybrid fusion happen in a sharded system?")
s0 = {"bm25": [("a", 10.0), ("b", 9.0)], "dense": [("a", 0.50), ("b", 0.49)]}
s1 = {"bm25": [("c", 2.0), ("d", 1.0)], "dense": [("c", 0.90), ("d", 0.10)]}
step(1, f"shard0: {s0}\n    shard1: {s1}")
g_bm, g_dn = merge_shard_lists([s0["bm25"], s1["bm25"]], 10), merge_shard_lists([s0["dense"], s1["dense"]], 10)
right = linear(g_bm, g_dn, 0.5)
step(2, f"Merge first (global lists), then fuse: {[(d, round(s, 3)) for d, s in right]}")
wrong = merge_shard_lists([linear(s0["bm25"], s0["dense"], 0.5), linear(s1["bm25"], s1["dense"], 0.5)], 10)
step(3, f"Fuse on each shard, then merge:       {[(d, round(s, 3)) for d, s in wrong]}")
note("Per-shard min-max maps each shard's best doc to 1.0, so shard1's mediocre 'c' (BM25 = 2) ties shard0's "
     "excellent 'a' (BM25 = 10). Each shard's local winner gets inflated.")
note("Our router asks each shard for its top-100 BM25 AND top-100 dense lists, merges each globally, "
     "and only then fuses. That is why sharded == single-node in tests/test_store.py.")
try_this("Does RRF have the same problem? (hint: are per-shard RANKS equal to global ranks?)",
         "What is the network cost of shipping two 100-item lists per shard instead of one fused list?",
         "With 3 shards, could a doc in the global top-10 be missing from every shard's top-100? Prove it can't.")
