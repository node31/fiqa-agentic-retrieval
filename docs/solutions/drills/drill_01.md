# Drill 1 solution: Run with 4 shards

Nothing in the code assumes 3 shards: `N_SHARDS` / `Cluster(n)` drive everything, and
`shard_of(doc_id, n)` re-partitions the corpus. Add to `scripts/bench.py::SERVICE`:

```python
("svc4-hybrid-linear-minilm", 4, {}, ["--strategy", "hybrid", "--fusion", "linear", "--alpha", "0.7"]),
```

Run `python scripts/bench.py --only service`. Expect identical quality (global stats + merge-then-fuse
make partitioning irrelevant) and **higher** p95 (fan-out over 4 → tail amplification, docs/07).
Gotcha: existing WAL state dirs belong to the 3-shard layout (docs hashed with `% 3`). The bench uses
`fresh=True`; in production you'd have to *move* documents (re-sharding), which is why consistent
hashing exists (toy 15).
