# Live-modify drills

The debrief says *"Expect to modify your own code live."* These drills rehearse exactly that.
For each: read the goal, predict what breaks, implement it **without peeking**, run the check,
then compare with `docs/solutions/drills/drill_NN.md`. Target times are for the second attempt.

| # | Drill | Touches | Check | Target |
|---|---|---|---|---|
| 1 | Run with **4 shards** and re-verify the 1% rule | `cluster.py`, `bench.py` | `svc4-*` metrics == single node | 10 min |
| 2 | Add **z-score fusion** as a third `fusion` method | `hybrid.py`, `ablations.py` | new unit test + dev sweep row | 15 min |
| 3 | Put a **cross-encoder rerank** in the default service path and re-measure p95 | `router.py` defaults, `bench.py` | is p95 still ≤ 50 ms? | 15 min |
| 4 | Add a new agent role: a **critic** that removes near-duplicate passages from the final 10 | `orchestrator.py` | new test in `test_agents.py` | 20 min |
| 5 | Implement **BM25 compaction** (merge delta into base, drop tombstones) | `bm25.py` | `test_bm25.py` + new compaction test | 25 min |
| 6 | **WAL snapshots**: dump state every N writes, truncate the log, recover from snapshot + tail | `store.py`, `wal.py` | crash test still passes | 30 min |
| 7 | Change the timeout policy: **hedged requests** (re-send to the same shard after p95 if no answer) | `router.py::_fan` | toy 14-style test | 20 min |
| 8 | Replace the DFS round trip with a **router-side df cache** (TTL 5 s) | `router.py` | nDCG within 1%, p50 drops by ~4 ms | 20 min |
| 9 | **Confidence gating** for the single agent: skip the LLM when the top-1 hybrid score > τ | `single_agent.py` | fewer LLM calls on plain set, same recall | 15 min |
| 10 | Add a `GET /doc/{id}` endpoint to the router returning text + version | `router.py`, `store.py` | curl it after an upsert | 10 min |

## Ground rules (same as the debrief)
- Say your plan out loud first: "I'll change X in file Y; this affects Z; I'll verify with W."
- Run the narrowest test first (`pytest -q tests/test_hybrid.py -k zscore`), then the module's tests.
- If something unexpected breaks, explain *why* before fixing it.
