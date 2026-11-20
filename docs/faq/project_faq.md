# Project FAQ: likely questions, with answers grounded in our numbers

Practise out loud: say your answer, *then* read ours. Numbers are from `results/` on the machine in
DESIGN.md. If you re-run, update the numbers you quote.

## Part A: retrieval quality

**Q: Why is your chosen config hybrid linear α=0.7 and not RRF?**
On dev, standard RRF (0.373 nDCG@10) was *worse* than dense alone (0.388), because it gives BM25
(0.264) an equal vote. Linear min-max fusion with α=0.7 reached 0.419 (+8% over dense). α was tuned on dev only.
The test split was run once at the end to measure how optimistic that is (DESIGN.md §2).

**Q: Your dev numbers are tuned on dev. How do you know they generalise?**
The single test run (`make test-final`, DESIGN.md §2) is the honest estimate. Also, the fusion curve is
smooth around α 0.6–0.8 (0.418/0.419/0.406), so we're not on a knife edge.

**Q: Why MiniLM as primary when bge-small is better dense?**
bge-small: +0.010 nDCG dense, but +12 ms per query encode (30.6 vs 18.8 ms). Through the router the
hybrid p95 is 40.5 ms with MiniLM; bge-small would push it to ~52 ms (over budget). In hybrid the
quality gap disappears (0.416 vs 0.419). ADR 0002.

**Q: What does the cold vs warm delta tell you?**
It's ~0. We preload models and run a warmup encode at process start, and keep HTTP connections alive.
What warmup hides (`results/microbench.json`): model load 1.5 s and a first encode of 19 ms vs 8 ms warm. p95 of 20 cold samples is a noisy number anyway.

**Q: Why did reranking not help?**
The MS MARCO MiniLM cross-encoder *lowered* nDCG@10 (0.406 at k=10, 0.396 at k=20, vs 0.419 without) and
costs ~30 ms per pair on our CPU (335 ms at k=10). Hypotheses: domain shift (web search vs forum finance
answers), truncation at 256 tokens of long FiQA answers, and the first stage already being strong. A FiQA-tuned or
larger reranker might help, but not within the 50 ms CPU budget.

**Q: Walk me through one failure.** → `results/failures.md`. Know all three by heart.

## Part B: distributed service

**Q: How do you guarantee sharded == single-node?**
Three rules: (1) each shard returns its top-100 per component, so the global top-100 is in the union; (2) the DFS
phase gives BM25 global N/avgdl/df; dense scores are absolute; (3) merge globally, then fuse. Result: 0.0%
difference on all metrics for all strategies.

**Q: Was the DFS phase worth it?**
Quality-wise, no: local IDF gives 0.2653 vs 0.2642 (noise), and saves 4 ms. We keep it for *exact*
equivalence and robustness to skewed shards. With a df cache at the router (drill 8) we'd get both.

**Q: Why is 3-shard p95 (40.5 ms) worse than 1-shard (26.3 ms)?**
At 19K docs per shard, per-shard compute is ~1–3 ms. Sharding saves almost nothing but adds a second fan-out
(DFS), 3× HTTP calls per phase and waits for the slowest of 3 (tail amplification). Sharding is for
scale (memory, write throughput), not for latency at this size.

**Q: A shard is slow. What happens?**
0.5 s timeout per call. Default policy `partial`: results from responsive shards, plus `partial: true,
missing_shards: [i]`. Policy `fail` gives 503. Tested by hard-killing a shard (integration test).

**Q: What exactly is your freshness guarantee?**
Searchable by the first query issued after the ack: synchronous apply under the write lock. Proven
by an integration test over 3 processes with no sleeps. The cost: write latency ≈ one embedding per
maintained model.

**Q: A write fails halfway. What can a reader see?**
Nothing half-applied: readers hold the shared lock, and writers apply BM25 + dense + doc store under the exclusive lock.
Crash after the WAL append → restart replays it (idempotent). Torn WAL line → truncated on reopen (never acked). We first only *skipped* it on replay; an audit showed the next append got glued onto the fragment, so a write acked after recovery vanished on the following restart. Regression test: `test_wal_write_after_torn_tail_survives_next_restart`.
An exception inside apply → the shard marks itself unhealthy (503) and is restarted (crash-only).

**Q: How does your migration avoid failed queries?**
Both indexes are live during the flip, and a request reads `active_model` once. Dual-write starts
atomically with the snapshot. The backfill skips docs whose version changed. Cutover is refused unless every shard is
ready. Demo: 2,045 queries, 0 failures, 116 writes during migration all present in both indexes.

**Q: When would you roll back?** Quality canary drop, latency SLO breach, errors. It's instant until `finalize`.

## Part C: agents

**Q: Why these sub-agent roles?** ADR 0009: planner (decomposition needs language understanding), parallel
search workers (no LLM; the planner already wrote good queries), verifier (catches off-topic sub-results
with one batched call), orchestrator in code (budgets, timeouts and slots must be deterministic).

**Q: How do sub-agents share the 10 slots?** Round-robin interleave of the sub-query lists, then fill from the
full-question search. Scores across sub-queries aren't comparable, so a score merge starves the hard half (toy 25).

**Q: Show me partial failure.** `tests/test_agents.py::test_orchestrator_partial_failure_still_returns_results`
and `python examples/24_partial_failure.py`.

**Q: Ship or not?** See DESIGN.md §7 for the numbers and routing policy. Short version, from the Azure gpt-6-sol run:
plain questions stay on the non-agentic path (~30 ms, free); the single agent is an opt-in "deep search" (+0.045 R@10,
+0.157 MRR, but ~4 s). For compound questions, use the single agent for quality (R@10 0.400, Cov 0.40, p95 9.3 s), or the
multi-agent path under a latency/cost budget (R@10 0.355, Cov 0.32, p95 4.7 s, 925 vs 3,380 tokens). The haiku dev run had
favoured multi; the flip is itself a good debrief story: the eval decides, and it must be re-run when the model changes.

## System design

**Q: 100M passages?** DESIGN.md §10 and docs/15: vectors 153.6 GB flat → IVF-PQ ~4.8 GB; flat search
time is the first thing to break; BM25 postings ~52 GB uncompressed; embedding takes ~11 days on our CPU, so use GPUs.

**Q: What would you build next?** Replicas + hedged reads, router df cache, WAL snapshots, BM25 compaction,
fine-tuning the dense model on FiQA train, confidence-gated agent routing.
