# Project FAQ

These are the questions I expect about this repo in the 60-minute debrief, with the answers I'd give. Every number
comes from `results/` or DESIGN.md, measured on the AMD EPYC 7763 box described there (4 torch threads per process).
If I re-run something during the debrief, I quote the new number. General interview questions on the same topics are
in [concepts_faq.md](concepts_faq.md).

## Walkthrough script

The order I'd demo things in. On Windows without `make`, I copy the command after the tab in the `Makefile`.

1. Start the cluster: `docker compose up -d` (router on :8000, three shard containers with `mem_limit: 2g`), or
   `python -m src.service.cluster --shards 3` for local processes. `GET /admin/status` shows each shard's `n_docs`
   (19,167 / 19,254 / 19,217 with an empty WAL), `apply_mode` and RSS, plus the router's `active_model`.
2. Run one query three ways:
   ```
   python src/search.py --query "what is short selling?" --top-k 10 --strategy hybrid
   python src/search.py --query "what is short selling?" --top-k 10 --strategy bm25
   python -c "import httpx; print(httpx.post('http://127.0.0.1:8000/search', json={'query': 'what is short selling?', 'k': 5}).json()['timings_ms'])"
   ```
   The last line shows where the router's time goes (stats, embed wait, scatter, merge, fuse).
3. Reproduce quality: `docker compose run --rm eval` (or `make eval` against a local cluster) runs the 500 dev
   queries through the router. nDCG@10 should come out at 0.4190.
4. Writes: `pytest -q tests/test_integration_service.py`. It starts three shard processes and a router on the
   24-doc fixture with the hash embedder and checks idempotency, freshness right after the ack, deletes, and a hard
   kill followed by WAL recovery and partial results. It takes seconds.
5. Shard failure: `docker compose stop shard1`, search again and point at `partial: true, missing_shards: [1]`, then
   `docker compose start shard1`. `python scripts/docker_trace.py --failover` does the same with network evidence.
6. Migration under load. Terminal 1: `python scripts/query_load.py --url http://127.0.0.1:8000 --duration 120`
   (`make load`). Terminal 2: `make migrate`, `make cutover`, `make rollback`. The load script prints `FAILED=0` and how
   many queries each model served. `python scripts/migration_demo.py` scripts the whole thing, writes included.
7. Agents: `pytest -q tests/test_agents.py` (fake LLM: caps, gate, partial failure).
   `python scripts/render_agent_table.py results/agent_eval.json` prints the Azure table, and I open one
   `per_query[...].trace` to show a planner split and a single-agent rewrite.
   `python scripts/agent_eval.py --backend fake --limit 10 --out artifacts\agent_fake.json` runs the eval loop for free.
8. If there's time, async writes: `$env:APPLY_MODE='async'; docker compose up -d`, then `/admin/status` shows
   `apply_mode: async` and the queue stats.

## Part A: retrieval quality

### Why hybrid linear α = 0.7 and not RRF?
On dev, standard RRF (k = 60, equal weights) scored 0.3733 nDCG@10, below dense alone at 0.3881, because it gives
BM25 (0.2642) an equal vote. Linear min-max fusion with α = 0.7 on the dense side reached 0.4190, 8% relative over
dense. I tuned α on dev only and ran the test split once at the end. ADR 0012.

### You tuned α on dev. How do you know it generalises?
The single test run is the honest number: 0.3968 nDCG@10 on 648 test queries vs 0.4190 on dev
(`results/runs/FINAL-test-hybrid-linear-minilm.json`). Some drop is expected after tuning. The curve is also flat at
the top: α 0.6 gives 0.4179, 0.7 gives 0.4190 and 0.8 gives 0.4064 (`results/ablation_fusion_minilm.json`).

### Why MiniLM as primary when bge-small is the better dense model?
bge-small is +0.010 nDCG dense-only (0.3981 vs 0.3881), but its single-node p50 is 30.6 ms vs 18.8 ms. My hybrid p95
through the router is 40.5 ms, so roughly 12 ms more of query encoding would put it over the 50 ms budget (an
estimate: I didn't benchmark bge-small hybrid through the 3-shard router). In hybrid the gap disappears anyway
(0.4155 vs 0.4190), because BM25 already supplies what bge-small adds. ADR 0002.

### Why didn't reranking help?
The ms-marco MiniLM cross-encoder lowered nDCG@10 to 0.4058 at k = 10, 0.3964 at k = 20 and 0.3771 at k = 50, vs
0.4190 without it, and k = 10 cost 338.6 ms p50. My hypotheses: domain shift from web snippets to long forum answers,
the 256-token truncation of long FiQA answers, and a first stage that leaves little to fix. A FiQA-tuned reranker
might help, but not inside 50 ms on this CPU.

### What does the cold vs warm delta tell you?
It's close to zero by design: 29.5 / 33.0 ms cold vs 28.6 / 40.5 ms warm (p50 / p95) for the chosen config. Models load
and run a warmup encode at startup, vectors are copied into RAM, and the router keeps HTTP connections alive.
`results/microbench.json` shows what warmup hides: 1.5 s of model load, and a first encode of 19 ms vs 8 ms after.
Warm p95 is higher than cold p95 because 500 samples catch more tail events than 20.

### What do the stratified metrics say?
Short queries (< 5 words) get Recall@10 0.385, but n = 13. Medium is 0.491 (n = 403), long 0.510 (n = 84). When the
gold passage is among the 10% longest docs, recall is 0.464 (n = 185) vs 0.508 for the rest (n = 315): truncation at
256 tokens and BM25 length normalisation both hurt long answers. The code is `stratified()` in `src/eval.py`.

### Walk me through a failure.
`results/failures.md` has three. 161 of 500 dev queries have Recall@10 = 0, and 58 of those have no gold doc in either
retriever's top-100, so no fusion or reranking can save them. In case 1 ("Personal finance in EFU and NAFA") BM25
ranks the gold first and dense never finds it; with α = 0.7 a lexical-only hit is capped at a fused 0.300, so it
lands at rank 19. Case 2 has the answer past the 256-token cut of a 1,477-word passage, and in case 3 the query is a
forum title that doesn't contain the question.

### Did you try to fix those?
For case 1, a query-adaptive α (0.5 when a query term has IDF > 9) gave 0.4206 vs 0.4190, and with threshold 8 it gave
0.4176. An effect of 0.4% that flips sign with the threshold is noise on 500 queries, so I didn't adopt it. For case 2,
150-word chunks raised the gold passages' cosine (0.464 to 0.548 and 0.436 to 0.527) but not enough to reach the top
10. Chunking needs a full dev eval before I'd adopt it.

## Part B: distributed service

### How do you guarantee sharded equals single node?
Three rules, built in from the start. Each shard returns its top-100 per component, so the global top-100 is in the
union. A DFS phase (`/stats`) sends every shard the global N, total length and df for BM25, and dense cosine is
already comparable. The router merges each list globally and only then fuses (ADR 0003). BM25, dense and hybrid all
match single node to 4 decimals, and `tests/test_store.py::test_sharded_router_equals_single_node` guards it.

### Was the DFS phase worth it?
For quality, no: shard-local IDF gives BM25 0.2653 vs 0.2642 and saves about 4 ms (6.8 vs 10.7 ms p50). Hash sharding
over 19K docs per shard gives every shard nearly the same df ratios. I kept DFS because the assignment asks for
single-node equivalence and because local IDF drifts on skewed shards (by topic or by time). A df cache at the router
would give both; I haven't built it. ADR 0004.

### Why is 3-shard latency worse than 1 shard?
One shard: 23.4 / 26.3 ms p50 / p95. Three shards: 28.6 / 40.5 ms. Search compute is tiny at 19K docs per shard: in
process, the parallel sharded search takes 3.7 ms p50 vs 7.9 ms single node (`results/latency_breakdown.json`). What
sharding adds is a second fan-out for DFS, three times the HTTP calls per phase on one host, and a wait for the slowest
of three, which shows up in p95. At this size sharding buys memory headroom, write throughput and failure isolation.

### A shard is slow or down. What happens?
Every search-path shard call has a 0.5 s timeout (`SHARD_TIMEOUT_S`). The default policy, `partial`, answers from the
shards that replied and sets `partial: true, missing_shards: [i]`, so the caller can tell. `FAILURE_POLICY=fail`
returns 503 with the missing shards instead. `test_crash_recovery_and_partial_results` hard-kills a shard and checks
both the flag and the recovery. ADR 0006.

### What exactly is your freshness guarantee?
It depends on `APPLY_MODE`, set per shard process. In sync mode (the default) the shard applies the write under its
lock before it acks, so the first search issued after the ack finds it;
`test_freshness_write_searchable_immediately_after_ack` checks that across three processes with no sleeps. In async
mode the ack comes after the WAL fsync with `"visible": false`, and the promise is "searchable within 1 s", kept by
back-pressure and proven by `tests/test_integration_async.py`. `POST /admin/flush` gives read-your-writes on demand.

### Why build an async mode at all?
To measure the trade-off. With 8 writers and 2 readers on the real corpus (`results/write_load.json`), async acked 94.2
writes/s vs 57.8 for sync, ack p50 78 vs 128 ms, with visible lag p95 466 ms and max 583 ms. Searches under that load
got slower (p50 184 vs 131 ms) because the shards were embedding 63% more writes. Sync stays the default because it's
the simpler promise. ADR 0013.

### Your first async version was slower than sync. Why?
v1 acked 47.1 writes/s and lag reached 1.3 s. I'd assumed batching makes embedding cheaper, but on this CPU eight
passages per call cost 31.5 ms each vs 25.6 ms one at a time (short passages get padded to the longest), and one
applier thread embedded serially while sync embeds in each request's own thread. v2 runs 8 parallel single-doc embed
workers and predicts each new write's wait before acking.

### A write fails halfway. What can a reader see?
Nothing half-applied. Readers take the shared side of a per-shard RW lock, and a write updates the doc store, BM25 and
every dense index under the exclusive side. A crash before the WAL append means no ack, so the client retries. A crash
after it is replayed on restart (at-least-once plus idempotent). An exception inside apply marks the shard unhealthy
(503, reported as missing) and it restarts from the base corpus plus the WAL.

### Tell me about the WAL bug.
A crash mid-append leaves a torn last line. My first version skipped it on replay, but the reopened file kept the
fragment, so the next acked write was appended onto it and became one unparseable line on the following restart. A
write acked after recovery was lost. The fix truncates back to the last newline on open (`WAL._truncate_torn_tail`),
and `test_wal_write_after_torn_tail_survives_next_restart` runs crash, recover, write, restart.

### How do you make writes idempotent?
Each write carries `(doc_id, version)` from the client. An equal version is a replay and an older one is stale, so both
are no-ops, and deletes leave a tombstone so a late older upsert can't bring a doc back. That makes `Router.write` safe
to retry on timeout (two retries with backoff). In async mode the check runs against accepted versions, or a queued v2
could be overwritten by a v1 (`test_async_versions_compare_with_accepted_not_applied`).

### How does the migration avoid failed queries?
`make migrate` creates the bge-small index on every shard and turns on dual-write in the same lock acquisition as the
doc snapshot, then backfills in 256-doc batches and skips docs whose version changed. `make cutover` flips the router's
persisted `active_model` only if every shard reports ready, and each request reads the flag once. Rollback is instant
because dual-write kept the old index current. In `results/migration_demo.json` 2,045 queries ran with 0 failures, and
all 116 docs written during the run were found by both models.

### When would you roll back?
On a drop in a quality canary, a latency SLO breach (bge-small costs about 12 ms more per query here) or errors on the
new path. Rollback is instant until `make finalize` drops the old index, so I'd keep it through a full traffic cycle.
The canary itself isn't built; I'd run a fixed dev query set on both models before cutover.

### How do you know the containers really talk to each other?
`scripts/docker_trace.py` checks container IPs and DNS, sends writes and uses `docker exec` to confirm each landed on
the shard `crc32(doc_id) % 3` predicts, and reads each shard's TCP table to show the router's connections.
`scripts/docker_check.py --mode both` runs the functional checks in each write mode. In Docker the chosen config
reproduced nDCG@10 0.4190 at 30.4 / 34.1 ms p50 / p95.

## Part C: agents

### Why these roles?
The orchestrator is plain code because budgets, timeouts and slot allocation have to be deterministic and testable.
The planner is an LLM because decomposition needs language understanding. Workers have no LLM: the planner already
wrote standalone queries, so each worker is a parallel search with a 10 s timeout. The verifier is one batched LLM
call that catches a drifted sub-query and may rewrite it once. ADR 0009.

### How do sub-agents share the 10 slots?
Round-robin across the sub-query lists (A1, B1, A2, B2, ...) with dedupe, then fill from the full-question list
(`interleave()` in `src/agent/orchestrator.py`, ADR 0010). Scores from different queries aren't comparable, so a score
merge would let the easy half take all 10 slots and Coverage@10 would drop. A known weakness: when a worker fails, the
surviving list fills every slot. Reserving about 3 slots for the full-question list would protect the failed half.

### Show me partial failure.
`tests/test_agents.py::test_orchestrator_partial_failure_still_returns_results`: one worker raises and one sleeps 2 s
against a 0.5 s timeout, and the answer still has 10 results with both errors listed. A planner error degrades to
"simple", a verifier error to "all relevant", and if every worker fails the orchestrator runs one direct search.

### What are the guardrails?
One `Budget` per user query, shared by every LLM role: 3 calls and 8K tokens in the orchestrator, 3 calls and 12K
tokens in the single agent. `Budget.reserve()` refuses a call before spending if the prompt estimate plus max output
could exceed the cap. The single agent's 3-step cap is the same budget: at most 3 LLM calls and 3 searches (the free
step-0 search plus two rewrites), and the last step must finish.

### How does the orchestrator avoid over-orchestrating?
`looks_simple()` skips the planner for free when a question has at most 12 words and no conjunction, comma or second
question mark, and the planner can still answer "simple". In the Azure run the gate kept 56 of 100 plain queries, the
planner kept 37 and split 7. Those 7 are the whole plain loss (recall 0.824 to 0.514 on them); the other 93 match the
baseline exactly.

### Ship or not?
From the Azure gpt-6-sol run (`results/agent_eval.json`), plain questions stay on the non-agentic path: R@10 0.502,
about 30 ms through the router, no LLM. The single agent becomes an opt-in "deep search" (R@10 0.547, MRR 0.642, but
4.09 s p50 and 2,118 tokens). For compound questions I use the single agent when quality matters (R@10 0.400, Cov@10
0.40, p95 9.28 s, 3,380 tokens) and multi-agent under a latency or cost budget (0.355, 0.32, p95 4.75 s, 925 tokens).
Both beat one search (0.204, 0.10) by a wide margin.

### Your dev run favoured multi-agent. Why did it flip?
The single agent's last step is an LLM rerank over 200-char snippets. claude-haiku-4.5 did that badly (plain MRR 0.485
to 0.432) and gpt-6-sol did it well (0.485 to 0.642, with 0 recall losses on 100 plain queries). The multi-agent LLM
never ranks, it only splits and checks, so the stronger model barely moved it (compound R@10 0.335 to 0.355). Which
architecture wins depends on the model, so the eval has to be re-run when the model changes.

### Is the single vs multi gap on compound real?
Per query, single minus multi is +0.045 R@10 with a bootstrap 95% CI of [+0.007, +0.085] (30 wins, 17 losses, sign test
p ≈ 0.08), and +0.08 Cov@10. It's real but small at n = 100. I'd say single is somewhat better and uses about 3.7 times
the tokens.

### What did it cost?
About 655K tokens: $0.79 at placeholder prices of $1 / $4 per 1M input / output tokens (`PRICE_PER_1M_INPUT` and
`PRICE_PER_1M_OUTPUT` in `src/config.py`), because I didn't know the real gpt-6-sol rates. The `--dry-run` estimate was
$0.47; it replays the fake policy, which makes fewer and shorter calls than the real model, so I treat it as a lower
bound. About 610K of the tokens were input, since every step re-sends search results.

### How did you build the compound set?
`data/build_compound.py` samples 200 dev queries with seed 42, embeds them with MiniLM and pairs each with a random
partner below cosine 0.25 (mean pair cosine 0.088). GitHub Copilot wrote the rewrites from the two query texts only,
never from gold passages. Gold is the union of both halves' qrels, each doc tagged "a" or "b" for Coverage@10. I
selected 10 rewrites for the hand-check in `data/compound_review_sheet.md`; that check is still open.

### Why your own JSON action protocol instead of native tool calling?
One code path works for all three backends (fake, Copilot CLI, Azure Responses API), and every step is visible in the
trace. `parse_json` tolerates code fences and chatter, an unknown strategy falls back to hybrid, and returned doc ids
are kept only if the agent actually saw them. ADR 0008.

### What will happen on the held-out compound test set?
`python data/build_compound.py --split test` writes `data/compound_test.jsonl` with template joins ("<a>? Also, <b>"),
because rewrites exist only for dev. `scripts/agent_eval.py` reads `data/compound_dev.jsonl` directly today, so
pointing it at the test file is a small live change (below). I haven't run it, and I won't tune on it.

## Design and self-assessment

### What are the weakest parts of the system?
There's no replication, so a dead shard's docs are missing until it restarts. The WAL is never snapshotted or
truncated and tombstones are never collected, so restart time grows with write history. Modulo sharding means going
from 3 to 4 shards moves about three quarters of the docs, and there's no resharding tool. The agent numbers rest on
n = 100 per set, one model, placeholder prices and the single-node router (same quality as the 3-shard service). The
planner still splits 7 of 100 plain questions it shouldn't.

### What's your counterintuitive finding?
Sharding made latency worse (above). A close second: textbook RRF scored below dense alone (0.3733 vs 0.3881) because
it gave a much weaker retriever an equal vote.

### Which approaches didn't pan out?
Cross-encoder reranking (worse at every depth and 7 times the latency budget at k = 10), plain RRF, upgrading the dense
model to bge-small (+0.010 dense, −0.003 in hybrid, +12 ms), and async v1. Shard-local IDF "worked" (same quality, 4 ms
faster), but I chose exact equivalence over it.

### Can another engineer read this in 10 minutes?
`src/` is 2,004 logical lines excluding the data loader (`python scripts/count_loc.py`); 60% of the 4,999 physical
lines are docstrings, comments and blank lines. I'd start at `Router.search` in `src/service/router.py` (phases 0 to 5, numbered in
comments), then `ShardStore._write` / `_apply` in `src/service/store.py`, then `Orchestrator.run` (steps 1 to 4).

### How did you use AI assistants?
GitHub Copilot wrote the compound rewrites (from the query texts only), and I used assistants while coding, as I would
at work. What I kept has a measured reason in DESIGN.md or an ADR. Where an idea didn't survive measurement, such as
plain RRF or batched async embedding, the numbers are in DESIGN.md §11 and ADR 0013.

### Scaling to 100M passages?
Flat vectors would be 100M × 384 × 4 B = 153.6 GB, so I'd move to IVF-PQ with 48-byte codes (4.8 GB) and rerank the
top-100 with full vectors from disk. BM25 would be about 6.5B postings, roughly 52 GB uncompressed, so compressed
postings scored with WAND or block-max instead of my O(N) score array. Re-embedding takes about 11 days on this CPU
(57,638 passages in 567 s), so a migration needs GPUs and days of dual-write. The first things to break are flat
float32 vectors per shard, then Python BM25, then router fan-out tail latency.

### What would you build next?
Replicas with hedged reads, a df cache at the router, WAL snapshots with BM25 segment compaction, chunked passages for
long answers, the planner fix plus reserved slots for the full question, and confidence-gated agent routing.

## Live changes I should be ready to make

I checked every file, function, flag and env var below against the code. After any change to `src/config.py` I
restart the processes, because config values are read once at import.

### Add a 4th shard
Locally: `python -m src.service.cluster --shards 4`. `Cluster` starts four `src.service.shard_server` processes with
`N_SHARDS=4`, each keeps the docs where `shard_of(doc_id, 4)` equals its id, and the router gets four `SHARD_URLS`.
State goes to a new `artifacts\cluster4` directory, which matters because WALs from the 3-shard layout hold docs routed
by `% 3`. In Docker: add a `shard3` service in `docker-compose.yml` (copy `shard2`, `SHARD_ID: "3"`,
`STATE_DIR: /app/artifacts/docker/shard3`), set `N_SHARDS: "4"` in the `x-shard` environment, append
`http://shard3:8101` to the router's `SHARD_URLS`, and clear `artifacts\docker` first.

To verify, `/admin/status` should show four `n_docs` summing to 57,638, and
`python -m src.eval --mode service --url http://127.0.0.1:8000 --strategy hybrid --fusion linear --alpha 0.7 --name svc4-hybrid`
should give nDCG@10 0.4190 exactly. For a fast unit check I change `n = 3` to `n = 4` in
`test_sharded_router_equals_single_node` (`tests/test_store.py`). Moving existing writes to a new layout isn't
supported; that needs a resharding job or consistent hashing.

### Change the fusion weight
Per request, `alpha` in the `/search` body or `--alpha` in `src/search.py`. The served default is `DEFAULT_ALPHA` in
`src/config.py`, which both `SearchReq` (router) and `src/search.py` read; `DEFAULT_FUSION` and `RRF_K` sit next to it.
Fusion itself is `linear()` / `fuse()` in `src/retrieval/hybrid.py`. One trap: `src/eval.py` defaults to
`--fusion rrf --alpha 0.5`, so I always pass both flags.

To verify, `python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.6 --name hybrid-linear-a06`
should give 0.4179, matching `results/ablation_fusion_minilm.json`. `python scripts/ablations.py --model minilm` reruns
the whole sweep.

### Turn the reranker on, or swap it
Per request, `rerank_k` in the `/search` body or `--rerank-k 10` in `src/search.py`. `Router.search` then fetches text
for the top `rerank_k` (phase 5) and calls `rerank()` from `src/retrieval/rerank.py` in a worker thread. To make it the
default, I change `rerank_k` in `SearchReq`. The model is `RERANKER_NAME` in `src/config.py`, loaded lazily by
`get_reranker()`, so the first reranked query pays the load.

To verify,
`python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.7 --rerank-k 10 --name rerank10` should
give 0.4058 at about 339 ms p50, which shows why it's off: it breaks the 50 ms budget.
`python scripts/ablations.py --model minilm --rerank --ks 10,20,50` reruns the depth sweep.

### Change the shard timeout
The router reads `SHARD_TIMEOUT_S` (default 0.5 s) through `src/config.py`. In Docker it's the router's
`SHARD_TIMEOUT_S` in `docker-compose.yml`. For local clusters, `Cluster` passes its own `timeout_s` (default 0.5) as
`SHARD_TIMEOUT_S`, which overrides the shell, and `src/service/cluster.py` has no CLI flag for it, so I'd change the
`timeout_s` default in `Cluster.__init__` or add a `--timeout` argument in `main()`. `scripts/bench.py` starts its
clusters with `timeout_s=2.0`. Writes have a separate `write_timeout` of 30 s in `Router.__init__`.

To verify, I set it very low (0.001 s) and search: every shard misses the deadline, so the router answers 503 with
all three in `missing_shards`. A value of 0 does not disable the timeout; `_fan` treats it as "use the default".

### Change the failure policy
`FAILURE_POLICY=fail` on the router (read in `create_app()` in `src/service/router.py`) turns a missing shard into 503.
`Cluster` copies the shell environment into its children, so `$env:FAILURE_POLICY='fail'` before
`python -m src.service.cluster` works; in Docker I add it to the router's `environment`. A new policy, for example
"fail only if more than one shard is missing", is a one-line edit of the
`if not ok or (missing and self.failure_policy == "fail")` check in `Router.search`.

To verify, stop one shard (`docker compose stop shard1`, or `Cluster.kill("shard1")` in a test) and search: `partial`
gives results with `missing_shards: [1]`, `fail` gives 503. `pytest -q tests/test_store.py -k partial` covers the
default.

### Switch the write mode
`APPLY_MODE=async` per shard process, read in `build_store()` in `src/service/shard_server.py`. Docker:
`$env:APPLY_MODE='async'; docker compose up -d` (compose passes `APPLY_MODE` through). Local:
`$env:APPLY_MODE='async'; python -m src.service.cluster --shards 3`. `ASYNC_MAX_LAG_S` (0.5), `ASYNC_BATCH_MAX` (32)
and `ASYNC_EMBED_WORKERS` (8) tune the `AsyncApplier` in `src/service/async_apply.py`.

To verify, `/admin/status` shows `apply_mode: async` with queue stats, `pytest -q tests/test_integration_async.py`
checks the 1 s bound, and
`python -m scripts.write_load --mode both --duration 30 --out artifacts\write_load_check.json` compares both modes
without overwriting `results/write_load.json`.

### Add an agent role
Say a reranker role after slot allocation. I'd add a `RERANKER` prompt starting with `ROLE: reranker` in
`src/agent/prompts.py`, a function shaped like `verify()` in `src/agent/orchestrator.py`, and a call after step 4 in
`Orchestrator.run`, inside the same try/except so an error falls back to the interleaved list. The budget already has
room: `max_llm_calls=3` covers planner, verifier and one more, though `max_tokens=8_000` may need raising. I'd add a
`reranker` branch to `heuristic_responder` in `src/agent/fake_policies.py` so tests and `--dry-run` work.

To verify, a new test next to `test_orchestrator_respects_call_cap` in `tests/test_agents.py`, then
`python scripts/agent_eval.py --backend fake --systems multi --limit 20 --out artifacts\multi_fake.json`, then the free
`--backend copilot` run for a real-LLM signal before spending on Azure. This is the "multi-agent retrieval plus one
rerank call" idea from DESIGN.md §13.

### Stop false planner splits
The 7 false splits on plain questions come from `plan()` and the `PLANNER` prompt. I'd require at least two
independent information needs in the prompt, and in `plan()` accept a split only if the sub-queries' top-10 lists
overlap less than 50% (that costs a search per sub-query before deciding). The gate threshold is `max_words=12` in
`looks_simple()`.

To verify: `pytest -q tests/test_agents.py`, then
`python scripts/agent_eval.py --backend azure --confirm-spend --systems multi --max-usd 1 --out artifacts\multi_planner_fix.json`
and compare `planner_decisions` on the plain set with the 7 splits in `results/agent_eval.json`. Calls whose prompt
didn't change hit `.llm_cache`, so only the new calls cost money.

### Run the agents on the held-out compound set
`python data/build_compound.py --split test` writes `data/compound_test.jsonl`. `scripts/agent_eval.py` hard-codes
`data/compound_dev.jsonl` where it loads the compound set, so I'd add a `--compound` path argument (or edit that line)
and pass `--sets compound --out results\agent_eval_compound_test.json`. By default the eval searches an in-process
single-node router (`--search local`); `--search service --url http://127.0.0.1:8000` sends the searches through the
running 3-shard cluster instead.

### Migrate to the other embedding model
`make migrate` (`scripts/migration.py migrate --model bge-small --wait`), then `make cutover` and `make rollback`;
`NEW_MODEL` / `OLD_MODEL` in the `Makefile` pick the models, and `--reencode` / `--throttle` on
`scripts/migration.py` force full re-embedding and slow the backfill. The model registry is `MODELS` in
`src/config.py`, so a third model is one more `ModelSpec` entry plus an embedding cache from
`python -m src.index.build_index --models <key>`. To verify, I keep `make load` running in another terminal and check
that it prints `FAILED=0`.
