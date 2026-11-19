# DESIGN: FiQA distributed + agentic retrieval

All numbers below come from files in `results/` and were produced on: **AMD EPYC 7763 (16 physical /
32 logical cores visible), 128 GB RAM, Windows 11, Python 3.13, CPU only, 4 torch threads per process.**
Quality metrics are on the **dev split (500 queries)** unless marked TEST. The test split was run **once**,
for the final configuration only (§2).

## 1. Chosen operating point

**Hybrid retrieval: BM25 + all-MiniLM-L6-v2, linear min-max fusion with α = 0.7 (dense weight),
candidate depth 100 per component, no reranker, served by a router over 3 shards.**

- **Quality:** best dev nDCG@10 of all configurations that meet the constraints (0.419; +8% rel. over dense-only, +59% over BM25).
- **Latency:** p95 40.5 ms through the router (constraint: ≤ 50 ms after warmup). bge-small would add ~12 ms of
  query encoding (estimated from the single-node p50 gap of 30.6 vs 18.8 ms) and break the budget for a gain that
  vanishes in hybrid (0.416 vs 0.419).
- **Memory:** 723 MB peak RSS per shard process (constraint: ≤ 2 GB; also enforced by `mem_limit: 2g` in compose).
- **CPU only**: both at index time and at serve time.

## 2. Benchmark table (`results/bench.json`, `results/ablation_*.json`)

| configuration | mode | nDCG@10 | R@10 | MRR@10 | warm p50 / p95 ms | cold p50 / p95 ms | peak RSS MB |
|---|---|---|---|---|---|---|---|
| BM25 | single node | 0.2642 | 0.3135 | 0.3222 | 1.2 / 2.0 | 1.2 / 1.9 | 511 (process) |
| dense all-MiniLM-L6-v2 | single node | 0.3881 | 0.4665 | 0.4532 | 18.8 / 20.8 | 19.1 / 21.2 | 806 |
| dense bge-small-en-v1.5 | single node | 0.3981 | 0.4681 | 0.4773 | 30.6 / 34.2 | 30.2 / 34.2 | 851 |
| hybrid RRF (α=0.5, k=60), MiniLM | single node | 0.3733 | 0.4420 | 0.4485 | 22.1 / 39.9 | 21.7 / 24.3 | 812 |
| **hybrid linear α=0.7, MiniLM** | single node | **0.4190** | **0.4914** | **0.4872** | 20.7 / 24.5 | 20.1 / 23.3 | 812 |
| hybrid linear α=0.7, bge-small | single node | 0.4155 | 0.4856 | 0.4953 | 35.3 / 40.3 | 36.3 / 42.0 | 854 |
| ablation: + cross-encoder rerank@10 | single node | 0.4058 | 0.4914 | 0.4688 | 338.6 / 366.8 | – | – |
| ablation: + cross-encoder rerank@20 | single node | 0.3964 | 0.4829 | 0.4634 | 621.0 / 634.6 | – | – |
| ablation: + cross-encoder rerank@50 | single node | 0.3771 | 0.4633 | 0.4444 | 1246.3 / 1366.6 | – | – |
| BM25 | **router + 3 shards** | 0.2642 | 0.3135 | 0.3222 | 10.7 / 11.5 | 10.9 / 12.1 | 722 / shard |
| BM25, shard-local IDF (no DFS) | router + 3 shards | 0.2653 | 0.3142 | 0.3226 | 6.8 / 7.4 | 7.1 / 8.1 | 722 / shard |
| dense MiniLM | router + 3 shards | 0.3881 | 0.4665 | 0.4532 | 22.2 / 27.0 | 22.8 / 24.3 | 723 / shard |
| **hybrid linear α=0.7 (CHOSEN)** | **router + 3 shards** | **0.4190** | **0.4914** | **0.4872** | **28.6 / 40.5** | 29.5 / 33.0 | **723 / shard** |
| hybrid linear α=0.7 | router + 1 shard | 0.4190 | 0.4914 | 0.4872 | 23.4 / 26.3 | 22.9 / 25.1 | 775 |
| **TEST split, 648 queries (run once)** | single node | **0.3968** | **0.4658** | **0.4788** | 21.5 / 24.9 | 21.2 / 25.9 | 813 |

**Test split:** the chosen config was run on test exactly once, after every choice was frozen on dev (`python -m src.eval --split test --final-test-run ...`, `results/runs/FINAL-test-hybrid-linear-minilm.json`). nDCG@10 drops from 0.419 (dev) to 0.397 (test). Some optimism is expected after tuning α on dev, and for reference the published BEIR FiQA *test* numbers are BM25 0.236 and MiniLM ~0.37. Stratified test recall shows the same pattern as dev: short queries 0.328 (n = 29), gold in the longest docs 0.443 (n = 246) vs 0.480.

Single-node rows run in-process (no HTTP). Service rows are measured by the client end to end through
the router's HTTP API. Fusion sweep (dev): RRF α ∈ {0.5…0.9} → 0.373…0.406; linear α ∈ {0.1…0.9} → 0.287…0.419,
peaking at 0.7 (`results/ablation_fusion_minilm.json`; bge-small has the same shape, peak 0.416 at 0.7).

**Sharded vs single node (B1 requirement ≤ 1% relative):** BM25, dense and hybrid are **identical to 4 decimals**
(0.0%). This held on the first try because the three correctness rules were built in from the start
(docs/08): top-D per shard, global BM25 statistics (DFS phase), merge-then-fuse.

**Stratified Recall@10 (chosen config):** short queries (< 5 words) 0.385 (n = 13), medium 0.491 (n = 403),
long (> 15) 0.510 (n = 84). Gold in the 10% longest docs: 0.464 (n = 185) vs the rest 0.508 (n = 315). Long answers are
harder: truncation at 256 tokens for dense, and length normalisation for BM25. Short queries give both retrievers little to work with (small n).

## 3. Cold vs warm latency

Cold (first 20 queries of a fresh process) ≈ warm (500 dev queries after 100 train-query warmup):
e.g. chosen config 29.5 / 33.0 ms cold vs 28.6 / 40.5 ms warm (p50 / p95). **The delta is near zero by design:**
models load and run one warmup encode at process start (router `startup` hook, shard preload), vectors are copied
into RAM at load time (no lazy mmap page faults on the hot path), and the HTTP client keeps connections alive.
What warmup hides is measured in `results/microbench.json`: model load 1.5 s, and the first encode in a fresh process takes 19 ms vs 8 ms after that. Warm p95 is *higher* than cold p95 for several configs because 500
samples catch more tail events (OS scheduling, GC, the slowest-of-3-shards effect) than 20 samples. p95 of 20
samples is just the 19th value. Without the startup warmup, the first query would pay ~1.5 s.

## 4. Architecture

```mermaid
flowchart LR
  subgraph clients
    CLI[src/search.py]:::c
    EV[src/eval.py]:::c
    AG[agents: single / orchestrator]:::c
  end
  CLI & EV & AG -->|HTTP /search, /upsert, /delete, /admin/*| R[Router :8000<br/>tokenize + embed query once<br/>DFS stats · merge · fuse · fetch<br/>active_model flag]
  R -->|/stats, /search, /docs| S0[Shard 0 :8101<br/>BM25 + dense[minilm(, bge-small)]<br/>doc store · versions · WAL]
  R --> S1[Shard 1 :8102]
  R --> S2[Shard 2 :8103]
  IDX[src/index/build_index.py<br/>offline embedding cache] -.->|artifacts/emb_*.npy| S0 & S1 & S2
  classDef c fill:#eef
```

**Query path:** router tokenises and embeds the query (in a thread, overlapped with phase 1) → phase 1 DFS:
`/stats` on all shards → sum → phase 2: `/search` with terms + global stats + vector + model id; each shard
returns BM25 top-100 and dense top-100 → router merges each list globally → linear fusion → top-k → optional
`/docs` fetch for text. Every shard call has a 0.5 s timeout. Missing shards → `partial: true, missing_shards`.

**Write path:** `/upsert {doc_id, version, text}` → router picks the owner `crc32(doc_id) % 3` → shard: version check
(no-op if not newer) → embed with every maintained model (outside the lock) → write lock → re-check → WAL append + fsync
→ apply to doc store, versions, BM25 delta segment and every dense index → unlock → ack. The router retries on
timeout (safe: idempotent).

## 5. Consistency and freshness

- **Idempotency:** `(doc_id, version)`; equal = replay = no-op; older = no-op; deletes leave tombstones
  so a late older upsert cannot resurrect a doc. Tested in-process and through the router.
- **Lexical/dense consistency:** a single RW lock per shard; readers never observe a half-applied write.
- **Freshness guarantee: a write is searchable by the first query issued after its ack** (synchronous apply).
  `tests/test_integration_service.py::test_freshness_write_searchable_immediately_after_ack` proves it across
  3 shard processes over HTTP for all three strategies, with no sleeps or retries.
- **What can go wrong and recovery:** crash before the WAL append = the write never happened (no ack; the client retries).
  Crash after the append = applied on restart by WAL replay (at-least-once + idempotent). A torn final WAL line is truncated when the WAL is reopened
  (it was never acked). An exception inside apply → the shard marks itself unhealthy (503 → the router reports it
  missing) and is restarted (crash-only recovery: base corpus + WAL replay; vectors are stored in the WAL, so
  replay needs no model). Tested by hard-killing a shard.
- **Not provided:** replication (a dead shard's docs are unavailable until restart), multi-document transactions,
  WAL truncation/snapshots, tombstone GC.

## 6. Migration and rollback (B3)

`make migrate` → every shard creates the bge-small index and **turns on dual-write in the same lock
acquisition** as it snapshots the doc list, then backfills in throttled 256-doc batches (reusing cached vectors
when the text hash matches; `--reencode` forces full re-embedding). The backfill skips docs whose version changed since the
snapshot. `make cutover` flips the router's persisted `active_model` only if **every** shard reports the index ready, after
warming the query encoder. A request reads the flag once, so in-flight requests finish on the model they started with.
`make rollback` flips back instantly (the old index was kept current by dual-write). `make finalize` drops the old index.

Demo (`results/migration_demo.json`): 4 query threads + 1 writer for 77 s across migrate → backfill (17 s) → cutover →
rollback: **2,045 queries, 0 failed**; 116 docs written during the run were found by both models; per-shard index
sizes are equal for both models. Roll back on: a quality canary drop, a latency SLO breach (bge-small costs +12 ms/query
on this CPU) or errors on the new path. Keep the old index until the new one survives a full traffic cycle.

## 7. Agent design and decision (Part C)

**Systems compared** (all use the same search tool: hybrid linear α = 0.7 through the router API, top-10):
- **baseline**: one non-agentic search of the full question.
- **single agent** (`src/agent/single_agent.py`): starts from the baseline results (step 0, free), then ≤ 3 LLM steps. Each
  step either issues one more `search(query, strategy)` or finishes with chosen doc ids. The final top-10 is the agent's
  picks, then an RRF fill over everything it searched. JSON action protocol, budget 3 calls / 12K tokens.
- **multi-agent** (`src/agent/orchestrator.py`): the orchestrator is **plain code, not an LLM**. It owns the budget, the
  timeouts and the slots. **Planner** (LLM): simple vs compound, and up to 3 standalone sub-queries. It is skipped by a free
  regex/length gate (`looks_simple`) for short single-clause questions. **Workers** (no LLM): one search per sub-query, plus the
  full question, in parallel threads with a 10 s timeout each. **Verifier** (LLM, one batched call): looks at the top-5
  snippets per sub-query and may rewrite a drifted sub-query once. **Slots:** round-robin interleave across sub-query lists
  (≈ 5 + 5 for two halves) with dedupe, then fill from the full-question list. **Guardrails:** one shared `Budget` per user
  query (3 calls, 8K tokens; reserve-before-spend). A planner/verifier error degrades to "simple" / "all relevant"; a
  failed worker's list is dropped; if all fail, the system falls back to a direct search. Tested with a fake LLM in `tests/test_agents.py`.

**Dev results.** Free backend: GitHub Copilot CLI, `claude-haiku-4.5`, 100 plain dev queries (seed 42) + the 100
compound queries (`results/agent_eval_dev_copilot.json`, table from `python scripts/render_agent_table.py <file>`):

| set | system | R@10 | MRR@10 | Cov@10 | LLM calls/q | tokens/q | errors |
|---|---|---|---|---|---|---|---|
| plain | baseline | 0.502 | 0.485 | - | 0 | 0 | 0 |
| plain | single | 0.494 | 0.432 | - | 1.67 | 1,789 | 0 |
| plain | multi | 0.476 | 0.472 | - | 0.53 | 138 | 0 |
| compound | baseline | 0.204 | 0.365 | 0.10 | 0 | 0 | 0 |
| compound | single | 0.327 | **0.513** | **0.29** | 2.24 | 2,681 | 0 |
| compound | multi | **0.335** | 0.493 | 0.26 | 2.00 | 940 | 0 |

Coverage@10 = the fraction of compound queries whose top-10 has ≥ 1 gold passage for **both** halves.
The scripted fake LLM (`results/agent_eval_dev_fake.json`) shows the same ordering (compound R@10: baseline 0.204 <
single 0.288 < multi 0.354). Latency/cost columns from the Copilot run are **not meaningful**: each CLI call spends
~30 s starting a process with its own ~26K-token harness prompt. Non-agentic latency is the Part B number (28.6 / 40.5 ms).
Real latency and $/query come from the Azure run below.

**What the traces say** (per-query traces are in the JSON):
- The planner decomposed **100 / 100** compound queries. On the plain set it correctly kept **91 / 100** simple
  (56 by the free gate, so no LLM call; 35 by the LLM). The multi-agent's whole loss on plain comes from the **9 false
  decompositions**: recall on those 9 fell from 0.696 (baseline) to 0.400. On the other 91, multi = baseline exactly.
- The verifier rewrote a sub-query in 36 / 100 compound queries (3 / 100 plain).
- The single agent lowers plain MRR (0.485 → 0.432): when it "finishes" with its own picks, it reorders a list the fused
  ranker had already ordered well. On compound queries the same freedom helps (MRR 0.513).
- Single vs multi on compound is close in quality. Multi uses **2.9× fewer tokens** (940 vs 2,681), because workers need
  no LLM and the verifier is one batched call. Its p95 is also bounded: the workers are parallel and the call count is fixed
  (≤ 2 sequential LLM calls), while the single agent's steps are sequential and vary from 0 to 3.

**Decision (routing policy).** Ship **one** agentic path: the multi-agent orchestrator, used as a router:
1. Free gate says simple → serve the **non-agentic** search (no LLM, 29 ms).
2. Otherwise one planner call. If it says simple → non-agentic results. If it decomposes → workers + verifier + interleave.

This keeps the plain-query quality of the baseline (91% of plain queries never leave it), captures the compound gain
(R@10 0.20 → 0.34, Coverage 0.10 → 0.26 on dev), and spends LLM tokens only where they help. **Don't ship** the single
agent as a general wrapper: on plain queries it costs ~1.7 calls for slightly worse quality, and on compound it matches
multi at ~3× the tokens. **Next fix:** the 9 false decompositions. Make the planner prompt require two
*independent information needs*, and require ≥ 2 sub-queries whose top-10 lists overlap < 50% before trusting a split.
**Known limitation:** when a worker fails, the surviving sub-query list fills every slot. Reserving ~3 slots for the
full-question list would protect the failed half's coverage.

**Final numbers on Azure AI Foundry `gpt-6-sol`** (Responses API; reasoning effort `low`; no temperature parameter,
because reasoning models fix sampling. Every response is cached on disk, so reruns are deterministic and free). Run by the
user: `python scripts/agent_eval.py --backend azure --dry-run`, then `--backend azure --confirm-spend --max-usd 2`.
The table is rendered with `python scripts/render_agent_table.py results/agent_eval.json`. **Pending: paste here, with total spend.**
Prices are placeholders (`PRICE_PER_1M_INPUT` / `PRICE_PER_1M_OUTPUT` in `src/config.py`, defaults $1 / $4 per 1M tokens) until
the deployment's actual rates are set; the dev run's estimate at those prices was $0.64 for all 600 system-query runs.

## 8. One counterintuitive finding: sharding made latency *worse*

**Expected:** splitting the corpus 3 ways would cut per-query latency (each shard scans a third). **Observed:**
1 shard p50/p95 23.4 / 26.3 ms vs 3 shards 28.6 / 40.5 ms, same quality. **Hypothesis (supported by the
per-strategy numbers):** per-shard compute at 19K docs is ~1–3 ms (BM25 single-node p50 is 1.2 ms for all 57K docs), so there is little to
parallelise. Sharding adds a second fan-out round trip (DFS), 3× the HTTP calls per phase on one Windows host, and the
router must wait for the *slowest* of three shards, which shows up mostly in p95 (tail amplification). Sharding here
buys capacity (memory, write throughput, failure isolation), not latency. A close second: **textbook RRF hurt**
(0.373 < dense 0.388) because it gives a much weaker retriever an equal vote.

## 9. Approaches that didn't pan out

1. **Cross-encoder reranking (ms-marco-MiniLM-L-6-v2).** *Expected* +2–5 nDCG on top of hybrid. *Observed*
   0.406 (k=10), 0.396 (k=20) and 0.377 (k=50) vs 0.419 without: monotonically worse with depth, at 339 ms p50 for k=10. *Why:* domain shift (MS MARCO web snippets
   vs long forum answers), 256-token truncation of long FiQA answers (the stratified metrics show long gold passages are already hard),
   and a strong first stage leaving little to fix. It is also ~30 ms per pair on our CPU, 7× the budget at k=10.
2. **Standard RRF for hybrid.** *Expected* the usual "free" hybrid gain. *Observed* −0.015 nDCG vs dense.
   *Why:* equal votes for a retriever that's 12 points worse. Weighted RRF (α=0.9 → 0.406) and especially linear fusion fix it.
3. **Upgrading the dense model to bge-small.** *Expected* the better MTEB model to win. *Observed* +0.010 dense but −0.003 in
   hybrid, at +12 ms. *Why:* BM25 already supplies the lexical signal bge-small adds over MiniLM; the extra latency isn't worth it.
4. **Skipping the DFS round trip (local IDF).** Quality was the same (0.2653 vs 0.2642) and it was 4 ms faster. We *kept* DFS for exact
   single-node equivalence, so this "worked", but it's a trade-off we chose against, not a free win.

## 10. Scaling to 100M passages (arithmetic in docs/15_scaling_to_100M.md)

- **Vectors:** 100M × 384 × 4 B = **153.6 GB** flat → would need ≥ 77 shards of 2 GB for vectors alone. **IVF-PQ** (48-byte codes)
  = **4.8 GB** + ids/centroids; re-rank the top-100 with full vectors fetched from disk.
- **Index type:** IVF-PQ (nlist ≈ 65K, nprobe tuned for recall ≥ 0.95) or HNSW on bigger-memory nodes (~180 GB incl. graph).
- **BM25:** ~6.5B postings ≈ 52 GB uncompressed → compressed postings (Lucene-style) ~10–17 GB, scored with WAND/block-max
  instead of our dense O(N) score array.
- **Text:** ~77 GB (measured mean 768 B/passage in FiQA) → object store / disk, fetch only the final top-k.
- **Embedding:** 57,638 passages took 567 s (MiniLM) on 32 CPU cores → 100M ≈ **11 days** on this box; ~3–6 h on one A100,
  < 1 h on 8 GPUs. A model migration means a full re-embed, so dual-write must last hours to days.
- **Cost (order of magnitude):** ~8–16 memory-optimised nodes incl. 2× replication ≈ $3–6K/month; a re-embed run on 8 GPUs ≈ $30–100.
- **Breaks first in our design:** (1) flat float32 vectors per shard (memory, then latency); (2) Python BM25 with an O(N) score array;
  (3) router fan-out tail latency with 50+ shards; (4) WAL replay without snapshots; (5) modulo sharding (resharding moves ~all docs).

## 11. Production concerns

- **Availability:** ≥ 2 replicas per shard behind the router; the router load-balances and *hedges* slow reads (Tail at Scale);
  a write goes to the primary and is replicated from the WAL. Today, one dead shard means partial results until restart.
- **Caching:** a router-side LRU on (query, strategy, model) for popular queries (invalidated on writes to returned docs); a query-embedding
  cache; a global-df cache (replaces the DFS round trip, ~4 ms).
- **Cost:** the query encoder is the largest single CPU cost (8 ms/query; faiss 1.4 ms/shard, BM25 1 ms). Batch concurrent queries into one forward pass, use ONNX/int8 quantisation, and
  route agentic paths only to queries that need them (§7).
- **Monitoring:** p50/p95/p99 per phase (embed, DFS, search, merge), `partial` rate per shard, WAL lag/size, index sizes per model (must
  match during migration), an offline nDCG canary on a fixed query set after every deploy/migration, LLM tokens/$ per query.
- **Security:** the agents' only tool is read-only search, which limits prompt-injection blast radius; secrets only in `.env`/a secret store.

---
**LLM / spend disclosure:** dev runs used a free backend (GitHub Copilot CLI, claude-haiku-4.5) and the scripted fake LLM.
Final agent numbers: Azure AI Foundry deployment `gpt-6-sol` (see §7 for the model, temperature/reasoning settings and total spend).
Compound-query rewrites were written in-session by GitHub Copilot from the two query texts only (no gold passages);
10 were selected for human hand-check (`data/compound_review_sheet.md`). Code volume (`python scripts/count_loc.py`): **1,686 logical code lines** in `src/` excluding the data loader (2,881 physical lines; 41% are docstrings/comments, by design: this repo doubles as a study guide). Scripts, tests, toys and docs are extra.
