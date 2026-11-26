# Design: distributed, agentic retrieval over FiQA

## 1. Problem statement

The problem underneath everything here: an LLM is text in, text out, trained on someone else's data, and I want it
to work on mine. There are three ways to do that:

| way | what it means | what it needs |
|---|---|---|
| distillation | ask a strong model many questions about my domain, keep its answers, and train my own (smaller) model on them | many paid calls, a model I can train, and terms of use that allow it |
| fine-tuning | take a model whose weights I can download and keep training it on my data | an open-weight model (section 6.12 explains why the weights decide this) |
| RAG, retrieval augmented generation | leave the model as it is; at question time, search my data and put the best passages into the prompt | a good search over my data, and a way to keep it fresh and safe |

This project builds RAG, in steps, because the search is useful on its own, works with any LLM, and sees my data
change without retraining anything. It also fine-tunes the small embedding model the search uses (section 6.12),
which is the second way applied to the retriever instead of the LLM.

I built a retrieval system over BEIR FiQA: 57,638 personal-finance forum answers, with 5,500 train, 500 dev and 648 test
questions, each labelled with the passages that answer it. The system has three layers:

- Part A: three retrieval strategies (BM25, dense, hybrid) with an evaluation harness, ablations and a failure analysis.
- Part B: a sharded service (one router, three shards) that accepts live writes and can switch its embedding model
  with no downtime.
- Part C: a single LLM agent and a multi-agent system that use the search service as a tool, and a measured decision
  on when each one is worth its cost.

I tuned everything on dev and ran the test split once, at the end, for the configuration I chose. After the tasks I
fine-tuned MiniLM on the FiQA train split; it beat the original on dev, became the chosen model, and the test split was
run once for it too (section 6.12).

## 2. Requirements

**Functional**

| area | requirement | what I built |
|---|---|---|
| A1 | BM25, two dense models, a hybrid; a CLI | own BM25, all-MiniLM-L6-v2 and bge-small-en-v1.5, hybrid linear fusion; `python src/search.py --query ... --top-k 10 --strategy hybrid` |
| A2 | nDCG@10, Recall@10, MRR, p50/p95, peak RAM, stratified recall, cold vs warm | `src/eval.py` writes `results/bench.json` |
| A3 | at least one real ablation | fusion method x weight sweep, cross-encoder rerank depth sweep, dense model, local vs global IDF |
| A4 | three analysed failures | `results/failures.md` |
| B1 | ≥ 3 shards, router, ≤ 1% quality gap, timeouts, 1 vs N shard latency | 3 shards, scatter-gather router, 0.0% gap, 0.5 s timeout with partial results |
| B2 | idempotent versioned upsert/delete, consistent indexes, a tested freshness guarantee | versions + tombstones + WAL; sync mode (searchable on the next query) and async mode (searchable within 1 s) |
| B3 | zero-downtime model migration with cutover and rollback | dual-write + background backfill, `make migrate`, `make cutover`, `make rollback`; 2,045 queries, 0 failed |
| C1 | 100 compound questions with half-tracked gold | `data/compound_dev.jsonl` + `data/build_compound.py` |
| C2 | single agent, one search tool, ≤ 3 steps | `src/agent/single_agent.py` |
| C3 | orchestrator + ≥ 2 sub-agent roles, shared slots, budgets, partial failure, skip when simple | `src/agent/orchestrator.py`: planner, workers, verifier |
| C4 | compare baseline, single, multi on plain and compound sets; routing policy; ship calls | `scripts/agent_eval.py`, `results/agent_eval.json` |

**Non-functional**

| constraint | target | my chosen configuration |
|---|---|---|
| hardware | CPU only, at index and serve time | CPU only |
| latency | p95 ≤ 50 ms through the router, after warmup | 40.5 ms |
| memory | ≤ 2 GB RAM per shard process | 723 MB peak (compose also sets `mem_limit: 2g`) |
| data hygiene | tune on dev, test once | test run once, recorded in `results/runs/FINAL-test-hybrid-linear-minilm.json` |
| reproducibility | `docker compose up` brings up the service and eval | Dockerfile + compose; quality numbers are identical in Docker (nDCG@10 0.4190) |
| code size | about 2,000 lines, excluding tests and data loaders | 2,004 logical lines in `src/` (`python scripts/count_loc.py`) |

## 3. Guesstimates

Simple numbers I used to size the system before building it, next to what I later measured.

| quantity | quick math | measured |
|---|---|---|
| vectors per model | 57.6K passages x 384 dims x 4 bytes ≈ 88 MB | `emb_minilm.npy` 84 MiB on disk |
| docs per shard | 57.6K / 3 ≈ 19.2K | 19,167 / 19,254 / 19,217 |
| RAM per shard | model weights + Python (~650 MB) + vectors (~30 MB) + BM25 (~20 MB) + text (~15 MB) | 723 MB peak, so the model dominates |
| query time | encode the query (~8 ms) + search a third of the corpus (a few ms) + two HTTP round trips | 28.6 ms p50 through 3 shards |
| write cost | one embedding per passage (~25 ms) per maintained model | sync mode: 128 ms ack p50 under 8 concurrent writers |
| agent cost | prompt = instructions + results so far; about 0.1K to 3.4K tokens per question | 130 (multi, plain) to 3,380 (single, compound) tokens |

## 4. Infrastructure architecture

Everything runs on one machine with `docker compose up -d`. One image (`fiqa-retrieval:latest`) runs every container.

```mermaid
flowchart LR
  U["host: CLI, eval, agent scripts"] -->|"HTTP :8000"| R["router container"]
  subgraph net["compose network"]
    R -->|"http://shard0:8101"| S0["shard0 container<br/>mem_limit 2g"]
    R -->|"http://shard1:8101"| S1["shard1 container<br/>mem_limit 2g"]
    R -->|"http://shard2:8101"| S2["shard2 container<br/>mem_limit 2g"]
  end
  V[("bind mounts:<br/>./data/raw and ./artifacts")]
  S0 --- V
  S1 --- V
  S2 --- V
  R --- V
  T["one-shot containers (profile tools):<br/>indexer, eval, tools"] --- V
```

| container | command | notes |
|---|---|---|
| `shard0`, `shard1`, `shard2` | `python -m src.service.shard_server` | `SHARD_ID` 0 to 2, `N_SHARDS=3`, port 8101 inside the network, `APPLY_MODE` sync (default) or async, state in `artifacts/docker/shardN`, healthcheck on `/health`, `mem_limit: 2g` |
| `router` | `python -m scripts.api_docs` (the router app from `src.service.router`, plus example requests on `/docs`) | published on port 8000, `SHARD_URLS` lists the three shards, `SHARD_TIMEOUT_S=0.5`, state in `artifacts/docker/router_state.json`, starts after all shards are healthy |
| `indexer` | download FiQA, then `python -m src.index.build_index` | run once; writes the embedding caches to `artifacts/` |
| `eval` | `python -m src.eval --mode service --url http://router:8000 ...` | dev eval through the router |
| `tools` | any command, e.g. `python src/search.py --url http://router:8000 ...` | has `ROUTER_URL` set and `.env` mounted read-only for the agent eval |

Each service runs 4 torch threads. The same layout also runs as plain local processes (`python -m src.service.cluster
--shards 3`: router on 8000, shards on 8101 to 8103), which I used for tests and benchmarks.

## 5. Component architecture

```mermaid
flowchart TB
  CLI["CLI<br/>src/search.py"]
  OPS["ops scripts<br/>migration, query_load, write_load,<br/>migration_demo, docker_check, docker_trace"]
  EVAL["eval harness<br/>src/eval.py, scripts/bench.py, scripts/ablations.py"]
  AE["agent eval<br/>scripts/agent_eval.py"]
  subgraph AG["agent layer (library, src/agent)"]
    SA["single agent"]
    OR["orchestrator<br/>planner, workers, verifier"]
    TOOL["search tool<br/>search(query, strategy)"]
    LLMI["LLM client<br/>Budget + disk cache"]
  end
  LLMB["LLM backends<br/>Azure gpt-6-sol, Copilot CLI, fake"]
  R["router<br/>src/service/router.py"]
  subgraph SH["shard x3"]
    SS["shard server<br/>shard_server.py"]
    ST["shard store<br/>store.py, async_apply.py"]
    WAL["WAL<br/>wal.py"]
  end
  LIB["retrieval library<br/>bm25, dense, hybrid, rerank, sharding"]
  IDX["indexer<br/>download_data.py, build_index.py"]
  ART[("artifacts/<br/>embedding caches, state")]
  CLI --> R
  OPS --> R
  EVAL --> R
  AE --> SA
  AE --> OR
  SA --> TOOL
  OR --> TOOL
  SA --> LLMI
  OR --> LLMI
  LLMI --> LLMB
  TOOL --> R
  R --> SS
  SS --> ST
  ST --> WAL
  R --> LIB
  ST --> LIB
  IDX --> ART
  ST --> ART
  WAL --> ART
  R --> ART
```

| component | code | runs as | responsibility |
|---|---|---|---|
| CLI | `src/search.py` | command | sends one `/search` request, prints ranked passages with scores |
| router | `src/service/router.py` | FastAPI service | embeds the query once, gathers global BM25 statistics, fans out, merges, fuses; routes writes to the owner shard; drives migration; keeps `active_model` |
| shard server + store | `src/service/shard_server.py`, `store.py` | FastAPI service, one per shard | owns a slice of the corpus: BM25 index, one dense index per model, doc store, versions; applies writes under a read/write lock |
| async applier | `src/service/async_apply.py` | thread inside a shard (async mode only) | queue, parallel embedding, back-pressure, background apply |
| WAL | `src/service/wal.py` | file per shard | durable log of writes, replayed on restart |
| retrieval library | `src/retrieval/bm25.py`, `dense.py`, `hybrid.py`, `rerank.py`, `src/index/sharding.py` | imported code | scoring, fusion, optional cross-encoder rerank, `crc32(doc_id) % N` ownership |
| indexer | `scripts/download_data.py`, `src/index/build_index.py` | one-shot job | downloads FiQA, embeds the corpus once per model into `artifacts/emb_*.npy` |
| eval harness | `src/eval.py`, `scripts/bench.py`, `scripts/ablations.py` | one-shot jobs | metrics, latency, memory, sweeps into `results/` |
| agent layer | `src/agent/` | library run by `scripts/agent_eval.py` | single agent, orchestrator with planner, workers and verifier, the search tool, prompts |
| LLM client and backends | `src/agent/llm.py` | library + external API | one `complete()` interface over fake, Copilot CLI and Azure; token budget; disk cache |
| ops scripts | `scripts/migration.py`, `query_load.py`, `write_load.py`, `migration_demo.py`, `docker_check.py`, `docker_trace.py` | commands | migration steps, load during migration, write load, Docker checks |

## 6. Deep dives

### 6.1 Retrieval (A1)

BM25 scores a passage by how many rare query words it contains, with diminishing returns for repeats and a penalty for
long passages. Dense retrieval embeds the query and every passage as 384-number vectors and ranks by cosine similarity,
so it matches meaning ("betting against a stock" finds "short selling"). Hybrid combines both lists.

```mermaid
flowchart LR
  Q["query"] --> T["tokenize"]
  Q --> E["encode with MiniLM"]
  T --> B["BM25 top-100"]
  E --> D["faiss exact top-100"]
  B --> F["min-max normalise each list<br/>score = 0.7 dense + 0.3 BM25"]
  D --> F
  F --> K["top-10"]
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| my own BM25 on scipy sparse matrices (CSC for search, CSR for deletes, a delta segment for new docs, a tombstone mask) | I needed add/delete for live writes and per-term statistics for global IDF across shards | `rank_bm25`, Pyserini/Lucene | `rank_bm25` has no updates or stats API; Lucene would hide the sharding and update logic the spec asks for |
| all-MiniLM-L6-v2 as the primary dense model | 0.388 nDCG at 18.8 ms p50; leaves room under 50 ms after HTTP | bge-small-en-v1.5 | +0.010 nDCG dense but +12 ms per query, and the gain disappears in hybrid (0.4155 vs 0.4190) |
| bge-small-en-v1.5 as the second model (migration target) | same 384 dimensions, a real "newer model" story; its query prefix lives in the model registry | all-mpnet-base-v2, bge-base | 768 dimensions and roughly 3x the encode cost on CPU |
| faiss `IndexIDMap2(IndexFlatIP)` | exact search is allowed at this size; inner product on unit vectors is cosine; `remove_ids` supports deletes | HNSW, IVF | approximate search would cost recall for no needed speed (search is ~1 to 3 ms per shard) |
| linear fusion, α = 0.7 on dense, min-max normalised, depth 100 | best dev nDCG of everything I tried (0.4190) | RRF (k = 60) | equal-weight RRF scored 0.3733, below dense alone, because BM25 is much weaker on FiQA |
| no reranker in the chosen point | the cross-encoder made quality worse and cost 339 ms | cross-encoder rerank | see 6.3 |

Results (dev, single node, in-process):

| strategy | nDCG@10 | R@10 | MRR@10 | p50 / p95 ms |
|---|---|---|---|---|
| BM25 | 0.2642 | 0.3135 | 0.3222 | 1.2 / 2.0 |
| dense MiniLM | 0.3881 | 0.4665 | 0.4532 | 18.8 / 20.8 |
| dense bge-small | 0.3981 | 0.4681 | 0.4773 | 30.6 / 34.2 |
| hybrid RRF, MiniLM | 0.3733 | 0.4420 | 0.4485 | 22.1 / 39.9 |
| hybrid linear α = 0.7, MiniLM (chosen for the tasks) | 0.4190 | 0.4914 | 0.4872 | 20.7 / 24.5 |
| hybrid linear α = 0.7, bge-small | 0.4155 | 0.4856 | 0.4953 | 35.3 / 40.3 |
| dense MiniLM fine-tuned on FiQA train (section 6.12) | 0.4280 | 0.5044 | 0.5017 | 22.5 / 28.8 |
| **hybrid linear α = 0.7, fine-tuned MiniLM (chosen now)** | **0.4469** | **0.5141** | **0.5212** | 26.1 / 41.8 |

### 6.2 Evaluation harness (A2)

`src/eval.py` runs a strategy over the 500 dev questions and records quality, latency and memory. `scripts/bench.py`
runs every configuration in its own process, so the "cold" numbers are real.

```mermaid
flowchart LR
  P["fresh process"] --> W1["first 20 dev queries<br/>timed as cold"]
  W1 --> W2["100 train queries<br/>warmup, not timed"]
  W2 --> M["500 dev queries<br/>timed as warm"]
  M --> R["bench.json: nDCG, Recall, MRR,<br/>p50/p95 cold and warm, peak RSS,<br/>stratified recall"]
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| my own metric code (`src/metrics.py`), unit-tested | I wanted to know exactly what each metric computes | pytrec_eval only | harder to read and debug; I compared against it instead |
| warmup on train queries | no dev query is seen before it is timed | warmup on dev | the first 100 dev queries would be measured warm and cached |
| nearest-rank p50/p95 | always a real measured value | interpolated percentiles | invents a value nobody measured |
| peak RSS from the OS | that is what the 2 GB limit means | tracemalloc | misses torch and faiss native memory |

Stratified Recall@10 for the chosen configuration (dev):

| slice | n | Recall@10 |
|---|---|---|
| short queries (< 5 words) | 13 | 0.385 |
| medium (5 to 15 words) | 403 | 0.491 |
| long (> 15 words) | 84 | 0.510 |
| a gold passage is in the 10% longest docs | 185 | 0.464 |
| no gold passage in the 10% longest docs | 315 | 0.508 |

Long gold passages are harder because the encoder reads only the first 256 tokens. The gap is 0.083 for dense MiniLM
alone (0.414 vs 0.497), about zero for BM25 (0.310 vs 0.315) and 0.044 for hybrid.

Cold vs warm: 29.5 / 33.0 ms cold vs 28.6 / 40.5 ms warm (p50 / p95) for the chosen configuration through 3 shards. The
delta is near zero because the router and shards load the models and run one warmup encode at startup, the vectors are
copied into RAM at load time, and the HTTP client keeps connections open. Startup is what the warmup hides: loading the
model takes 1.5 s and the first encode in a fresh process takes 19 ms instead of 8 ms. Warm p95 can be higher than cold
p95 because 500 samples catch more tail events than 20.

### 6.3 Ablations (A3)

I varied one thing at a time on dev and kept everything else at the chosen configuration.

```mermaid
flowchart LR
  A["dense model<br/>MiniLM or bge-small"] --> F["fusion<br/>RRF or linear, α sweep"]
  B["BM25 statistics<br/>global or shard-local"] --> F
  F --> R["rerank depth<br/>off, 10, 20, 50"]
  R --> T["top-10"]
```

Fusion method and weight (MiniLM, dev nDCG@10; α is the dense weight):

| α | 0.1 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| linear | 0.2871 | 0.3360 | 0.4022 | 0.4179 | **0.4190** | 0.4064 | 0.4005 |
| RRF | | | 0.3733 | 0.3865 | 0.3953 | 0.4038 | 0.4062 |

bge-small has the same shape (linear peaks at 0.4155 at α = 0.7; RRF tops out at 0.4082 at α = 0.9).

Cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2` on top of the chosen hybrid):

| rerank depth | nDCG@10 | R@10 | MRR@10 | p50 / p95 ms |
|---|---|---|---|---|
| off | 0.4190 | 0.4914 | 0.4872 | 20.7 / 24.5 |
| 10 | 0.4058 | 0.4914 | 0.4688 | 338.6 / 366.8 |
| 20 | 0.3964 | 0.4829 | 0.4634 | 621.0 / 634.6 |
| 50 | 0.3771 | 0.4633 | 0.4444 | 1246.3 / 1366.6 |

BM25 statistics through 3 shards: global (DFS phase) 0.2642 at 10.7 ms p50; shard-local 0.2653 at 6.8 ms p50.

What I concluded:
- Linear fusion beats RRF here because the two retrievers are far apart in quality (0.264 vs 0.388). RRF's equal vote
  lets the weaker one drag the ranking down; a dense-heavy weight fixes it.
- The reranker gets worse the deeper it goes. It was trained on short MS MARCO web snippets, FiQA answers are long
  forum posts it truncates at 256 tokens, and the first stage is already decent. At ~30 ms per pair on this CPU it would
  also break the latency budget.
- bge-small is better alone and not better in hybrid, because BM25 already supplies the lexical signal it adds.
- Shard-local IDF gives the same quality within noise and saves one round trip. I kept global statistics because they
  make the sharded system exactly equal to single node and stay correct when shards are skewed.

### 6.4 Failure analysis (A4)

161 of the 500 dev queries have Recall@10 = 0 for the chosen configuration. I analysed three of them in `results/failures.md`
(query text, top-5 with scores, gold rank and score, diagnosis, fix).

| case | query | what went wrong | fix I would try |
|---|---|---|---|
| 1 (qid 4746) | "Personal finance in EFU and NAFA" | gold is BM25 #1 but absent from dense; with α = 0.7 its fused score caps at 0.300 | a query-adaptive α when a query has a very rare term; on dev it moved nDCG only 0.4190 → 0.4206, so I didn't adopt it |
| 2 (qid 4029) | "Family suggests my first real estate. Advice?" | two gold passages are long; the encoder never reads the part that answers the question | index overlapping 150-word chunks and score a doc by its best chunk; cosine to the query rose 0.464 → 0.548 for one gold doc |
| 3 (qid 7490) | "Clarify on some Stocks Terminology" | the query is a post title; the real question (trading volume) was in the post body | send title + body as the query, or let an agent ask a clarifying question; report this class separately |

In 58 of the 161 failures neither retriever had any gold passage in its top 100, so fusion or reranking could not help.

### 6.5 Router: sharding and scatter-gather (B1)

The router owns the query path. Each shard holds one third of the corpus and answers for its slice only.

```mermaid
sequenceDiagram
  participant C as Client
  participant R as Router
  participant S as Shards 0 to 2
  C->>R: POST /search
  par
    R->>R: tokenize and embed the query once
  and
    R->>S: stats for the query terms (DFS phase)
    S-->>R: df, doc count, total length per shard
  end
  R->>S: search with terms, global stats, vector, model, depth 100
  S-->>R: BM25 top-100 and dense top-100 per shard, took_ms
  R->>R: merge each list globally, then fuse, then top-k
  R-->>C: results, partial, missing_shards, latency_ms, timings_ms
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| `crc32(doc_id) % 3` | stable across processes and languages | Python `hash()` | salted per process, so writes and reads would disagree on the owner |
| a DFS phase: the router sums df, doc count and length from all shards | BM25 then uses global IDF, so sharded scores equal single-node scores | shard-local statistics | 0.2653 vs 0.2642, same within noise but not exact, and wrong for skewed shards |
| shards return top-100 per component, router merges globally, then fuses | min-max and RRF depend on the whole list, so fusing per shard crowns each shard's local winner | fuse on each shard, merge fused lists | changes the ranking depending on how docs are spread |
| embed the query once at the router and send the vector | saves two extra encodes per query | each shard encodes | 3x the CPU for the most expensive step |
| 0.5 s timeout per shard call, policy `partial` by default | an answer from 2 of 3 shards is more useful than an error for search | policy `fail` (also implemented) | a single slow shard would fail every query |

How the caller knows an answer is partial: the response has `partial: true` and `missing_shards: [i]`. With
`FAILURE_POLICY=fail` the router answers HTTP 503 with the missing shards instead.

Results (dev, through the router):

| configuration | nDCG@10 | p50 / p95 ms | peak RSS |
|---|---|---|---|
| single node, in-process | 0.4190 | 20.7 / 24.5 | 812 MB |
| router + 1 shard | 0.4190 | 23.4 / 26.3 | 775 MB |
| router + 3 shards | 0.4190 | 28.6 / 40.5 | 723 MB per shard |

BM25, dense and hybrid all match single node to four decimals (0.0% difference). Every `/search` reply includes
`timings_ms`; in a real-corpus run the parts were roughly: stats round trip ~15 ms, query-encode overhang 0 to 15 ms,
slowest shard's own compute ~4 to 5 ms, network and queueing ~8 to 12 ms, merge and fuse under 1 ms.

The search itself did get faster with shards. `results/latency_breakdown.json` times each step in one process, with
no HTTP (500 dev questions, depth 100):

| step | one node, 57.6K docs | 3 shards in parallel, ~19.2K docs each |
|---|---|---|
| encode the query | 11.2 ms p50 | 11.2 ms (once, at the router) |
| BM25 | 1.3 ms | 0.7 ms (slowest shard) |
| faiss | 6.2 ms | 2.4 ms (slowest shard) |
| fuse (and, sharded, merge) | 0.2 ms | 0.6 ms |
| search total, without the encode | 7.9 ms p50 / 9.7 ms p95 | 3.7 ms p50 / 4.6 ms p95 |

So sharding halves the search work, but the work is only a few milliseconds, and the extra HTTP round trips cost
more than it saves (section 11).

### 6.6 Shard write path (B2)

Every write carries `(doc_id, version)`. A shard applies it only if the version is newer than what it has, so a replay
or a late older write is a no-op, and a delete leaves a tombstone so an older upsert can't bring the doc back. The write
goes to a write-ahead log before it is acknowledged. I built two modes, chosen per shard with `APPLY_MODE`.

```mermaid
sequenceDiagram
  participant C as Client
  participant R as Router
  participant S as Owner shard
  participant A as Async applier
  C->>R: POST /upsert doc_id, version, text
  R->>S: forward to crc32(doc_id) % 3, retry on timeout
  alt sync mode (default)
    S->>S: version check, embed with every model
    S->>S: write lock, WAL append + fsync, apply to BM25 and dense
    S-->>R: applied
  else async mode
    S->>S: validate, version check vs accepted writes, back-pressure
    S->>S: WAL append (text only) + fsync, enqueue
    S-->>R: applied, visible false
    A->>A: embed with 8 workers, apply under a short lock
  end
  R-->>C: ack
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| client-supplied versions, apply only if newer | makes retries and replays safe; the router can retry on timeout | last-writer-wins by arrival | a late older write would overwrite a newer one |
| tombstones for deletes | a delayed older upsert can't resurrect a deleted doc | remove the entry | forgets that the doc was deleted at version v |
| WAL with fsync before the ack; replay on restart | an acknowledged write survives a crash | in-memory only | a crash loses acked writes |
| one read/write lock per shard over docs, versions, BM25 and every dense index | readers see a write fully or not at all | separate locks per index | a reader could see a doc in BM25 but not in dense |
| embed outside the lock, re-check the version inside | embedding takes ~25 ms; inside the lock it would block every search | embed inside the lock | readers wait for every write |
| sync as default | simplest guarantee: searchable on the next query | async only | 1.6x write throughput but a visible lag of up to ~0.6 s |
| async with back-pressure | throughput for write-heavy feeds while keeping a bound | async without back-pressure | the queue grows and the lag has no bound under load |
| crash-only recovery: an exception during apply marks the shard unhealthy; restart replays base + WAL | one simple, tested recovery path | try to repair in place | many partial states to reason about |

What a reader sees if a write fails halfway: the old version. A crash before the WAL append means the write never
happened and was never acked. A crash after the append is applied on restart by WAL replay. A torn last WAL line is
truncated when the log reopens (it was never acked). An exception inside apply makes the shard answer 503, the router
reports it in `missing_shards`, and a restart recovers it.

Freshness guarantees and their tests:
- sync: searchable by the first query issued after the ack.
  `tests/test_integration_service.py::test_freshness_write_searchable_immediately_after_ack` runs 3 shard processes and
  the router over HTTP, for every strategy, with no sleeps or retries.
- async: searchable within 1 s of the ack. `tests/test_integration_async.py::test_acked_write_is_searchable_within_one_second`
  polls until the doc is visible and asserts the lag. `POST /admin/flush` waits until every queued write is applied.

Results under heavy writes (`results/write_load.json`: real corpus, 3 shards, 8 writers and 2 readers, 30 s):

| | sync | async |
|---|---|---|
| writes acked per second | 57.8 | 94.2 |
| ack p50 / p95 ms | 128 / 181 | 78 / 120 |
| search p50 under this load | 131 ms | 184 ms |

Async makes the write ack fast; it was never meant to make search faster. Search p50 rose under async because the
two runs did not carry the same load: in the same 30 s async accepted 2,835 writes against sync's 1,741, and its 8
workers embedded every one of them on the same CPU the searches use. A linear estimate from the idle 3-shard p50
(28.6 ms) puts the cost at 1.77 ms of search p50 per write per second for sync, (131 - 28.6) / 57.8, and 1.64 ms
for async, (183.5 - 28.6) / 94.2. At sync's write rate, async would then search at about 28.6 + 1.64 x 57.8 = 123 ms,
slightly faster than sync. That is an estimate from two points, not a measurement. The price of async is freshness:
the visible lag was 466 ms at p95 and 583 ms at worst, inside the 1 s bound; sync has none.

### 6.7 Model migration (B3)

I build the new model's index next to the old one, keep both up to date, switch reads with a flag, and keep the old one
until I'm sure.

```mermaid
stateDiagram-v2
  [*] --> ServingA
  ServingA --> Migrating: make migrate (create B, dual-write on, backfill)
  Migrating --> ReadyB: backfill done on every shard
  ReadyB --> ServingB: make cutover
  ServingB --> ServingA: make rollback
  ServingB --> Finalized: make finalize (drop A)
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| turn on dual-write in the same lock acquisition as the backfill snapshot | no write can fall between "snapshot taken" and "dual-write on" | start dual-write after the snapshot | writes in that gap would be missing from B |
| backfill in 256-doc batches with an optional throttle, skipping docs whose version changed | searches keep running between batches; never overwrite a newer vector | one big rebuild under the lock | blocks searches for the whole build |
| reuse cached vectors when the text hash matches (`--reencode` forces a full re-embed) | the cutover demo runs in seconds instead of ~18 minutes | always re-embed | the same work, much slower |
| router-level `active_model` flag, read once per request, persisted to `router_state.json` | in-flight requests finish on the model they started with; restarts keep the decision | per-shard flags | shards could disagree mid-switch |
| refuse cutover unless every shard reports the new index ready | never serve queries against a half-built index | trust the operator | one mistake means bad results |

Demo (`results/migration_demo.json`): 4 query threads and 1 writer through migrate, backfill, cutover and rollback:
**2,045 queries, 0 failed**. The 116 docs written during the run were found by both models, and per-shard index sizes
matched for both models. I would roll back on a quality canary drop, a latency SLO breach (bge-small costs about 12 ms
more per query here) or errors on the new path.

### 6.8 Compound question set (C1)

`data/build_compound.py` (seed 42) samples 200 dev questions, embeds them with MiniLM, and pairs each with a random
unpaired partner whose cosine similarity is below 0.25, falling back to the least similar partner when none qualifies.
The halves end up on different topics: mean pair cosine 0.088, maximum 0.346. Each pair was rewritten
into one natural question by an LLM from the two question texts only. Gold labels are the union of both halves' qrels,
each tagged with the half it belongs to. I hand-checked ten of the rewrites (`data/compound_review_sheet.md`): all ten
read as one natural question and kept both halves' intents, so none needed a fix.

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| a cosine threshold of 0.25 between halves | same-topic pairs would be answered by one search | random pairs | some pairs would be near-duplicates |
| rewrite from the question texts only | no words from gold passages leak into the query | let the rewriter see the passages | inflates recall |
| keep the half of every gold passage | needed for Coverage@10 | a flat gold list | can't tell if both halves were answered |

### 6.9 Single agent (C2)

```mermaid
flowchart LR
  Q["question"] --> S0["step 0: one hybrid search, free"]
  S0 --> L["LLM step 1 to 3:<br/>search again or finish"]
  L -->|"search(query, strategy)"| T["search tool"]
  T --> L
  L -->|"finish with doc ids"| F["final top-10:<br/>agent picks, then RRF fill"]
```

The agent starts from one hybrid search, then gets at most 3 LLM steps. Each step returns JSON: either another
`search(query, strategy)` or `finish` with the doc ids it picks. The final list is its picks, then an RRF fill over
everything it searched. Budget: 3 calls and 12K tokens. Results are passed as 200-character snippets.

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| a free step 0 | the first decision is grounded in real results | let the LLM write the first query | one more call for little gain |
| my own loop with a JSON action protocol (native function tools on Azure) | the task spec rules out agent frameworks; one loop works for every backend | LangGraph and similar | not allowed |
| 200-character snippets | keeps prompts small; tokens grow with every step | full passages | input tokens would explode |

### 6.10 Multi-agent system (C3)

```mermaid
flowchart TB
  Q["question"] --> G{"free gate:<br/>looks simple?"}
  G -->|"yes"| B["one hybrid search"]
  G -->|"no"| P["planner (LLM):<br/>simple, or up to 3 sub-queries"]
  P -->|"simple"| B
  P -->|"sub-queries"| W["workers (no LLM):<br/>one search per sub-query + full question,<br/>parallel, 10 s timeout each"]
  W --> V["verifier (LLM, one batched call):<br/>top-5 snippets per sub-query,<br/>may rewrite one sub-query once"]
  V --> I["interleave sub-query lists round-robin,<br/>dedupe, fill from the full question"]
  I --> K["final top-10"]
```

The orchestrator is plain code. It owns one `Budget` per user question (3 LLM calls, 8K tokens, reserved before each
call), the timeouts and the 10 slots.

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| planner role | splitting a question needs language understanding | regex splitting only | misses real compound questions |
| workers without an LLM | the planner already wrote good sub-queries; search is free | one LLM agent per worker | more calls and tokens for no gain |
| verifier role, one batched call | catches a sub-query that drifted off topic | no verifier | drifted sub-queries fill slots with junk |
| orchestrator in code | budgets, timeouts and slots must be deterministic and testable | an LLM orchestrator | unpredictable call counts and cost |
| round-robin slot interleave (about 5 + 5 for two halves) | scores from different sub-queries aren't comparable; interleaving guarantees both halves get slots | merge by score | one half can take all 10 slots |
| a free word and punctuation gate before the planner | most plain questions never reach the LLM | always call the planner | an LLM call for every question |

Partial failure: if a worker fails or times out, its list is dropped and the others fill the slots; a planner or
verifier error falls back to "simple" or "all relevant"; if every worker fails, the orchestrator runs one direct search.
`tests/test_agents.py::test_orchestrator_partial_failure_still_returns_results` shows it.

### 6.11 Agent evaluation and routing (C4)

I compared the non-agentic hybrid search (baseline), the single agent and the multi-agent system on 100 plain dev
questions (seed 42) and the 100 compound questions, with Azure AI Foundry `gpt-6-sol`, reasoning effort low.

```mermaid
flowchart LR
  Q["incoming question"] --> G{"free gate:<br/>simple?"}
  G -->|"yes"| H["hybrid search<br/>about 30 ms"]
  G -->|"no"| P{"planner:<br/>compound?"}
  P -->|"no"| H
  P -->|"yes, quality first"| S["single agent<br/>p95 9.3 s"]
  P -->|"yes, latency or cost budget"| M["multi-agent<br/>p95 4.7 s"]
  H -.->|"user opts in"| D["single agent as deep search"]
```

| set | system | R@10 | MRR@10 | Cov@10 | LLM calls / q | tokens / q | $ / q | p50 / p95 s |
|---|---|---|---|---|---|---|---|---|
| plain | baseline | 0.502 | 0.485 | - | 0 | 0 | 0 | 0.06 / 0.08 |
| plain | single | 0.547 | 0.642 | - | 1.96 | 2,118 | 0.0025 | 4.09 / 7.27 |
| plain | multi | 0.481 | 0.466 | - | 0.51 | 130 | 0.0002 | 0.03 / 3.59 |
| compound | baseline | 0.204 | 0.365 | 0.10 | 0 | 0 | 0 | 0.08 / 0.10 |
| compound | single | 0.400 | 0.609 | 0.40 | 2.70 | 3,380 | 0.0039 | 5.75 / 9.28 |
| compound | multi | 0.355 | 0.518 | 0.32 | 2.00 | 925 | 0.0013 | 3.40 / 4.75 |

Coverage@10 is the share of compound questions with at least one gold passage from each half in the top 10. Dollar
figures use placeholder prices of $1 / $4 per 1M input / output tokens, because I didn't have the real gpt-6-sol rates.

What the traces showed:
- The single agent's last step is effectively an LLM rerank of what it has seen. With gpt-6-sol that helped: on plain
  questions recall improved on 7 and got worse on 0, and the first relevant hit moved up on 30 (down on 3).
- With the free dev model (claude-haiku-4.5 through the Copilot CLI) the same step hurt (plain MRR 0.485 → 0.432), and
  single and multi tied on compound (R@10 0.327 vs 0.335). The best architecture changed with the model.
- The multi-agent's whole plain-set loss comes from 7 questions the planner split by mistake (their recall fell
  0.824 → 0.514); the other 93 equal the baseline exactly. Its LLM never ranks, so a stronger model barely helps it.
- Single minus multi on compound, per question: R@10 +0.045 (bootstrap 95% CI +0.007 to +0.085), Coverage +0.08 (CI
  +0.01 to +0.15). Real, but small at n = 100.

| path | ship? | why |
|---|---|---|
| non-agentic hybrid | yes, the default for plain questions | ~30 ms, free, and agents don't beat it by enough to justify 4 s |
| single agent | yes: for compound questions when quality matters, and as an opt-in "deep search" for plain ones | best recall and coverage (R@10 0.400, Cov 0.40), at p95 9.3 s and 3,380 tokens |
| multi-agent | yes: for compound questions under a latency or cost budget | R@10 0.355, Cov 0.32 at p95 4.7 s and about a quarter of the tokens |

### 6.12 Fine-tuned MiniLM (after the tasks)

Only a model whose weights I can download can be fine-tuned, so the first question is how open a model is. The MTEB
leaderboard scores openness on six yes/no checks (an idea it credits to the OSAI index): open weights (can I run and
fine-tune it?), an open license (may I use and ship the result?), open training code (can I reproduce it?), open
training data (can I audit what it learned?), a paper (can I read the method?) and a model card (do I know its
intended use?). Weights and license decide whether I can use a model; code and data whether I can reproduce and
audit it; paper and card whether I can understand it. Scored on 26 November 2026 from each vendor's own pages
(partial = 0.5):

| model | weights | license | code | data | paper | card | score |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (the base model) | yes | Apache-2.0 | no | no (in MTEB's metadata) | yes | yes | 4/6 |
| bge-small-en-v1.5 | yes | MIT | no | yes | yes | yes | 5/6 |
| OpenAI GPT-6.2 Astra | no (API only) | no | no | no | system card | yes | 2/6 |
| Anthropic Claude Fable 5.2 | no | no | no | no | system card | yes | 2/6 |
| Google Gemini 4 Carbon | no | no | no | no | safety report | yes | 2/6 |
| xAI Grok 4.7 | no | no | no | no | model card | yes | 2/6 |
| Meta Llama 4 Maverick | yes (gated) | partial: custom license with a use policy and a 700M-user clause | no | no | partial: model card only | yes | 3/6 |
| OpenAI gpt-oss-120b | yes | Apache-2.0 | no (inference code only) | no | yes (arXiv 2508.10925) | yes | 4/6 |

The four API flagships score 2: there is nothing to download, so the only ways to adapt them are a better prompt
(RAG) or distillation. Both embedding models are open-weight and openly licensed, which is what made this section
possible.

FiQA's 5,500 train questions have labelled answers that nothing used. I fine-tuned all-MiniLM-L6-v2 on them
(`scripts/finetune.py`, `make finetune`) and registered the result as a third model, `minilm-ft`. Training is the
offline path; putting it into service is the B3 migration (6.7) with no new serving code. The full walkthrough, with a
toy example of the loss, is [docs/topics/18_fine_tuning.md](docs/topics/18_fine_tuning.md).

```mermaid
flowchart LR
  T["5,000 train questions<br/>(500 more held out)"] --> N["hard negatives:<br/>hybrid ranks 10-30"]
  N --> L["in-batch negatives loss<br/>batch 64, lr 2e-5, CPU"]
  L --> P["epoch picked on the<br/>held-out train questions"]
  P --> E["embed the corpus"] --> M["migrate → cutover → finalize<br/>on the live cluster"]
```

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| hard negatives from hybrid ranks 10 to 30 | top-ranked non-gold hits are often unlabelled real answers | ranks 1 to 9, or random passages | false negatives, or negatives too easy to teach anything |
| never a dev or test gold passage as a negative; never a dev or test question | the evaluation must not see training signal | use all passages | leakage |
| pick the epoch on 500 held-out train questions | dev stays clean for the go/no-go | pick on dev | tunes on dev twice |
| a plain PyTorch loop with the loss written out | readable, and no new dependencies (`datasets`, `accelerate`) | `SentenceTransformer.fit` | needs those packages in v3+ |
| a hash of the weights in the embedding cache's metadata | a retrained model keeps its name, so name checks can't spot stale vectors | name + max_seq_length only | old vectors would load silently |

Training: 12,818 triples, 2.7 h on 32 CPU cores. Held-out nDCG@10 (dense) 0.4680 → 0.4964 after epoch 1 and 0.4936
after epoch 2 (loss still falling), so the epoch-1 model was kept. The α sweep stays peaked at 0.7 (0.4469; 0.4419 at
0.6, 0.4434 at 0.8; `results/ablation_fusion_minilm-ft.json`).

| dev (500) | MiniLM | fine-tuned | Δ (paired bootstrap 95% CI) |
|---|---|---|---|
| dense nDCG@10 | 0.3881 | 0.4280 | +0.040 (+0.025 to +0.055) |
| hybrid nDCG@10 | 0.4190 | 0.4469 | +0.028 (+0.016 to +0.040); 152 questions better, 79 worse |
| hybrid R@10, gold in the 10% longest docs | 0.464 | 0.496 | |
| hybrid nDCG@10 through the router + 3 shards, and in Docker | 0.4190 | 0.4469 | identical to single node |
| **test (648, run once) hybrid nDCG@10** | 0.3968 | **0.4358** | **+0.039 (+0.028 to +0.051)** |

Online switch on the Docker stack: migrate, cutover and finalize under 4 query threads; 426 queries, 0 failed
(`results/migration_minilm_ft_load.json`); the backfill took 6 s because it reused the offline vectors. The stack now
serves only `minilm-ft`, and the Docker tour passes against it. The agent results (6.11) were measured with plain
MiniLM and were not re-run.

### 6.13 Extensions, in five categories

What I built beyond the three parts, grouped the same way in the deck, the docs and the replays:

| category | what it covers | built | headline |
|---|---|---|---|
| Distributed | how the cluster behaves under writes | sync vs async apply (6.6) | async: 1.6x the writes per second, visible within 1 s |
| Agentic Security | who is calling, what they may do, and abuse of the LLM | signed tokens (AuthN), tenant ownership and filtering (AuthZ, read and write your own only), harmful-intent and prompt-injection guardrails, topic scope against off-topic use | 0 of 6,648 real FiQA questions refused; 23 of 25 held-out fraud questions caught; off-topic caught 70% (minilm-ft) |
| Agentic Privacy | personal data and accountability | PII redaction (emails, phones, cards with Luhn, IBANs with mod 97, SSNs, IPs), a hash-chained audit log | the LLM provider never sees the PII with redaction on; editing any audit line breaks the chain from there |
| Agentic Cost Optimization | what each answer costs | fine-tuning (6.12), budgets, call and token caps, the cascade, the semantic cache, trimmed prompts, LLM query rewrite for follow-ups, rate limiting | worst case $0.01425 a question; cache answers 65% of paraphrases |
| Agentic Extensions | agents that talk to agents, remember, and look outside the corpus | A2A 1.0 (agent card, tasks, a coordinator that delegates), memory with attribution, isolation, expiry, access, deletion and opt-out, web fallback | Coverage@10 0.11 to 0.34 through the coordinator; follow-up R@10 0.103 to 0.455 with memory |

Each one sits behind a toggle or a flag and is off by default; [docs/design/EXTENSIONS.md](docs/design/EXTENSIONS.md)
lists them all and [docs/topics/17_security.md](docs/topics/17_security.md) to
[22_agent_memory.md](docs/topics/22_agent_memory.md) explain them.

#### How a tenant-filtered search keeps tenants apart

Every document has an owner: base-corpus docs with an odd id belong to acme and even ids to globex, and a new doc
belongs to whoever wrote it first. With `tenant_filter` on, the router sends the caller's tenant to every shard, and
each shard filters inside both retrievers, before it picks its top k.

BM25 builds a row mask, one true or false per doc. It is used twice. First, the statistics (N, total length and each
term's df) are counted over the tenant's rows only, so each tenant gets its own IDF. In a toy corpus of 6 docs where
acme owns 1, 3 and 5, and "roth" appears in docs 1, 3 and 4:

| | N | df("roth") | IDF = ln(1 + (N - df + 0.5) / (df + 0.5)) |
|---|---|---|---|
| everyone | 6 | 3 | ln(2.0) = 0.69 |
| acme only | 3 | 2 | ln(1.6) = 0.47 |

Shared statistics would let globex change acme's scores just by writing documents, and let acme probe how common a
word is in globex's documents. Second, globex's rows are scored 0 before the top k, so they can never come back.

Dense search passes faiss an `IDSelectorBatch` with the tenant's internal ids, and faiss skips every other vector
during the scan, so acme gets the exact 10 nearest acme docs. Dropping globex's hits after ranking would return
fewer than 10, or none, whenever globex's docs were closer (the `authz` toggle alone shows exactly that). The router
then merges and fuses as usual.

#### Off-topic: what the LLM does without the check

With topic scope off, "give me a recipe for chocolate chip cookies" sent to the agent's `answer` skill returned three
unrelated finance passages, and the LLM ignored them and wrote a full recipe (half a cup of butter, 350°F, 10 to 12
minutes), paid for by the finance service. With `guard_scope=block` the best passage scores 0.403 on minilm-ft,
below the calibrated 0.4421, the router returns nothing, and the agent makes no LLM call
([demo_security_a2a_memory.md](docs/guides/demo_security_a2a_memory.md), D7).

#### Replays

Every recorded replay is in [replays/](replays/README.md) and opens from disk: the eight scenarios on each of the four
cluster types, the security replay (14 cases, each protection off and on), the A2A replay (two agents and an LLM,
one arrow per message) and the memory replay (16 steps with what the LLM read and answered).

## 7. API design

### Router (`src/service/router.py`, port 8000)

| endpoint | method | called by | request | response | errors |
|---|---|---|---|---|---|
| `/search` | POST | CLI, eval, agent tool, load scripts | `query`, `k` (10), `strategy` (`bm25`, `dense`, `hybrid`), `fusion`, `alpha`, `include_text`, `rerank_k` (0) | `results` [`doc_id`, `score`, `text` if requested], `partial`, `missing_shards`, `model`, `strategy`, `latency_ms`, `timings_ms` | 503 with `missing_shards` when shards fail under the `fail` policy; 409 for an unknown strategy |
| `/upsert` | POST | writers, ingestion scripts | `doc_id`, `version`, `text` | owner shard's reply: `applied`, `doc_id`, `version`, `shard` (async mode adds `visible: false`, `mode`), plus `attempts`; or `applied: false, reason: stale_or_duplicate, current_version` | 422 if `text` is empty; 503 if the owner shard fails after retries |
| `/delete` | POST | writers | `doc_id`, `version` | same shape as `/upsert` | 503 as above |
| `/health` | GET | compose healthcheck, cluster.py | none | `ok`, `active_model`, `n_shards` (router process only) | none |
| `/admin/status` | GET | ops, migration script | none | `active_model`, `previous_model`, per-shard health, `missing_shards` | none |
| `/admin/flush` | POST | tests, load scripts | `timeout` (10 s) | `drained`, `missing_shards` (instant in sync mode) | none |
| `/admin/migrate` | POST | `make migrate` | `model`, `reencode`, `throttle_s` | `target`, per-shard start status | 503 / 409 |
| `/admin/cutover` | POST | `make cutover` | `model` | `active_model`, `previous_model` | 409 if the model isn't ready on every shard |
| `/admin/rollback` | POST | `make rollback` | none | `active_model`, `previous_model` | 409 if there is nothing to roll back to |
| `/admin/finalize` | POST | `make finalize` | `model` | `dropped`, per-shard result, `missing_shards` | 409 if asked to drop the active model |
| `/admin/security` | GET, POST | admin | POST: the toggles to change, e.g. `{"tenant_filter": "on"}` | every toggle's value and whether it is in effect; POST adds what changed | 401 / 403 without an admin token, always; 400 for an unknown toggle or value |
| `/admin/audit`, `/admin/audit/verify` | GET | admin | `limit` (50) | the last audit events; whether the hash chain is intact | 401 / 403 as above |
| `/guardrails/check` | POST | agents, before their LLM sees a question | `text` | `allowed`, `findings` | 403 when a guardrail in block mode refuses it |

Shard readiness and progress per model are visible in `/admin/status` through each shard's health.

Every endpoint accepts `Authorization: Bearer <token>` ([scripts/mint_token.py](scripts/mint_token.py)). What the token is
needed for, and what each security toggle adds to the requests and responses above (401, 403 and 429, a `security`
block and a `tenant` per hit), is in [docs/topics/17_security.md](docs/topics/17_security.md). With every toggle off, the
default, the API is exactly as in this table.

### Shard (`src/service/shard_server.py`, port 8101 in Docker, 8101 to 8103 locally)

The router is the only caller. Besides `GET /health`, every call is `POST /{op}`:

| op | request | response |
|---|---|---|
| `stats` | `terms`, optional `tenant` | this shard's `n_docs`, `total_len`, `df` per term (that tenant's docs only, if given) |
| `search` | `terms`, `k`, optional global `stats`, `vector`, `model`, `tenant` | `bm25` and `dense` lists of [doc_id, score], `owners`, `shard`, `took_ms` |
| `docs` | `ids` | {doc_id: text} for the ids this shard owns |
| `upsert` / `delete` | `doc_id`, `version`, `text`, optional `tenant`, `enforce_tenant` | `applied` true or false with a reason (`stale_or_duplicate`, `forbidden_tenant`) |
| `flush` | `timeout` | `drained` |
| `build_model` | `model`, `reencode`, `throttle_s` | `status`, `n_docs` or `ready` |
| `drop_model` | `model` | `dropped` |
| `health` | none | `shard_id`, `healthy`, `n_docs`, per-model `ready` / `progress` / `n`, `counters`, `apply_mode`, async stats, RSS |

### Internal interfaces

| interface | where | contract |
|---|---|---|
| agent search tool | `src/agent/tools.py` | `search(query, strategy)` returns the top-10 hits with `doc_id`, `score` and a 200-character `text` snippet; an unknown strategy falls back to `hybrid` |
| LLM client | `src/agent/llm.py` | `complete(system, user, max_output_tokens, ...)` returns the text, input and output tokens, latency and whether it was cached; when a `Budget` is passed it reserves the worst case before calling; responses are cached on disk by prompt hash |

## 8. Database design

I didn't use a database. Each shard keeps its data in memory and persists only what it needs to recover.

| data | how it is stored today | where |
|---|---|---|
| passage text | Python dict doc_id → text | shard memory, loaded from `data/raw` |
| versions and tombstones | dict doc_id → (version, deleted) | shard memory, rebuilt from the WAL |
| document owners | dict doc_id → tenant (kept for tombstones), plus per-tenant BM25 masks and faiss selectors cached until the next write | shard memory, rebuilt from the corpus rule and the WAL |
| BM25 index | scipy CSC matrix (term columns, for search) and CSR (doc rows, for deletes) for the base corpus, a dict delta segment for new docs, an `alive` mask for deletes | shard memory |
| dense index | one faiss `IndexIDMap2(IndexFlatIP)` per model, float32 unit vectors | shard memory |
| embedding cache | `emb_<model>.npy` (57,638 x 384 float32) + `.meta.json` (model, max_seq_length, dim, doc id per row, text hash per row) | `artifacts/` |
| write log | `wal.jsonl`: one JSON line per write (`op`, `doc_id`, `version`, `text`, `tenant`, vectors per model in base64 in sync mode) | per-shard state dir |
| maintained models | `models.json`, e.g. ["bge-small", "minilm"] | per-shard state dir |
| router state | `router_state.json`: `active_model`, `previous_model` | `artifacts/` |
| security toggles, audit log | `security_state.json` (one value per toggle); `audit.jsonl` (one hash-chained line per request) | next to the router state |

If I moved this to a real database (not built), I'd keep the same logical model:

| table / collection | fields | notes |
|---|---|---|
| `passages` | doc_id (key), tenant, version, deleted, text, updated_at | the source of truth; version check (and, with authz, the tenant check) as one conditional update |
| `embeddings_<model>` | doc_id, version, vector(384) | one collection per model makes blue/green migration a matter of building a second collection |
| lexical index | inverted index on `passages.text` (search engine or full-text index) | needs global term statistics across partitions |
| `router_state` | active_model, previous_model | small replicated config |

The database's own log and replication would replace my WAL and crash recovery. Partitioning by `crc32(doc_id)` would
stay. I'd still want the database to expose per-term statistics and per-partition top-k lists, or the merge-then-fuse
and global-IDF guarantees would be lost.

## 9. Scaling to the next 100M passages

| layer | change | why the current approach won't scale |
|---|---|---|
| vector index | IVF-PQ (48-byte codes, nlist around 65K) or HNSW on large-memory nodes; re-score the top candidates with full vectors | flat float32 is 100M x 384 x 4 B = 153.6 GB, at least 77 shards of 2 GB for vectors alone; IVF-PQ is about 4.8 GB |
| lexical index | compressed postings and WAND / block-max scoring | ~6.5B postings ≈ 52 GB uncompressed, and my BM25 builds an O(N) score array per query |
| passage text | object store or disk, fetch only the final top-k | ~77 GB of text (768 bytes per passage on average) |
| embedding | GPUs for indexing | 57,638 passages took 567 s on 32 CPU cores, so 100M would take about 11 days |
| sharding | consistent hashing or range partitions, more shards | `% N` moves almost every doc when N changes |
| router | replicas, hedged requests, a cached global-df table | waiting for the slowest of 50+ shards dominates p95; the DFS round trip grows with N |
| writes and storage | WAL snapshots and truncation, tombstone cleanup, group commit | replay time grows without bound; tombstones are never removed |
| model migration | dual-write lasting hours or days, throttled re-embedding | a full re-embed of 100M is a long job |
| agents | route only compound questions to agents, cache LLM calls | per-question cost and latency scale with traffic |

## 10. Pitfalls in the current design

| where it fails | effect today |
|---|---|
| one router, no replicas | the router is a single point of failure; a dead shard means partial results until it restarts |
| every shard loads the embedding model | most of each shard's 723 MB is the model, not the data |
| WAL is never truncated, tombstones are never removed | restart time and memory grow with the write history |
| BM25 delta segment is never compacted | new docs are scored one posting at a time |
| client-supplied versions | correctness depends on clients sending increasing versions |
| async mode | a search can return the previous version for up to about 0.6 s after the ack; overload slows writers |
| modulo sharding | changing the shard count needs a full re-shard |
| dense encoder truncates at 256 tokens | long gold passages are found less often (0.464 vs 0.508 Recall@10) |
| planner splits some plain questions | those 7 lost recall (0.824 → 0.514) |
| a failed worker's half loses its slots | the surviving sub-query fills all 10 slots |
| agent quality depends on the model | the haiku and gpt-6-sol runs led to different routing decisions |
| shards trust the router | with security on, anything inside the Docker network can still call a shard directly ([17_security.md](docs/topics/17_security.md) section 7) |

## 11. Results

### Benchmark table (dev, 500 questions)

| configuration | mode | nDCG@10 | R@10 | MRR@10 | warm p50 / p95 ms | cold p50 / p95 ms | peak RSS MB |
|---|---|---|---|---|---|---|---|
| BM25 | single node | 0.2642 | 0.3135 | 0.3222 | 1.2 / 2.0 | 1.2 / 1.9 | 511 |
| dense all-MiniLM-L6-v2 | single node | 0.3881 | 0.4665 | 0.4532 | 18.8 / 20.8 | 19.1 / 21.2 | 806 |
| dense bge-small-en-v1.5 | single node | 0.3981 | 0.4681 | 0.4773 | 30.6 / 34.2 | 30.2 / 34.2 | 851 |
| hybrid RRF, MiniLM | single node | 0.3733 | 0.4420 | 0.4485 | 22.1 / 39.9 | 21.7 / 24.3 | 812 |
| hybrid linear α = 0.7, MiniLM | single node | 0.4190 | 0.4914 | 0.4872 | 20.7 / 24.5 | 20.1 / 23.3 | 812 |
| hybrid linear α = 0.7, bge-small | single node | 0.4155 | 0.4856 | 0.4953 | 35.3 / 40.3 | 36.3 / 42.0 | 854 |
| + cross-encoder rerank@10 | single node | 0.4058 | 0.4914 | 0.4688 | 338.6 / 366.8 | | |
| BM25 | router + 3 shards | 0.2642 | 0.3135 | 0.3222 | 10.7 / 11.5 | 10.9 / 12.1 | 722 per shard |
| BM25, shard-local IDF | router + 3 shards | 0.2653 | 0.3142 | 0.3226 | 6.8 / 7.4 | 7.1 / 8.1 | 722 per shard |
| dense MiniLM | router + 3 shards | 0.3881 | 0.4665 | 0.4532 | 22.2 / 27.0 | 22.8 / 24.3 | 723 per shard |
| hybrid linear α = 0.7 (chosen for the tasks) | router + 3 shards | 0.4190 | 0.4914 | 0.4872 | 28.6 / 40.5 | 29.5 / 33.0 | 723 per shard |
| hybrid linear α = 0.7 in Docker | router + 3 shards | 0.4190 | 0.4914 | 0.4872 | 30.4 / 34.1 | 30.5 / 34.2 | about 505 MiB per container |
| hybrid linear α = 0.7 | router + 1 shard | 0.4190 | 0.4914 | 0.4872 | 23.4 / 26.3 | 22.9 / 25.1 | 775 |
| dense, fine-tuned MiniLM | single node | 0.4280 | 0.5044 | 0.5017 | 22.5 / 28.8 ¹ | 23.1 / 26.4 | 885 |
| hybrid linear α = 0.7, fine-tuned MiniLM | single node | 0.4469 | 0.5141 | 0.5212 | 26.1 / 41.8 ¹ | 25.1 / 71.7 | 891 |
| **hybrid linear α = 0.7, fine-tuned MiniLM (chosen now)** | **router + 3 shards** | **0.4469** | **0.5141** | **0.5212** | **37.4 / 46.5** ¹ | 33.9 / 41.1 | **716 per shard** |
| hybrid linear α = 0.7, fine-tuned MiniLM in Docker | router + 3 shards | 0.4469 | 0.5141 | 0.5212 | 33.2 / 38.5 | 34.1 / 44.7 | about 610 MiB per container |

Single-node rows run in-process; service rows are measured by the client end to end through the router's HTTP API.
¹ The fine-tuned rows were measured on a busier day. Controls with plain MiniLM in the same session: 23.9 / 28.9 ms
single node (fine-tuned: 24.1 / 29.3 ms in the repeat run) and 35.6 / 49.6 ms through 3 shards. Only the weights differ,
so the encode cost is the same; compare latencies within a session, not across days.

### Test split (run once)

| configuration | nDCG@10 | R@10 | MRR@10 | p50 / p95 ms |
|---|---|---|---|---|
| hybrid linear α = 0.7, MiniLM, 648 test questions | 0.3968 | 0.4658 | 0.4788 | 21.5 / 24.9 |
| **hybrid linear α = 0.7, fine-tuned MiniLM, 648 test questions** | **0.4358** | **0.5013** | **0.5163** | 25.0 / 34.5 |

nDCG drops from 0.419 on dev to 0.397 on test, which is the expected optimism from tuning α on dev. The stratified
pattern is the same as on dev: gold in the longest docs 0.443 (n = 246) vs 0.480 for the rest. The fine-tuned model
was chosen on dev before its one test run; on test it gains +0.039 nDCG@10 (paired bootstrap 95% CI +0.028 to +0.051),
and the long-doc slice rises to 0.483 vs 0.513 for the rest.

### Other results

| what | result |
|---|---|
| sharded vs single node | identical to four decimals for BM25, dense and hybrid |
| live writes | sync 57.8 writes/s with lag 0; async 94.2 writes/s with lag p95 466 ms, max 583 ms |
| migration under load | 2,045 queries, 0 failed; 116 writes during the run found by both models |
| agents (gpt-6-sol) | compound R@10: baseline 0.204, single 0.400, multi 0.355 |
| LLM spend | about 655K tokens for the whole agent eval, $0.79 at placeholder prices; the dry-run estimate was $0.47 because it replays a simpler scripted policy |

### Counterintuitive finding: sharding made latency worse

I expected three shards to be faster, since each searches a third of the corpus. I measured 1 shard at 23.4 / 26.3 ms and
3 shards at 28.6 / 40.5 ms (p50 / p95), with the same quality. At 19K docs per shard the actual search takes 1 to 3 ms
(BM25 over all 57K docs is 1.2 ms), so there is little to parallelise. Sharding adds a second round trip (the DFS phase),
three HTTP calls per phase, and a wait for the slowest of three shards, which shows up mostly in p95. At this size
sharding buys capacity and failure isolation, not speed.

### Approaches that didn't pan out

| approach | expected | observed | why I think it failed |
|---|---|---|---|
| cross-encoder rerank | +2 to 5 nDCG | 0.406 (k = 10), 0.396 (k = 20), 0.377 (k = 50) vs 0.419; 339 ms p50 at k = 10 | trained on short web snippets; FiQA answers are long and get truncated; the first stage leaves little to fix |
| standard RRF | the usual hybrid gain | 0.373, below dense alone (0.388) | it gives a much weaker retriever an equal vote |
| bge-small as the dense model | the better benchmark model wins | +0.010 dense, -0.0035 in hybrid, +12 ms per query | BM25 already supplies what bge-small adds |
| first async write mode | faster than sync | 47 writes/s vs 58, lag up to 1.3 s | one applier thread embedded serially, and batching doesn't save time on CPU (25.6 ms per doc alone vs 31.5 ms in batches of 8); fixed with 8 parallel single-doc workers |

## 12. Questions and answers

| question | answer |
|---|---|
| Chosen operating point, and why | for the tasks: hybrid linear fusion, α = 0.7, BM25 + all-MiniLM-L6-v2, depth 100, no reranker, router + 3 shards: best dev nDCG (0.4190) that meets every constraint (p95 40.5 ms, 723 MB per shard, CPU only). Since then: the same with MiniLM fine-tuned on FiQA train, dev nDCG 0.4469, same latency and memory (section 6.12) |
| Benchmark table | section 11: 17 configurations plus two test runs (one per chosen model) |
| Cold vs warm delta | near zero (29.5 / 33.0 vs 28.6 / 40.5 ms) because models load and warm up at startup; the hidden costs are 1.5 s model load and a 19 ms first encode |
| Stratified metrics | short 0.385, medium 0.491, long 0.510; gold in the 10% longest docs 0.464 vs 0.508 |
| Ablation and its deltas | fusion sweep (linear peaks at α = 0.7; RRF 0.373 to 0.406); rerank depth (every depth worse, 339 to 1,246 ms); bge-small vs MiniLM; local vs global IDF |
| Three failures | qid 4746 (rare acronyms, fusion weight), 4029 (long passages truncated), 7490 (title-only query); details in `results/failures.md` |
| Does sharding match single node within 1%? | yes, 0.0% for every strategy, on the first run, because of global IDF, top-100 per shard and merge-then-fuse |
| What happens when a shard is slow or down? | 0.5 s timeout; default `partial` returns the other shards' results with `partial: true` and `missing_shards`; `fail` returns 503 |
| 1 vs N shard latency | 23.4 / 26.3 ms vs 28.6 / 40.5 ms (p50 / p95) |
| Idempotency | `(doc_id, version)`, apply only if newer; deletes leave tombstones; replays and older writes are no-ops |
| What can a reader see if a write fails halfway, and recovery | the old version, never a half-applied write (one lock over both indexes); WAL replay on restart, torn tail truncated, crash-only restart on apply errors |
| Freshness guarantee and its test | sync: searchable by the first query after the ack (`test_freshness_write_searchable_immediately_after_ack`); async: within 1 s (`test_acked_write_is_searchable_within_one_second`) |
| How migration works | dual-write + throttled backfill, cutover by a router flag only when every shard is ready, instant rollback, finalize drops the old index; `make migrate`, `make cutover`, `make rollback` |
| Failed queries during migration | 0 of 2,045 |
| When I would roll back | quality canary drop, latency SLO breach, or errors on the new path |
| Sub-agent roles and why | planner (splitting needs language understanding), workers without an LLM (search is free), verifier (catches drifted sub-queries); orchestrator in code for deterministic budgets |
| How sub-agents share 10 slots | round-robin interleave across sub-query lists with dedupe, then fill from the full question |
| Guardrails | one budget per question: 3 LLM calls and 8K tokens (multi), 3 calls and 12K tokens (single), reserved before each call |
| Partial failure | a failed worker's list is dropped, errors fall back to a direct search; covered by a test |
| Skipping decomposition | a free gate plus the planner kept 93 of 100 plain questions on one search |
| Agent quality, cost and latency | section 6.11 table |
| Routing policy | simple → hybrid search; compound → single agent for quality, multi-agent under a latency or cost budget; single agent as opt-in deep search |
| Ship / don't ship | ship all three paths, each for its own traffic (section 6.11) |
| LLM, temperature and spend | Azure AI Foundry `gpt-6-sol`, reasoning effort low (no temperature parameter on this reasoning model), about 655K tokens, $0.79 at placeholder prices |
| Counterintuitive finding | sharding made latency worse (section 11) |
| Approaches that didn't pan out | cross-encoder rerank, standard RRF, bge-small, first async version (section 11) |
| Scaling to 100M | section 9 |
| Production concerns | replicas with hedged reads; caches for query embeddings, global df and popular queries; batching and quantising the encoder; per-phase latency, partial rate, WAL size and an nDCG canary as monitoring |
| Test split | run once for the configuration chosen for the tasks: nDCG@10 0.3968, R@10 0.4658, MRR 0.4788; run once for the fine-tuned model: 0.4358, 0.5013, 0.5163 |
| Measuring machine | in the README, section "Measured on" |
| Code size | 2,004 logical lines in `src/` excluding the data loader |

## 13. Next steps

The eight I'd do first are below. The longer list, grouped by part of the system (parameters, fine-tuning, datasets, chunking, evaluation,
scaling, writes and agents), is in [docs/design/EXTENSIONS.md](docs/design/EXTENSIONS.md).

1. Replicas per shard with hedged reads, so one slow or dead shard doesn't hurt p95 or completeness.
2. A cached global-df table at the router to drop the DFS round trip (about 4 ms).
3. WAL snapshots with truncation, tombstone cleanup and BM25 compaction.
4. Chunked passage indexing with best-chunk scoring for long answers.
5. Stop false planner splits: require two independent information needs and sub-query results that overlap by less than half.
6. Multi-agent retrieval followed by one rerank call by the strong model, to get the single agent's ordering at the multi-agent's latency.
7. Reserve about 3 slots for the full question when a worker fails.
8. Batch or quantise the query encoder. (Fine-tuning the dense model on FiQA train, the other half of this item, is
   done: section 6.12.)
9. Filter abusive words in questions and in the LLM's answers (a word list first, then a toxicity classifier), so the
   service neither answers in kind nor stores them in memory.

Security and LLM guardrails are built, each behind its own runtime toggle and off by default:
[docs/topics/17_security.md](docs/topics/17_security.md) has the threat model, how each piece works, how it is checked, and
its own list of next steps.

LLM cost optimizations and a web fallback are built too ([docs/topics/19_llm_cost.md](docs/topics/19_llm_cost.md) and [docs/topics/20_web_search.md](docs/topics/20_web_search.md)):

- A model cascade (a one-call cheap tier, escalating to the single agent) matches the single agent's quality: plain
  R@10 0.546 vs 0.547 at $0.00240 vs $0.00251 a question. With one model in both tiers that is all it saves. With a
  tier-1 model at a tenth of the price, the shadow replay shows 18% savings on plain.
- A semantic cache at cosine 0.85 answers 65% of paraphrased questions, while 0.8% of dev questions hit a different
  question.
- The web fallback (toggle `web_fallback`) fires on 1% of dev questions and 70% of off-topic ones, and sends the
  question PII-free.

The agents also run as an A2A (Agent2Agent 1.0) service with conversation and user memory
([docs/topics/21_a2a.md](docs/topics/21_a2a.md) and [docs/topics/22_agent_memory.md](docs/topics/22_agent_memory.md)):

- A coordinator agent that splits compound questions and delegates each part over A2A raises Coverage@10 from 0.11
  to 0.34, with no LLM.
- Without memory a follow-up question keeps R@10 0.103 of the original's 0.491. Prepending the last turn recovers
  0.439, and an LLM rewrite 0.455.
- Memory is kept for identified users only, PII-redacted, isolated per user, expiring, and deletable.

## Appendix A. The formulas, and what each one means

**BM25** (`src/retrieval/bm25.py`, k1 = 1.2, b = 0.75). Rewards a word that appears often in this doc but rarely in
the corpus, with diminishing returns for repeats and a penalty for long docs.

    score(q, d) = Σ_t IDF(t) · tf(t,d) · (k1 + 1) / (tf(t,d) + k1 · (1 - b + b · |d| / avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

tf is how often t appears in d, |d| the doc's length, avgdl the average length, N the number of docs and df(t) how
many contain t. Sharded, N, the lengths and df are summed over every shard first (the DFS phase), so the score is the
one a single node would give; with a tenant filter they are counted over that tenant's docs only.

**Cosine similarity** (dense). How closely two embeddings point the same way, from -1 to 1; with unit-length vectors it
is the dot product, which is what faiss's inner-product index computes: cos(q, d) = q · d / (|q| |d|).

**Min-max normalisation and linear fusion** (`src/retrieval/hybrid.py`, α = 0.7). Puts both lists on 0 to 1, then
blends them, weighting dense more because it is stronger on FiQA.

    norm(s) = (s - min) / (max - min)          fused(d) = α · norm_dense(d) + (1 - α) · norm_bm25(d)

**Reciprocal rank fusion, RRF** (c = 60). Joins ranked lists by rank, not by score, so scales don't matter: each list
gives a doc 1 / (c + rank), and a doc high in several lists wins. RRF(d) = Σ_lists 1 / (c + rank_list(d)).

**Recall@k and Coverage@k** (`src/metrics.py`). Recall@k is the share of a question's gold passages found in the top
k. Coverage@k, for compound questions, is 1 only if the top k holds at least one gold passage for each half.

**DCG, IDCG and nDCG@10.** Rewards putting relevant passages near the top: each hit is discounted by its rank, and the
sum is divided by the best possible sum, so a perfect ranking scores 1.

    DCG@k  = Σ_{i=1..k} (2^rel_i - 1) / log2(i + 1)      IDCG@k = DCG@k of the ideal order      nDCG@k = DCG@k / IDCG@k

**MRR@10.** One over the rank of the first relevant passage (0 if none in the top 10), averaged over questions.

**Percentiles, p50 and p95.** The latency that 50% (95%) of requests stay under. p95 shows the slow tail that the
average hides, such as waiting for the slowest of three shards.

**In-batch negatives loss** (`scripts/finetune.py`, scale 20). Each question in a batch of 64 should score its own
answer higher than the other 127 passages in the batch (the others' answers and the hard negatives):
loss = -log( e^(20 · cos(q, p+)) / Σ_j e^(20 · cos(q, p_j)) ). The scale sharpens the softmax.

**Paired bootstrap confidence interval.** Resample the questions with replacement many times, recompute the nDCG
difference each time, and read the 2.5th and 97.5th percentiles: if the interval excludes 0, the gain is not luck.

**Cost per call** (`src/agent/llm.py`). usd = input_tokens / 1M · price_in + output_tokens / 1M · price_out
(placeholders: $1 and $4 per million). A budget reserves the worst case (the full max_output_tokens) before a call.

**Token bucket** (`src/security/ratelimit.py`, 120 a minute, bursts of 20). Each user has a bucket of up to 20 tokens
that refills at 2 a second; a request takes one, and an empty bucket means HTTP 429.

**Hash chain** (`src/security/audit.py`). Each audit line stores the previous line's hash and its own:
hash_i = sha256(line_i without its hash, including prev = hash_{i-1}). Editing, deleting or reordering any line
breaks every link after it, and `/admin/audit/verify` reports the first bad one.

**Luhn and mod 97** (`src/security/pii.py`). Checksums that tell a real card number or IBAN from a random digit
string: Luhn doubles every second digit from the right and needs the sum to be a multiple of 10; an IBAN, rearranged
and turned into digits, must leave remainder 1 when divided by 97. They keep order numbers from being redacted.

**Thresholds by percentile** (topic scope, web fallback). The scope threshold is the 1st percentile of the best dense
score over the 500 dev questions, so by construction about 1% of real questions fall below it. The semantic cache
answers a new question from a cached one when their cosine is at least 0.85.

## Appendix B. Harness, loop, agent, software factory

The words used about agents, and where each thing lives:

| term | what it is | where it lives |
|---|---|---|
| LLM | text in, text out; remembers nothing between calls | the cloud (here Azure AI Foundry, gpt-6-sol) |
| harness | the program that runs everything, including the security checks | a machine or a server (here `scripts/ask.py`, the A2A server) |
| loop | the while loop inside the harness: build the context, call the LLM, run the tool it asked for, repeat | inside the harness (`src/agent/single_agent.py`) |
| agent | the harness running the loop with one system prompt and one tool list | not a separate box: a configuration (`src/agent/prompts.py`) |
| software factory | many harnesses with different agent configs (planner, coder, tester, reviewer), chained with CI checks and a human sign-off | across machines and pipelines |

One pass of the loop: build the context (system prompt, history, tool specs) and send it to the LLM; read the reply;
if it is a tool request, check it (allowed? needs approval? sandboxed?), run the tool, add the result to the history
and go round again; if it is not, it is the final answer and the loop stops. Memory, context compaction and logs also
live in the harness. In this repository the single agent is exactly that loop with one tool (search) and a budget of
3 LLM calls; the multi-agent system runs three agent configs (planner, workers, verifier) inside one harness.

---
Reproducibility: quality metrics come from `results/bench.json`, `results/ablation_*.json`, `results/write_load.json`,
`results/migration_demo.json`, `results/agent_eval.json` and `results/compare_*.json`, and Docker reproduces the chosen configuration's quality
exactly. Agent results depend on the LLM; every response is cached on disk, so a rerun with the same cache is identical.

AI and LLM use: the agent results use Azure AI Foundry `gpt-6-sol` (spend above), development runs used
claude-haiku-4.5 through the GitHub Copilot CLI and a scripted fake LLM, the compound questions were rewritten by an LLM
from the two question texts, and I used an AI coding assistant while building the project.
