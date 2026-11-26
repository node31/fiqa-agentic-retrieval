# AGENTS.md

Guidance for coding agents (and people) working in this repository: what each part does, how a request flows
through it, the invariants that must not break, how to run and test things, and where to put a change. The
human-facing overview is [README.md](README.md); the design and the numbers are in [DESIGN.md](DESIGN.md); the
deep dives are the numbered topic docs in [docs/topics/](docs/topics/).

## 1. What this repo is

A distributed, agentic retrieval system over BEIR FiQA (57,638 personal-finance forum answers), built in three
parts plus extensions:

| part | what | code |
|---|---|---|
| A | BM25, dense (MiniLM, bge-small, a fine-tuned MiniLM) and hybrid retrieval, and the evaluation harness | `src/retrieval/`, `src/index/`, `src/eval.py`, `src/metrics.py` |
| B | a router and N shards: scatter-gather search, versioned writes with a WAL, sync or async apply, zero-downtime model migration | `src/service/` |
| C | a single agent and a multi-agent orchestrator that use the search service as their one tool, with budgets and cost accounting | `src/agent/` |
| extensions | security and privacy toggles, guardrails, rate limit, audit log, LLM cost levers (cascade, semantic cache), web fallback, A2A protocol, agent memory, fine-tuning | `src/security/`, `src/agent/cascade.py`, `src/agent/semantic_cache.py`, `src/service/websearch.py`, `src/a2a/`, `src/agent/memory.py`, `scripts/finetune.py` |

The extensions fall into five categories, used for colour everywhere (docs, deck, replays; `scripts/replay_style.py`):
Distributed (sync vs async), Agentic Security (AuthN, AuthZ, guardrails, LLM abuse), Agentic Privacy (PII
redaction, audit), Agentic Cost Optimization (fine-tuning, LLM cost, LLM query rewrite, rate limiting) and Agentic
Extensions (A2A, memory, web search).

## 2. How a request flows

```
client ──POST /search──► router (src/service/router.py, :8000)
                           │  Policy.identify / check_rate / check_question   (src/security/policy.py; toggles)
                           │  embed the query once (the active model)
                           │  phase 1: stats  → every shard: N, total length, df of the query terms (DFS)
                           │  phase 2: search → every shard: terms + GLOBAL stats + vector, depth 100 per retriever
                           │  merge each retriever's lists across shards, then fuse (linear alpha 0.7, or RRF)
                           │  Policy.post_search: authz double-check, topic scope, passage injection, PII redaction
client ◄── top k, partial, missing_shards, model, timings_ms, security ──┘
                         shards (src/service/shard_server.py → store.py, :8101+ or Docker shard0..2)
```

- A doc lives on shard `crc32(doc_id) % N` (`src/index/sharding.py`).
- Writes (`/upsert`, `/delete`) go to the owning shard only. The shard checks the version, appends to its WAL and
  fsyncs, then either applies before acking (sync) or acks and lets background workers apply (async,
  `src/service/async_apply.py`, applied within 1 s). `/admin/flush` waits until every acked write is applied.
- Model migration: `/admin/migrate` builds the new index on every shard and turns on dual-write in the same lock as
  the backfill snapshot; `/admin/cutover` flips the router (refused until every shard is ready); `/admin/rollback`
  flips back; `/admin/finalize` drops a model's index. The router saves `active_model`/`previous_model` to its state
  file.
- Agents: `scripts/ask.py`, `scripts/agent_eval.py` and the A2A server call `/guardrails/check` first (a refused
  question never reaches an LLM; with `pii_redaction` on they work from the redacted text it returns), then search
  through `SearchTool` (`src/agent/tools.py`) and call the LLM through `src/agent/llm.py`.
- A2A: `src/a2a/server.py` (:8300) exposes the agents as A2A skills (`retrieve`, `single`, `multi`, `cascade`,
  `answer`) over JSON-RPC, with an agent card, tasks and memory; `src/a2a/coordinator.py` (:8310) is a second A2A
  agent that splits a compound question and delegates the parts with the caller's token.

## 3. File map

| path | what is in it |
|---|---|
| `src/config.py` | paths, the model registry (`MODELS`, `ModelSpec`, `model_path`), defaults (`CANDIDATE_DEPTH` 100, `RRF_K` 60, `DEFAULT_FUSION` linear, `DEFAULT_ALPHA` 0.7, `SHARD_TIMEOUT_S` 0.5), LLM prices; loads `.env` with `override=False` |
| `src/beir_data.py`, `src/search.py`, `src/eval.py`, `src/metrics.py` | data loading, the single-node search entry point, the eval harness (nDCG@10, Recall@k, bootstrap CIs), metrics with doctests |
| `src/retrieval/bm25.py` | BM25 with an alive mask (tombstones), `local_stats(terms, row_mask)` for global and per-tenant statistics, `search(..., stats, row_mask)` |
| `src/retrieval/dense.py` | embedders (`get_embedder`), the faiss flat index, `selector()` (an `IDSelectorBatch` for one tenant's ids) |
| `src/retrieval/hybrid.py`, `rerank.py` | min-max + linear fusion and RRF; the optional cross-encoder rerank |
| `src/index/build_index.py`, `sharding.py` | embedding caches in `artifacts/emb_<model>.npy`; doc-to-shard assignment |
| `src/service/router.py` | `Router` (scatter-gather, merge, fuse, migration admin) and `create_app()` (FastAPI endpoints, request models) |
| `src/service/store.py` | the shard's state: docs, versions, owners, per-model dense indexes, the per-tenant filters (`_row_mask`, `_selector`, cached until the next write), `_write` (version check, WAL, apply) |
| `src/service/wal.py`, `async_apply.py`, `shard_server.py`, `cluster.py` | write-ahead log; async workers; the shard's HTTP server; start N local shard processes and a router |
| `src/service/websearch.py` | web fallback (Wikipedia) when the corpus can't answer; the question is redacted before it leaves |
| `src/agent/llm.py` | `LLM` base (budget reserve, disk cache, counters `requests_sent`/`cache_hits`, `usage_line`), `FakeLLM`, `CopilotLLM`, `AzureLLM` (Responses API, refuses to construct without `confirm_spend=True`), `get_llm` |
| `src/agent/single_agent.py`, `orchestrator.py`, `cascade.py`, `semantic_cache.py` | the single agent loop, the multi-agent (free gate, planner, workers, verifier), the cheap-then-strong cascade, the cosine cache |
| `src/agent/tools.py`, `prompts.py`, `fake_policies.py`, `memory.py` | `SearchTool`, HTTP search and guard functions; every system prompt; scripted policies for the fake LLM; `MemoryStore` + `contextualize` |
| `src/security/` | `tokens.py` (HMAC-signed tokens), `tenants.py` (odd id acme, even globex), `config.py` (the ten toggles, persisted), `policy.py` (identify, rate, question checks, `post_search`, audit), `pii.py`, `guardrails.py` (harmful, injection, topic scope with per-model `SCOPE_MIN`), `ratelimit.py` (token buckets), `audit.py` (hash-chained JSONL) |
| `src/a2a/` | `protocol.py` (JSON-RPC, task states, errors), `server.py` (skills, tasks, `/memory`, `/ui`), `client.py`, `coordinator.py`, `ui.html` |
| `scripts/` | `ask.py` (one question), `agent_eval.py`, `llm_check.py` (Azure settings check + one uncached call), `demo_call.py` (the colourful HTTP helper the guide calls `call`), `mint_token.py`, `api_docs.py` (the Swagger page with named examples; the Docker router runs it), benchmarks, ablations, load and migration tools, `scenario.py` + `replay_html.py`, `run_simple_scenarios.py`, the replay generators (section 6), `finetune.py`, `guardrails_eval.py`, `docker_tour.*`, `bootstrap.*` |
| `scenarios/` | `SimpleScenarios/` (8 CSV stories, `results/<cluster>/<scenario>/` with console, events and the animated replay) and `ComplexScenarios/` (JSONL) |
| `replays/` | every recorded replay as a page that opens from disk, `index.html` by category, `data/` with the JSON each page was rendered from |
| `results/` | every number the docs quote (`bench.json`, `runs/`, `ablation_*.json`, `write_load.json`, `agent_eval*.json`, `guardrails_eval.json`, ...) |
| `docs/topics/00..22` | one topic per file, in the order of the project; `docs/guides/` hand-run guides (manual tests, the security/A2A/memory demo); `docs/design/` the generated decks and docx (`DesignV3.*` current, `DESIGN.*` the earlier version) plus `EXTENSIONS.md` |
| `tests/` | unit tests, in-process integration tests, and multi-process cluster tests (`conftest.py`, a tiny FiQA fixture) |

## 4. Invariants (break one and something silently goes wrong)

- **Sharded = single node.** Phase 1 sums every shard's statistics so BM25 uses global IDF, and each retriever's
  lists are merged across shards before fusion. Sharded quality must equal one node's (0.0% gap; tested).
- **Ack means durable.** A write is in the WAL and fsynced before the ack, in both modes. Versions only move
  forward: an older or equal version is a no-op, so retries and replays are safe; a delete leaves a tombstone.
- **Async freshness bound is 1 s.** Scenario checks rely on it; a search sent right after an async ack may miss.
- **Tenant filtering happens before top-k, inside both retrievers.** BM25 zeroes other tenants' rows and computes
  N, total length and df over the tenant's rows only (its own IDF); dense passes an `IDSelectorBatch` into the faiss
  scan. Filtering after ranking returns fewer than k (that is what `authz` alone shows, on purpose).
- **Ownership never moves.** The first writer of a doc id owns it; the owner is kept with the tombstone and in WAL
  records. Base-corpus docs: odd id acme, even id globex.
- **Every security toggle is off by default and persists across restarts.** They must be off for eval numbers and
  the Docker tour. `/admin/security` always needs an admin token, whatever the toggles.
- **PII.** The audit log and the web provider always get the question redacted; passages are redacted on the way
  out when `pii_redaction` is on; memory stores turns and facts redacted; with `pii_redaction` on every agent path
  searches and prompts with the redacted question. A plain `/search` uses the real words (shards store no questions).
- **Topic scope needs a calibrated threshold per model** (`guardrails.SCOPE_MIN`). A model without one is not
  checked, and the response carries a finding in mode `skipped`.
- **Web results reach an LLM only through the `answer` skill**, as `[W1]` lines after the passages, and only while
  the admin toggle `web_fallback` is on. They are already neutralised and redacted by the router. `"web"` in A2A
  metadata is refused on every other skill.
- **LLM spend is opt-in.** `AzureLLM` cannot be constructed without `confirm_spend=True`; budgets reserve the
  worst case before a call. The disk cache key is (backend, model, system, user, max tokens) and does not include
  the endpoint, so cached prompts succeed even with broken settings: `scripts/llm_check.py` bypasses the cache.
- **Memory is keyed by (tenant, user, contextId)**, and tenant and user come from the signed token. A contextId is
  only a label the client sends back; it is not a secret. Anonymous callers get no memory; entries expire after
  `MEMORY_TTL_DAYS` (30).
- **A2A identity is passed along.** The coordinator calls the FiQA agent with the caller's token, never its own.
  Tasks are visible only to their owner (anyone else gets `-32001 Task not found`).
- **The service contract is HTTP.** `scripts/` and the tests talk to the router over `/search`, `/upsert`,
  `/delete`, `/guardrails/check`, `/admin/*`; change a request or response shape only together with its callers,
  the tests and `scripts/api_docs.py`.

## 5. Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_data.py                     # FiQA -> data/raw/
python -m src.index.build_index                     # embedding caches -> artifacts/ (~30 min CPU)
```

`./scripts/bootstrap.sh` (or `.\scripts\bootstrap.ps1`) does the Docker version in one step and downloads the
embedding caches from the GitHub release instead of building them (the fine-tuned model comes from `scripts/finetune.py` or its own release asset; docs/topics/18_fine_tuning.md). Set
`HF_HUB_OFFLINE=1` once the models are cached, and on Windows `PYTHONIOENCODING=utf-8`. Copy `.env.example` to
`.env` for the Azure settings; `python scripts/llm_check.py --confirm-spend` checks them with one tiny call.

## 6. Commands

| what | command |
|---|---|
| all tests | `python -m pytest -q` (several minutes: starts real clusters) |
| quick tests | `python -m pytest -q tests/test_no_secrets.py tests/test_metrics.py tests/test_hybrid.py tests/test_agents.py` |
| local cluster | `python -m src.service.cluster --shards 3` (router :8000, shards :8101+); `PRIMARY_MODEL=minilm-ft` to serve the fine-tuned model |
| Docker cluster | `docker compose up -d --build` (router :8000; stop the local cluster first; rebuild every service after a pull that changed `src/`, the router and shards share one image) |
| API page | `http://localhost:8000/docs` (named examples, including the migration endpoints for `minilm`, `minilm-ft` and `bge-small`), or `python -m scripts.api_docs --local` without Docker |
| dev eval | `python -m src.eval --strategy hybrid --fusion linear --alpha 0.7 --name <name> --out <scratch file>` |
| Docker tour (checked; restores the stack) | `.\scripts\docker_tour.ps1` / `./scripts/docker_tour.sh` (needs a running stack) |
| a scenario | `python scripts/scenario.py scenarios/SimpleScenarios/2_SimpleReadAfterWrite.csv --start-cluster --tiny` |
| all eight on four clusters | `python scripts/run_simple_scenarios.py`, then `python scripts/scenario_pages.py` |
| one HTTP call, shown in full | `python -m scripts.demo_call POST /search '{"query": "x", "k": 3}' --as alice`; `python -m scripts.demo_call alloff` turns every toggle off |
| security tests | `python -m pytest -q tests/test_security.py src/security` (fast) and `tests/test_security_service.py` (toggle matrix, starts a cluster) |
| security on the live stack | `python scripts/security_demo.py` (every toggle on and off, then restored); tokens: `python -m scripts.mint_token admin` |
| guardrail calibration | `python scripts/guardrails_eval.py` (red-team sets, FiQA false blocks, scope thresholds for minilm-ft, minilm, bge-small) |
| cascade, semantic cache, web fallback tests | `python -m pytest -q tests/test_cascade.py --doctest-modules src/agent/cascade.py src/agent/semantic_cache.py src/service/websearch.py` (fast) |
| cascade on Azure / rules offline | `python scripts/agent_eval.py --backend azure --confirm-spend --systems cascade --cascade-shadow --out results/agent_eval_cascade.json`, then `python scripts/cascade_policies.py` (no LLM calls) |
| web fallback on the live stack | `python scripts/web_fallback_eval.py` (calls Wikipedia; toggles restored) |
| A2A + memory tests | `python -m pytest -q tests/test_a2a.py --doctest-modules src/a2a src/agent/memory.py` (fast, in-process) |
| A2A server / UI | `python -m src.a2a.server --router http://localhost:8000 --demo-logins` (:8300, fake LLM; `--backend azure --confirm-spend` for a real one); the page is `/ui` |
| one question through an agent | `python scripts/ask.py "..." --url http://127.0.0.1:8000 --strategy single|multi|cascade --backend azure --confirm-spend` |
| replays | `python scripts/security_replay.py --confirm-spend` (`--only 6` re-records one case), `python scripts/memory_replay.py --confirm-spend`, `python scripts/a2a_two_agents.py --confirm-spend`, `python scripts/replays_index.py`; each has `--render-only` |

Test clusters use their own ports (18000+, 19000+), so they can run while the dev cluster or Docker holds 8000.

## 7. Where to put a change

- **A new embedding model:** add a `ModelSpec` to `MODELS` in `src/config.py`, build its cache with
  `python -m src.index.build_index --models <key>`, calibrate its scope threshold with
  `scripts/guardrails_eval.py --models <key>` and add it to `guardrails.SCOPE_MIN`, and add Swagger examples in
  `scripts/api_docs.py`. Move a running stack with migrate, cutover, finalize.
- **A new security toggle:** declare it and its allowed values in `src/security/config.py`, enforce it in one place
  in `src/security/policy.py`, add a row to the toggle matrix in `tests/test_security_service.py`, a case to
  `scripts/security_replay.py` (with its category in `category_of`) and a section to `docs/topics/17_security.md`.
- **A new A2A skill:** add it to the card's skills in `src/a2a/server.py` and a branch in `run_task`; LLM skills
  must keep the guard check first and set `llm_calls`, `usd` and the token counts in the task metadata.
- **A new scenario:** copy `scenarios/SimpleScenarios/simple_template.csv`, which documents the columns (actor, operation, text, t_ms, doc_id,
  version); the runner refuses a client doing admin work. Re-run `scripts/scenario_pages.py` afterwards.
- **A new replay page:** use `scripts/replay_style.py` (`nav_html`, `CATEGORIES`, `NAV_CSS`) so it looks like the
  others: a left nav, the category colour, red only for errors, data embedded so it opens from disk; write it to
  `replays/` and its data to `replays/data/`, and add it to `scripts/replays_index.py`.
- **A number in a doc:** it must come from a file in `results/` or a recorded run, and the run's command goes next
  to it. Estimates are labelled as estimates.

## 8. Rules

- **No secrets in git.** The Azure key lives in `.env` (gitignored; see `.env.example`).
  `tests/test_no_secrets.py` also fails on keys, absolute home paths and internal hostnames in tracked files.
- **LLM spend is opt-in.** The default backend is `fake` (free, deterministic, used by all tests). `--backend azure`
  needs `--confirm-spend` and is capped by `--max-usd`; answers are cached in `.llm_cache/`.
- **Never evaluate on the test split** (`--split test`). It was used once for the final number; tune on dev.
- **Numbers come from runs.** Write scratch outputs somewhere other than `results/` (`--out`).
- **Don't commit generated or local data:** `data/raw/`, `artifacts/`, `dist/`, `.llm_cache/`, `toy_runs/`,
  `exercises/`, cluster logs, Office lock files (`~$*`). They are gitignored.
- **Docs style:** first person, plain sentences, no em or en dashes in prose, only measured numbers.
- Line endings: `.gitattributes` keeps `*.sh` as LF, or bash fails on Windows checkouts.

## 9. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| a run says `LLM calls 0` or `0 tokens` | a shell variable overrides `.env`, the answer came from the disk cache, or the multi-agent's free gate judged the question simple | `python scripts/llm_check.py --confirm-spend` shows the settings and where each came from |
| the Docker tour's shape check fails after a pull | only some services were rebuilt | `docker compose up -d --build` |
| `401`/`403` from plain curl calls | a security toggle is still on (they persist) | `python -m scripts.demo_call alloff` |
| Git Bash turns `/search` into a file path | MSYS path conversion | `export MSYS_NO_PATHCONV=1` |
| `UnicodeEncodeError` on Windows | the console code page | `PYTHONIOENCODING=utf-8` |
| model downloads on every start | Hugging Face checks the hub | `HF_HUB_OFFLINE=1` once the models are cached |
| guard_scope=block lets an off-topic question through | the served model has no calibrated threshold (a `skipped` finding says so) | calibrate it (section 7) |
