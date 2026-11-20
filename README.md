# FiQA Distributed + Agentic Retrieval

A retrieval system over **BEIR/FiQA** (57,638 finance Q&A passages):

- **Part A:** BM25, dense and hybrid retrieval with a full evaluation harness, ablations and failure analysis.
- **Part B:** a **sharded service** (router + 3 shards) with live, idempotent updates and **zero-downtime
  embedding-model migration**.
- **Part C:** a hand-written **single agent** and **multi-agent** orchestrator on top, evaluated on plain
  and compound questions.

This repo is also built **to learn from**. Every module has a LEARN doc, a runnable toy example,
exercises with solutions, ADRs and "live-modify" drills. Start at [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md).

> **New to search, embeddings and agents? Start at [docs/from_zero/](docs/from_zero/README.md).** It's a from-scratch guide
> (no prior knowledge assumed) with many diagrams, hand-worked examples on real FiQA data, a "why we're doing this"
> box for every step, and a milestone plan to rebuild the whole project yourself.
> Its hands-on part lives in **[exercises/](exercises/README.md)**: a stub, a worksheet and an automatic checker per chapter.
> Toy checks take seconds (`python -m pytest exercises`); `--full` runs your code on the real 57,638 docs and compares it
> with our numbers. The answer key is in `exercises/solutions/`.
Design, results and trade-offs are in [DESIGN.md](DESIGN.md).

## Headline numbers (dev split; full table in DESIGN.md)

| config (through the router, 3 shards) | nDCG@10 | R@10 | MRR@10 | p50 / p95 | peak RSS / shard |
|---|---|---|---|---|---|
| BM25 | 0.264 | 0.314 | 0.322 | 10.7 / 11.5 ms | 722 MB |
| dense MiniLM | 0.388 | 0.467 | 0.453 | 22.2 / 27.0 ms | 723 MB |
| **hybrid linear α=0.7 (chosen)** | **0.419** | **0.491** | **0.487** | **28.6 / 40.5 ms** | **723 MB** |

The sharded service reproduces single-node quality exactly (0.0% difference) for every strategy.

**Live writes, two modes** (`APPLY_MODE`; real corpus, 8 writers + 2 readers, `results/write_load.json`):
- **sync** (default): searchable on the very next query, 58 writes/s.
- **async**: 94 writes/s, searchable within 1 s (measured max 583 ms).

**Agents** (Azure `gpt-6-sol`, 100 plain + 100 compound dev questions, `results/agent_eval.json`):
- **Compound questions:** Recall@10 goes from 0.204 with one search to 0.400 with the single agent (p95 9.3 s) and
  0.355 with the multi-agent (p95 4.7 s, ~¼ of the tokens).
- **Plain questions:** stay on plain search by default.
- **Spend:** $0.79 for the whole eval, at placeholder prices. Details: DESIGN.md §7.

## Setup

**One command, from a fresh clone (Docker only):** `./scripts/bootstrap.sh` (Linux/macOS) or `.\scripts\bootstrap.ps1`
(Windows). It builds the image, fetches the data and embedding caches, starts the 4 containers and verifies them.
Details, options and troubleshooting: **[DEPLOY.md](DEPLOY.md)**.

### Option A: Docker by hand (no local Python needed)

```bash
docker compose build                       # corporate PyPI mirror? add --build-arg PIP_INDEX_URL=...
docker compose run --rm indexer            # download FiQA + embed corpus (~30 min CPU), once
#   or unzip the pre-built artifact zip (make package-index) into ./artifacts to skip embedding
docker compose up -d                       # router on http://localhost:8000 (+ 3 shards, 2 GB limit each)
#   APPLY_MODE=async docker compose up -d  # same stack with async writes (default: sync)
docker compose run --rm tools python src/search.py --url http://router:8000 --query "what is short selling?"
docker compose run --rm eval               # dev eval through the router (ours: nDCG@10 0.4190, p50/p95 30.4/34.1 ms)
```

### Option B: local Python 3.11+

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (Linux)
pip install -r requirements.txt
python scripts/download_data.py
python -m src.index.build_index                        # embedding caches -> artifacts/ (~30 min CPU)
python -m src.service.cluster --shards 3               # router :8000 + shards :8101-8103 (stop Docker first: same port)
python src/search.py --query "what is short selling?" --top-k 10 --strategy hybrid
```

## Make targets

`make` is optional: every target in the [Makefile](Makefile) is a one-line Python command you can
run directly (handy on Windows).

| target | what |
|---|---|
| `make test` | unit + integration + doctests + toy smoke tests |
| `make toys` | run all 27 toy examples |
| `make bench` | the full benchmark matrix → `results/bench.json` |
| `make ablations` | fusion sweeps + rerank sweep |
| `make failures` | failure candidates → `results/failure_candidates.json` |
| `make migrate` / `cutover` / `rollback` / `finalize` | B3 against the running router (`make load` in another terminal) |
| `make migration-demo` | migrate → cutover → rollback under live queries + writes, with a report |
| `make compound` | rebuild `data/compound_dev.jsonl` |
| `make agent-dry-run` | token/$ estimate for the Azure agent eval (free) |
| `make agent-eval` | **costs money**: the Azure agent evaluation (needs `.env`) |
| `make test-final` | the single test-split run of the chosen configuration |

**Scenario runner + replay:** script who does what (🦊 users, 🤖 bots, 🛠️ admin, 🧑‍🔧 ops): writes, searches,
migrations and shard crashes in a small JSONL file, run it against a live cluster in sync or async mode, and
get pass/fail checks plus an animated HTML replay:
`python scripts/scenario.py scenarios/freshness_demo.jsonl --start-cluster --tiny --mode async`.
Guide and scenario catalog: [docs/scenarios.md](docs/scenarios.md).

## LLM configuration (Part C)

Copy `.env.example` to `.env` and fill in `AZURE_OPENAI_ENDPOINT` (Foundry project `/openai/v1/` URL),
`AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_API_KEY`. `.env` is gitignored, and `tests/test_no_secrets.py`
fails if a key-like string is ever committed. The default backend is the free `fake`. `azure` needs
`--confirm-spend` and is capped by `--max-usd`.

## Repo map

```
src/            search.py (CLI) · eval.py · metrics.py · config.py · beir_data.py · util.py
  retrieval/    bm25.py · dense.py · hybrid.py · rerank.py
  index/        build_index.py (embedding cache) · sharding.py
  service/      router.py · shard_server.py · store.py · async_apply.py (async write mode) · wal.py · cluster.py
  agent/        llm.py · tools.py · prompts.py · single_agent.py · orchestrator.py · fake_policies.py
data/           build_compound.py · compound_dev.jsonl · compound_rewrites.json · compound_review_sheet.md
scripts/        bench.py · ablations.py · failure_cases.py · migration.py · query_load.py · migration_demo.py · agent_eval.py · scenario.py (+ replay_html.py)
                write_load.py (sync vs async) · docker_check.py · docker_trace.py · bootstrap.sh / bootstrap.ps1
scenarios/      runnable JSONL scenarios for scripts/scenario.py (freshness, migration, shard failure, ...)
tests/          unit, in-process and multi-process integration, toy smoke tests
results/        bench.json · runs/ (incl. the one TEST run) · ablation_*.json · migration_demo.json · write_load.json · agent_eval*.json · failures.md
examples/       27 narrated toy scripts (offline, seconds each)
docs/           LEARN docs 00-15 · from_zero/ (study guide) · GLOSSARY · LEARNING_PATH · adr/ · DRILLS · project_faq · scenarios · solutions/ · walkthroughs/
exercises/      per-chapter stubs + checkers for docs/from_zero (answer key in exercises/solutions/)
agent_context/  machine-readable handoff for AI agents (see AGENTS.md)
```

## Measured on

AMD EPYC 7763 (16 physical / 32 logical cores visible), 128 GB RAM, Windows 11, Python 3.13, CPU only.
Latency is hardware-dependent; quality metrics are not.
