# FiQA Distributed + Agentic Retrieval

A retrieval system over **BEIR/FiQA** (57,638 finance Q&A passages):

- **Part A:** BM25, dense and hybrid retrieval with a full evaluation harness, ablations and failure analysis.
- **Part B:** a **sharded service** (router + 3 shards) with live, idempotent updates and **zero-downtime
  embedding-model migration**.
- **Part C:** a hand-written **single agent** and **multi-agent** orchestrator on top, evaluated on plain
  and compound questions.

This repo is also built **to learn from**. Every module has a LEARN doc, a runnable toy example,
exercises with solutions, ADRs and "live-modify" drills. Start at [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md).
Design, results and trade-offs are in [DESIGN.md](DESIGN.md).

## Headline numbers (dev split; full table in DESIGN.md)

| config (through the router, 3 shards) | nDCG@10 | R@10 | MRR@10 | p50 / p95 | peak RSS / shard |
|---|---|---|---|---|---|
| BM25 | 0.264 | 0.314 | 0.322 | 10.7 / 11.5 ms | 722 MB |
| dense MiniLM | 0.388 | 0.467 | 0.453 | 22.2 / 27.0 ms | 723 MB |
| **hybrid linear α=0.7 (chosen)** | **0.419** | **0.491** | **0.487** | **28.6 / 40.5 ms** | **723 MB** |

The sharded service reproduces single-node quality exactly (0.0% difference) for every strategy.

## Setup

### Option A: Docker (no local Python needed)

```bash
docker compose build                       # corporate PyPI mirror? add --build-arg PIP_INDEX_URL=...
docker compose run --rm indexer            # download FiQA + embed corpus (~30 min CPU), once
#   or unzip the pre-built artifact zip (make package-index) into ./artifacts to skip embedding
docker compose up -d                       # router on http://localhost:8000 (+ 3 shards, 2 GB limit each)
docker compose run --rm tools python src/search.py --url http://router:8000 --query "what is short selling?"
docker compose run --rm eval               # dev eval through the router
```

### Option B: local Python 3.11+

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (Linux)
pip install -r requirements.txt
python scripts/download_data.py
python -m src.index.build_index                        # embedding caches -> artifacts/ (~30 min CPU)
python -m src.service.cluster --shards 3               # router :8000 + shards :8101-8103
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
  service/      router.py · shard_server.py · store.py · wal.py · cluster.py
  agent/        llm.py · tools.py · prompts.py · single_agent.py · orchestrator.py · fake_policies.py
data/           build_compound.py · compound_dev.jsonl · compound_rewrites.json · compound_review_sheet.md
scripts/        bench.py · ablations.py · failure_cases.py · migration.py · query_load.py · migration_demo.py · agent_eval.py
tests/          unit, in-process and multi-process integration, toy smoke tests
results/        bench.json · ablation_*.json · migration_demo.json · agent_eval*.json · failures.md
examples/       27 narrated toy scripts (offline, seconds each)
docs/           LEARN docs 00-15 · GLOSSARY · LEARNING_PATH · adr/ · DRILLS · project_faq · solutions/ · walkthroughs/
```

## Measured on

AMD EPYC 7763 (16 physical / 32 logical cores visible), 128 GB RAM, Windows 11, Python 3.13, CPU only.
Latency is hardware-dependent; quality metrics are not.
