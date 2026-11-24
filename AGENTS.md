# AGENTS.md

Guidance for coding agents (and people) working in this repository. The human-facing overview is
[README.md](README.md); the design and the numbers are in [DESIGN.md](DESIGN.md).

## What this repo is

A retrieval system over BEIR FiQA (57,638 finance passages):

- `src/retrieval/`, `src/eval.py`: BM25, dense and hybrid retrieval plus the evaluation harness (Part A).
- `src/service/`: a router and N shards with versioned writes, a WAL, sync or async apply, and embedding-model
  migration (Part B). `src/service/cluster.py` runs it as local processes; `docker-compose.yml` runs it in Docker.
- `src/agent/`: a single agent and a multi-agent orchestrator on top of the search service (Part C).
- `scripts/`: benchmarks, ablations, load and migration tools, the scenario runner (`scenario.py` +
  `replay_html.py`), `ask.py` (one question), `api_docs.py` (the API page with examples).
- `scenarios/`: `SimpleScenarios/` (CSV stories with recorded runs) and `ComplexScenarios/` (JSONL).
- `results/`: every number quoted in the docs. `docs/`: topic docs, guides, design files.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_data.py                     # FiQA -> data/raw/
python -m src.index.build_index                     # embedding caches -> artifacts/ (~30 min CPU)
```

`./scripts/bootstrap.sh` (or `.\scripts\bootstrap.ps1`) does the Docker version in one step and downloads the
embedding caches from the GitHub release instead of building them. Set `HF_HUB_OFFLINE=1` once the models are
cached, and on Windows `PYTHONIOENCODING=utf-8`.

## Commands

| what | command |
|---|---|
| all tests | `python -m pytest -q` (several minutes: starts real clusters) |
| quick tests | `python -m pytest -q tests/test_no_secrets.py tests/test_metrics.py tests/test_hybrid.py tests/test_agents.py` |
| local cluster | `python -m src.service.cluster --shards 3` (router :8000, shards :8101+) |
| Docker cluster | `docker compose up -d` (router :8000; stop the local cluster first) |
| API page | `http://localhost:8000/docs`, or `python -m scripts.api_docs --local` without Docker |
| dev eval | `python -m src.eval --strategy hybrid --fusion linear --alpha 0.7 --name <name> --out <scratch file>` |
| Docker tour (guide section 3, checked; restores the stack) | `.\scripts\docker_tour.ps1` / `./scripts/docker_tour.sh` (needs a running stack) |
| a scenario | `python scripts/scenario.py scenarios/SimpleScenarios/2_SimpleReadAfterWrite.csv --start-cluster --tiny` |
| security tests | `python -m pytest -q tests/test_security.py src/security` (fast) and `tests/test_security_service.py` (toggle matrix, starts a cluster) |
| security on the live stack | `python scripts/security_demo.py` (every toggle on and off, then restored); tokens: `python -m scripts.mint_token admin` |
| cascade, semantic cache, web fallback tests | `python -m pytest -q tests/test_cascade.py --doctest-modules src/agent/cascade.py src/agent/semantic_cache.py src/service/websearch.py` (fast); web fallback service tests are in `tests/test_security_service.py` |
| cascade on Azure / rules offline | `python scripts/agent_eval.py --backend azure --confirm-spend --systems cascade --cascade-shadow --out results/agent_eval_cascade.json`, then `python scripts/cascade_policies.py` (no LLM calls) |
| web fallback on the live stack | `python scripts/web_fallback_eval.py` (calls Wikipedia; toggles restored) |
| A2A + memory tests | `python -m pytest -q tests/test_a2a.py --doctest-modules src/a2a src/agent/memory.py` (fast, in-process) |
| A2A server / demo | `python -m src.a2a.server --router http://localhost:8000` (:8300, fake LLM); `python scripts/a2a_demo.py` (starts its own) |

Test clusters use their own ports (18000+, 19000+), so they can run while the dev cluster or Docker holds 8000.

Security toggles ([docs/topics/17_security.md](docs/topics/17_security.md)) are all off by default and must be off for the
eval numbers and the Docker tour; `GET /admin/security` with an admin token shows them. That includes `web_fallback`,
which sends questions to Wikipedia ([docs/topics/20_web_search.md](docs/topics/20_web_search.md)).

## Rules

- **No secrets in git.** The Azure key lives in `.env` (gitignored; see `.env.example`).
  `tests/test_no_secrets.py` also fails on keys, absolute home paths and internal hostnames in tracked files.
- **LLM spend is opt-in.** The default backend is `fake` (free, deterministic, used by all tests). `--backend azure`
  needs `--confirm-spend` and is capped by `--max-usd`; answers are cached in `.llm_cache/`.
- **Never evaluate on the test split** (`--split test`). It was used once for the final number; tune on dev.
- **Numbers come from runs.** A number written into a doc must come from a file in `results/` or a recorded run.
  Write scratch outputs somewhere other than `results/` (`--out`).
- **Don't commit generated or local data:** `data/raw/`, `artifacts/`, `dist/`, `.llm_cache/`, `toy_runs/`, cluster
  logs. They are gitignored.
- **Keep the service contract stable.** `scripts/` and the tests talk to the router over HTTP (`/search`, `/upsert`,
  `/delete`, `/admin/*`); change a request or response shape only together with its callers and tests.
- Line endings: `.gitattributes` keeps `*.sh` as LF, or bash fails on Windows checkouts.
