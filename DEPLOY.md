# Deploy: from a fresh clone to a running, checked system

One command sets up everything: the data, the embedding caches, the Docker image, the 4 containers (3 shards + router)
and a smoke test. I made it safe to re-run: every step skips work that is already done.

| machine | command (from the repo root) |
|---|---|
| Linux / macOS | `./scripts/bootstrap.sh` |
| Windows (PowerShell) | `.\scripts\bootstrap.ps1` (if blocked: `powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1`) |

| bash | PowerShell | what it does |
|---|---|---|
| `--mode async` | `-Mode async` | start the shards in async write mode ([09](docs/topics/09_live_updates_consistency.md)); default `sync` |
| `--migrate-check` | `-MigrateCheck` | also run a live blue/green model migration (minilm → bge-small → rollback → cleanup) as part of the check |
| `--build-index` | `-BuildIndex` | encode the embeddings locally (~30 to 40 min CPU) instead of downloading them |
| `--down` | `-Down` | stop all containers (data, caches and WALs stay on disk) |

## Prerequisites

| | Linux | macOS | Windows |
|---|---|---|---|
| Docker | Docker Engine ≥ 24 with the Compose v2 plugin (`docker compose version` must work) | Docker Desktop ≥ 4.x (Intel or Apple Silicon) | Docker Desktop ≥ 4.x (WSL 2 backend) |
| memory for Docker | ≥ 8 GB recommended, ≥ 6 GB minimum | Docker Desktop → Settings → Resources → Memory ≥ 8 GB | same as macOS |
| disk | ~10 GB free (image ≈ 2.8 GB, data + caches ≈ 0.3 GB, build cache) | same | same |
| network | first run only: PyPI (or a mirror), download.pytorch.org, GitHub, the BEIR data URL | same | same |
| port | `8000` free on the host (the router) | same | same |
| other tools | `git`, `curl`, and `gh` only if the repo is private (to download the pre-built caches) | same (`curl` is built in) | optional `gh` |

The host needs no Python. Everything Python runs inside the project image. On Linux the user must be able to run
`docker` (`sudo usermod -aG docker $USER`, then log out and back in), or run the script with `sudo`.

## What the bootstrap does

| step | what happens | first run | re-run |
|---|---|---|---|
| 0. prerequisites | checks docker, compose v2, that the daemon is running, memory, disk, and that port 8000 is free | seconds | seconds |
| 1. configuration | creates `.env` from `.env.example` if missing (only optional LLM settings); creates `data/raw`, `artifacts`, `results/runs`; detects a PyPI mirror (`PIP_INDEX_URL`, else `pip config`, else pypi.org) | seconds | seconds |
| 2. image | `docker compose build`: python 3.11.18-slim + CPU torch 2.15.0 + every package pinned by `constraints.txt`. The 3 models are baked in, so containers start offline. Builds natively on amd64 and arm64 (Apple Silicon) | 5 to 10 min | cached, seconds |
| 3. data | BEIR FiQA → `data/raw/fiqa` (57,638 passages, 6,648 queries), downloaded inside a container. Fallback: a copy attached to the GitHub release | ~1 min | skipped |
| 4. embeddings | `artifacts/emb_minilm.*` and `artifacts/emb_bge-small.*` (88 MB each), downloaded from GitHub release `index-artifacts-v1`. Fallback (or `--build-index`): encoded inside a container | ~1 min (download) / 30 to 40 min (encode) | skipped |
| 5. start | `docker compose up -d --wait`: 4 containers from 1 image, `shard0`, `shard1`, `shard2` (2 GB memory cap each) and `router` (port 8000). The router starts only after all shards are healthy, and `--wait` returns when every healthcheck passes | 1 to 2 min | seconds |
| 6. verify | `scripts/smoke_test.py` inside a container against `http://router:8000`: 3 healthy shards with 57,638 docs, a hybrid search with 10 results and no partial flag, freshness (sync: first search after the ack; async: within 1 s), idempotency. With `--migrate-check` also a full migration + rollback | ~10 s (+ ~1 min migration) | same |

Every container bind-mounts `./data/raw` → `/app/data/raw` and `./artifacts` → `/app/artifacts`. That is how they
share one copy of the data and caches, and why the shard WALs (`artifacts/docker/shardN/wal.jsonl`) and the router
state (`artifacts/docker/router_state.json`) survive restarts.

## Once it is up

| task | command |
|---|---|
| search | `curl -s -X POST localhost:8000/search -H 'content-type: application/json' -d '{"query":"what is short selling?"}'` |
| see every shard (docs, models, write mode, async lag) | `curl -s localhost:8000/admin/status` |
| browse the API | http://localhost:8000/docs |
| add / delete a document | `curl -s -X POST localhost:8000/upsert -H 'content-type: application/json' -d '{"doc_id":"my-1","version":1,"text":"..."}'` |
| full dev evaluation (500 queries) | `docker compose --profile tools run --rm eval` (my run: nDCG@10 0.4190) |
| scenario files (writes, searches, migrations, crashes) with an HTML replay | see [docs/guides/scenarios.md](docs/guides/scenarios.md) |
| deeper Docker check: kill/restart a shard container, migration, memory, eval (needs host Python + venv) | `python -m scripts.docker_check --mode both --eval` |
| show that the router talks to 3 separate containers: container IPs, DNS inside the router, writes landing in the predicted container (each shard asked directly via `docker exec`), one search merging all 3, TCP connections from the router's IP, stop/start one shard (needs host Python + venv and a running stack) | `python -m scripts.docker_trace --failover` |
| migrate the embedding model by hand | `make migrate`, `make cutover`, `make rollback`, `make finalize` (or the `/admin/*` endpoints) |
| logs | `docker compose logs -f router shard0` |
| switch write mode | `./scripts/bootstrap.sh --mode async` (recreates the shard containers with `APPLY_MODE=async`) |
| stop (keeps everything on disk) | `./scripts/bootstrap.sh --down` |
| start again | `./scripts/bootstrap.sh` (all steps skip; ~1 to 2 min) |
| wipe all written docs and router state (keeps data + caches) | stop, then delete `artifacts/docker/` |

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `pip` timeouts or `No matching distribution` in step 2 | pypi.org is blocked or slow (corporate network). Set the mirror: `PIP_INDEX_URL=https://<mirror>/simple ./scripts/bootstrap.sh` |
| step 4 says "release download not possible" | the GitHub repo is private and there is no token. Run `gh auth login`, or `export GITHUB_TOKEN=<a token with repo read access>` (PowerShell: `$env:GITHUB_TOKEN="..."`), and re-run. Or let it encode locally (30 to 40 min, once) |
| containers killed, `exit 137`, or shards restarting in a loop | Docker has too little memory. Give Docker Desktop ≥ 8 GB |
| `port 8000 is already used` | another program holds port 8000, possibly an old stack from a different clone. Find it with `docker ps` and `docker stop` it |
| `--wait` times out in step 5 | check `docker compose logs shard0`. Usually missing caches (re-run, step 4 repairs it) or too little memory |
| Linux: `Permission denied` when deleting files in `artifacts/` | containers run as root, so new files (WALs) are root-owned. The bootstrap hands caches and data back to you; for later files use `sudo rm -rf artifacts/docker` |
| Apple Silicon: slow first build | expected: arm64 wheels are built or downloaded natively. Later builds are cached |
| `.env` turned into a folder | a `docker compose run` happened before `.env` existed. `rmdir .env` and re-run the bootstrap (it creates the file first) |
| "I only see one container" | Docker Desktop's Containers tab groups a compose stack into one row (`fiqa-agentic-retrieval`); click ▸ to expand it into 4. The Images tab correctly shows one image. `docker ps` lists all 4 |

## Without Docker (local Python, for development)

```bash
python3.11 -m venv .venv && . .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints.txt
python scripts/download_data.py && python -m src.index.build_index   # or unzip the release zip at the repo root
python -m pytest -q                                                   # 100+ tests, ~3 min
python -m src.service.cluster --shards 3                              # router :8000 + shards :8101-8103
```
