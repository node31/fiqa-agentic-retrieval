# Manual tests on macOS

Step by step: bring the system up, try every retrieval mode in one process, then the same things against the sharded
Docker cluster. Every command runs in Terminal (zsh or bash) from the repo root. The Windows version is
[manual_tests_windows.md](manual_tests_windows.md).

Nothing here has run on a Mac. `bootstrap.sh` ran on a clean Linux box, and the `curl` snippets in section 3 ran in Git
Bash on Windows against the Docker stack; the rest I checked by reading. Treat the first run as the real test. The expected numbers next to each
step come from my Windows laptop; quality numbers must match exactly, latencies will differ.

Everything that writes a file writes under `toy_runs/manual/` (gitignored), never over the committed numbers in
`results/`.

## 0. What you need

| | why |
|---|---|
| Docker Desktop, running | the sharded cluster (section 3) |
| Python 3.11+ (`python3 --version`; `brew install python@3.13` if older) | the host tools: `ask.py`, the evals, scenarios |
| ~5 GB free disk | data (~65 MB), embedding caches (~170 MB), the image (~2.8 GB), the venv (~1.1 GB) |
| an Azure OpenAI key (optional) | only for the agent steps; everything else is free |

On Apple Silicon the image should build for arm64: the base image is multi-arch and the CPU torch index has aarch64
wheels. I have not tried it.

## 1. Bring the system up

### 1a. The Docker cluster (router + 3 shards)

```bash
./scripts/bootstrap.sh
```

It checks prerequisites, creates `.env` from `.env.example`, builds the image, downloads FiQA and the embedding caches
into `data/` and `artifacts/`, starts the 4 containers and runs a 6-check smoke test inside a container. It took ~545 s
on a clean Linux box; a re-run skips what is already there. `--mode async` starts the async write mode, `--down` stops
everything and keeps the data. More options are in [DEPLOY.md](../../DEPLOY.md).

Check it:

```bash
docker compose ps                                              # router, shard0, shard1, shard2: all "healthy"
curl -s http://localhost:8000/admin/status | python3 -m json.tool   # active_model minilm, 3 shards, ~19.2K docs each
open http://localhost:8000/docs                                   # every API in the browser, with examples (section 4)
```

### 1b. Host Python (for sections 2 and 3)

The host tools reuse the `data/` and `artifacts/` that bootstrap downloaded, so there is nothing to rebuild:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # on macOS the PyPI torch wheel has no CUDA, so no special index
```

Every new Terminal window needs:

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1                  # models from the local cache; leave it out the very first time so MiniLM downloads once
```

### 1c. The LLM key (only for the agent steps)

Open `.env` (`open -e .env`) and fill in `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (gpt-6-sol) and
`AZURE_OPENAI_API_KEY`. `.env` is gitignored. Every LLM answer is cached in `.llm_cache/`, so asking the same thing twice
costs nothing. Without a key, use `--backend copilot` (free, through the GitHub Copilot CLI, ~40 s per call) or
`--backend fake` (scripted, only shows the plumbing).

## 2. Single process (no Docker, no HTTP)

Here the whole corpus lives in one Python process (`single_node_router()`): one shard, no network. It is the baseline
the sharded cluster has to match.

### 2a. BM25, dense and hybrid on your own question

```bash
python scripts/ask.py "what is a roth ira?" --strategy bm25
python scripts/ask.py "what is a roth ira?" --strategy dense
python scripts/ask.py "what is a roth ira?" --strategy hybrid
```

Each run loads the corpus, the indexes and the encoder first (~30 s), then prints the top 10 with scores and a snippet.
On "what is a roth ira?" hybrid's #1 is doc 553031 ("A Roth IRA isn't an investment, per se..."). `-k 3` prints fewer
hits.

### 2b. BM25, dense and hybrid on the dev set (500 questions)

```bash
python -m src.eval --mode local --strategy bm25   --name my-bm25   --out toy_runs/manual/bm25_dev.json
python -m src.eval --mode local --strategy dense  --name my-dense  --out toy_runs/manual/dense_dev.json
python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.7 --name my-hybrid --out toy_runs/manual/hybrid_dev.json
```

| strategy | nDCG@10 | R@10 | warm p50 (my Windows laptop) |
|---|---|---|---|
| bm25 | 0.2642 | 0.3135 | ~1 ms |
| dense (MiniLM) | 0.3881 | 0.4665 | ~19 ms |
| hybrid (linear, α 0.7) | 0.4190 | 0.4914 | ~21 ms |

Hybrid needs `--fusion linear --alpha 0.7`: the eval's own defaults are RRF and α 0.5, while the service ships linear
0.7 (tuned on dev, [16_ablations.md](../topics/16_ablations.md)). `--limit 50` gives a quick partial run. Never pass
`--split test`: the test split was used once, for the final number.

### 2c. Single agent and multi-agent on your own question

```bash
python scripts/ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy single --backend azure --confirm-spend
python scripts/ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy multi  --backend azure --confirm-spend
```

The single agent prints each step: step 0 searches the question as asked (no LLM call), then the model either asks for
another search (with its query) or finishes. The multi-agent prints the plan (the free gate's "simple", the planner's
"simple", or the sub-queries it split into) and any verifier rewrites. Both end with the top 10, the LLM calls, tokens
and the estimated cost. A compound question costs about $0.001 to $0.004.

### 2d. Single agent and multi-agent on the compound set

`data/compound_dev.jsonl` has 100 two-topic questions built from pairs of dev questions (`data/build_compound.py`,
seed 42). Start with 10:

```bash
python scripts/agent_eval.py --backend azure --confirm-spend --systems single --sets compound --limit 10 --out toy_runs/manual/single_compound.json
python scripts/agent_eval.py --backend azure --confirm-spend --systems multi  --sets compound --limit 10 --out toy_runs/manual/multi_compound.json
```

Drop `--limit` for all 100. `--dry-run` prints the token and dollar estimate without calling anything, and `--max-usd`
(default 2) stops the run before it spends more. `--systems baseline,single,multi --sets plain,compound` is the full
comparison. My full run (gpt-6-sol, $0.79 in total at placeholder prices):

| compound set, 100 questions | R@10 | MRR@10 | Coverage@10 | LLM calls | $/question |
|---|---|---|---|---|---|
| baseline (plain hybrid, no LLM) | 0.204 | 0.365 | 0.10 | 0 | 0 |
| single agent | 0.400 | 0.609 | 0.40 | 2.7 | 0.0039 |
| multi-agent | 0.355 | 0.518 | 0.32 | 2.0 | 0.0013 |

Coverage@10 is the share of questions where both halves have a relevant doc in the top 10. On 10 questions the numbers
move a lot; my first 5 gave single R@10 0.293.

### 2e. Create your own scenario

A scenario is a script of searches, writes and admin actions, checked as it runs and replayed as an HTML page. The
simple format is one line per operation: `actor, operation, text, t_ms[, doc_id]`. Copy the template and edit it:

```bash
mkdir -p toy_runs
cp scenarios/SimpleScenarios/simple_template.csv toy_runs/my_scenario.csv
open -e toy_runs/my_scenario.csv
```

For example:

```text
fox, search, "roth ira", 10
elephant, search:bm25, "investments in mortgage", 20
bot1, upsert, "roth investments for IRA are as follows", 40
fox, wait_visible, , 60
ADMIN, migrate, , 100
ADMIN, wait_ready, , 120
ADMIN, cutover, , 200
fox, search, "roth ira", 220
ADMIN, rollback, , 300
ADMIN, finalize, , 400
```

TextEdit may turn straight quotes into curly ones; switch that off (Edit > Substitutions > Smart Quotes) or use any
code editor. Run it on a fresh local cluster and open the replay when it finishes:

```bash
python scripts/scenario.py toy_runs/my_scenario.csv --start-cluster --tiny --open             # 24-doc toy corpus, ~10 s
python scripts/scenario.py toy_runs/my_scenario.csv --start-cluster --shards 1 --open          # real corpus, one shard
python scripts/scenario.py toy_runs/my_scenario.csv --start-cluster --mode async --open        # real corpus, 3 shards, async writes
```

`--start-cluster` starts its own router and shards as local processes on ports 19000 and 19101+, so it never touches
Docker. The run prints one line per op and a summary ending in `RESULT PASS`, and writes
`toy_runs/scenarios/<name>-<mode>-<time>/replay.html` (the path is printed at the end). To open it later:
`open toy_runs/scenarios/<run>/replay.html`. The template passes with 22 ops and 0 errors in both modes.

In the replay, press Space to play. Under the narration there is a filter bar: click persona chips (say fox and bot1)
to see only their cards in both lanes, or type a doc id or query text (`bot1-1`, `roth`) to see only the cards that
mention it. Replays recorded before the filters existed get them with
`python scripts/scenario.py --replay-only toy_runs/scenarios/<run>/events.jsonl`; nothing is re-run.

The full format, every operation and the JSONL format for parallel lanes are in [scenarios.md](scenarios.md).
`--to-jsonl` prints what a CSV turns into.

### 2f. Run the simple scenarios

`scenarios/SimpleScenarios/` has eight short stories in the CSV format, each building on the one before: two users
searching, a write then a search for it, a delete, a model migration, a migration with a write before it, one with a
write during it, a rollback, and retries and stale writes (idempotency). Each file's header says what it shows and what
to look for; [scenarios/SimpleScenarios/README.md](../../scenarios/SimpleScenarios/README.md) has the overview.

One scenario, one cluster, replay opened at the end:

```bash
python scripts/scenario.py scenarios/SimpleScenarios/3_SimpleReadAfterWriteAndDelete.csv --start-cluster --tiny --open               # tiny, sync
python scripts/scenario.py scenarios/SimpleScenarios/3_SimpleReadAfterWriteAndDelete.csv --start-cluster --tiny --mode async --open  # tiny, async
python scripts/scenario.py scenarios/SimpleScenarios/3_SimpleReadAfterWriteAndDelete.csv --start-cluster --open                      # real corpus, sync
python scripts/scenario.py scenarios/SimpleScenarios/3_SimpleReadAfterWriteAndDelete.csv --start-cluster --mode async --open         # real corpus, async
```

The tiny runs take ~13 s each (mostly starting the cluster), the real-corpus ones ~50 s. Each starts its own fresh
3-shard cluster on ports 19000/19101+ and stops it afterwards, so runs never see each other's writes and Docker is not
touched. The replay goes to `toy_runs/scenarios/<name>-<mode>-<time>/replay.html`.

All eight on all four clusters (SimpleSyncTinyCluster, SimpleAsyncTinyCluster, SimpleSyncFullCluster,
SimpleAsyncFullCluster), ~17 min:

```bash
python scripts/run_simple_scenarios.py
python scripts/run_simple_scenarios.py --configs SimpleSyncTinyCluster --scenarios 4 7   # a subset
open scenarios/SimpleScenarios/results/SimpleAsyncFullCluster/2_SimpleReadAfterWrite/replay.html
```

The results go to `scenarios/SimpleScenarios/results/<cluster>/<scenario>/` (console.txt, events.jsonl, replay.html),
with a grid of every run in `results/SUMMARY.md`. My runs are checked in (console.txt, events.jsonl and replay.html for each,
plus SUMMARY.md): all 32 pass, the migrated searches are answered by bge-small and the rolled-back ones by
minilm, and scenario 8's retry and late writes are no-ops.

## 3. The sharded cluster (Docker)

Same tests, now through the router at `http://localhost:8000`: every search fans out to 3 shard containers, every
write goes to the one shard that owns the doc.

> **All of section 3 in one command.** `./scripts/docker_tour.sh` runs every step below except 3h and 3i against the
> running stack, checks each result against the numbers in this section, and writes a report with an animated replay
> per step (`toy_runs/docker_tour/<stamp>/index.html`, opened at the end). It stops and starts shard1, then deletes the docs
> it wrote and verifies the stack is back in exactly the shape it found it: containers, write mode, models, live docs
> per shard and the answers to six probe searches. About 2 to 3 minutes; needs only Docker and the stack in sync mode
> (the bootstrap default). 3i stays manual: the tour never changes the write mode. `--azure` adds the agent step
> (3e, costs a few cents), `--skip eval,simple` or `--only writes,shard` pick phases (search, eval, writes, migration, simple,
> agents, shard), and `--no-open` skips opening the report.

```bash
U=http://localhost:8000
J='Content-Type: application/json'
```

### 3a. Search with each strategy over raw HTTP

```bash
for s in bm25 dense hybrid; do
  curl -s -X POST $U/search -H "$J" -d "{\"query\":\"what is a roth ira?\",\"k\":3,\"strategy\":\"$s\"}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('$s', 'partial', r['partial'], 'model', r['model'], [h['doc_id'] for h in r['results']])"
done
```

Expect `partial False` and the same ids as the single process (hybrid: 553031, 488556, 214174). Add
`"include_text": true` to the body for passage texts. The body also takes `fusion` (`linear`/`rrf`), `alpha` and
`rerank_k`. Each response has `timings_ms`, the router's split of where the time went.

The same through the tools:

```bash
python scripts/ask.py "what is a roth ira?" --strategy hybrid --url $U
python src/search.py --url $U --query "what is short selling?" --top-k 10 --strategy hybrid
```

### 3b. The dev set through the router

```bash
python -m src.eval --mode service --url $U --strategy hybrid --fusion linear --alpha 0.7 --name my-docker-hybrid --out toy_runs/manual/docker_hybrid_dev.json
```

Expect nDCG@10 0.4190 and R@10 0.4914, exactly the single-process numbers (global BM25 statistics + a global merge make
3 shards rank like 1). Latency is higher: p50 ~30 ms in my reference run, against ~21 ms in one process. Swap
`--strategy bm25` or `dense` for the other two. `docker compose --profile tools run --rm eval` does the hybrid run
inside a container, but it writes `results/runs/docker-hybrid-linear-minilm.json`.

### 3c. Writes: upsert, search, delete

```bash
curl -s -X POST $U/upsert -H "$J" -d '{"doc_id":"my-test-1","version":1,"text":"zebra bonds are a made up product for my manual test"}'; echo
curl -s -X POST $U/search -H "$J" -d '{"query":"zebra bonds manual test","k":3,"strategy":"bm25"}' | python3 -m json.tool
curl -s -X POST $U/delete -H "$J" -d '{"doc_id":"my-test-1","version":2}'; echo
curl -s -X POST $U/search -H "$J" -d '{"query":"zebra bonds manual test","k":3,"strategy":"bm25"}' | python3 -m json.tool
curl -s -X POST $U/upsert -H "$J" -d '{"doc_id":"my-test-1","version":1,"text":"zebra again"}'; echo
```

What I got: the upsert says `"applied": true, "shard": 1`; the first search has `my-test-1` at #1 (in sync mode it is
searchable as soon as the upsert returns); the delete is applied and the doc is gone from the second search; the last
upsert says `"applied": false, "reason": "stale_or_duplicate", "current_version": 2`, because version 1 is older than
the delete. Writes are stored in each shard's WAL under `artifacts/docker/`, so they survive restarts. On a second try
use a new `doc_id` or a higher version.

### 3d. Model migration: migrate, cutover, rollback, finalize

```bash
curl -s -X POST $U/admin/cutover  -H "$J" -d '{"model":"bge-small"}'; echo    # refused: 409, not ready yet
curl -s -X POST $U/admin/migrate  -H "$J" -d '{"model":"bge-small"}'; echo    # every shard starts building
curl -s $U/admin/status | python3 -m json.tool | grep -A3 '"bge-small"'          # repeat until every shard says "ready": true
curl -s -X POST $U/admin/cutover  -H "$J" -d '{"model":"bge-small"}'; echo    # queries now use bge-small
curl -s -X POST $U/search -H "$J" -d '{"query":"what is a roth ira?"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])"
curl -s -X POST $U/admin/rollback -H "$J" -d '{}'; echo                       # back to minilm, instant
curl -s -X POST $U/admin/finalize -H "$J" -d '{"model":"bge-small"}'; echo    # drop the bge-small index
```

The build took ~1 s because it reads the cached bge-small vectors from `artifacts/`. Queries keep working the whole
time and keep using minilm until the cutover. Always end on minilm, as here: that is what the eval numbers assume.
`python scripts/migration.py migrate --model bge-small --wait` (then `cutover`, `rollback`, `finalize --model
bge-small`) does the same, and `python scripts/query_load.py --duration 120` in a second window shows no query fails
during it.

### 3e. Agents through the router

```bash
python scripts/ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy single --backend azure --confirm-spend --url $U
python scripts/agent_eval.py --backend azure --confirm-spend --systems single,multi --sets compound --limit 10 --search service --url $U --out toy_runs/manual/agents_docker.json
```

The agents' searches now go to the cluster. Quality is the same as in section 2; latency adds the HTTP hops.

### 3f. A scenario against the cluster

```bash
python scripts/scenario.py toy_runs/my_scenario.csv --url $U --open
```

This runs against the live Docker stack. My template run: 22 ops, 0 errors, wait_visible 78 to 95 ms (the doc was there on the
first poll, so that is one search round trip). It deletes the docs it wrote and ends back on minilm, so the index is as it was. `kill_shard` and `restart_shard` only work with `--start-cluster`; for Docker, see
the next step.

The simple scenarios run against Docker the same way:

```bash
python scripts/scenario.py scenarios/SimpleScenarios/2_SimpleReadAfterWrite.csv --url $U --open
```

Two things differ from `--start-cluster`. Their writes stay in Docker (the doc ids contain the run id, so a rerun never
collides), and scenarios 4, 5 and 6 end with bge-small active. Put the stack back on minilm afterwards, as at the end
of 3d:

```bash
curl -s -X POST $U/admin/rollback -H "$J" -d '{}'; echo
curl -s -X POST $U/admin/finalize -H "$J" -d '{"model":"bge-small"}'; echo
```

Scenario 7 ends with a rollback, so it only needs the finalize.

### 3g. Lose a shard

```bash
docker compose stop shard1
curl -s -X POST $U/search -H "$J" -d '{"query":"what is a roth ira?","k":3}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['partial'], r['missing_shards'])"
docker compose start shard1      # healthy again after ~15 s (it replays its WAL)
```

With shard 1 down, searches still answer (`True [1]`, ~1.2 s because the router waits for the shard's timeout), and a
write to a doc that shard 1 owns fails with HTTP 503 instead of being lost. After the restart `partial` is `False`
again.

### 3h. The automated Docker checks

```bash
python -m scripts.docker_trace --failover        # proves each request crossed containers; stops and starts a shard
python -m scripts.docker_check --mode both --keep # 38 checks across sync and async; --keep leaves the stack up
```

`docker_check` restarts the stack in each mode from a clean state: it deletes `artifacts/docker/` (the shard WALs and
the router state), so writes from section 3c are gone afterwards. Without `--keep` it stops the stack at the end.
`--eval` also runs the dev eval container (which writes into `results/runs/`).

### 3i. Async write mode and shutdown

```bash
./scripts/bootstrap.sh --mode async    # same stack; writes ack after the WAL and apply within 1 s
./scripts/bootstrap.sh                 # back to sync
./scripts/bootstrap.sh --down          # stop; data, caches and WALs stay
docker compose logs -f router shard0   # when something looks wrong
```

In async mode the upsert reply says `"visible": false, "mode": "async"`, and a search sent right after it may miss the
doc for a few ms. A scenario with `wait_visible` measures that gap; [scenarios.md](scenarios.md#freshness-what-the-numbers-mean)
has my numbers.

## 4. Every API from the browser

The router serves an API page (Swagger UI) at http://localhost:8000/docs. It lists all 10 endpoints, grouped as
search, writes, model migration and status, each with ready-made examples: open an endpoint, pick one from the
"Examples" dropdown (or edit the JSON), press **Execute**, and the response, its status code and the matching `curl`
command appear below. The top of the page has a short tour in the order I would try things: status, the three search
strategies, upsert, a stale write, delete, then migrate, cutover, rollback and finalize.

With Docker it is there as soon as the stack is up (section 1a). Without Docker:

```bash
python -m scripts.api_docs --local              # 1 shard + the router on :8000, real corpus (under a minute to start)
python -m scripts.api_docs --local --shards 3   # 3 shard processes, like src.service.cluster
python -m scripts.api_docs --local --port 8600  # another port, e.g. while Docker holds 8000
```

Ctrl+C stops it. Writes are kept in `artifacts/cluster<N>/` (`--fresh` starts empty). The examples come from
`scripts/api_docs.py`, which serves the router's own app with the examples added, so the requests behave exactly as
they do from `curl` or the scripts. The page loads Swagger UI from cdn.jsdelivr.net, so the browser needs internet
access. As in 3d, end a migration on minilm.
