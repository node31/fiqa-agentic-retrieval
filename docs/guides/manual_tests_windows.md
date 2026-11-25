# Manual tests on Windows

Step by step: bring the system up, try every retrieval mode in one process, then the same things against the sharded
Docker cluster. Every command runs in PowerShell 7 from the repo root. The macOS version is
[manual_tests_mac.md](manual_tests_mac.md). Security, A2A and agent memory have their own hand-run guide,
[demo_security_a2a_memory.md](demo_security_a2a_memory.md) (written for zsh; it also runs in Git Bash).

Next to each step is what I got on my laptop, so a result far from it means something is wrong. Everything that writes
a file writes under `toy_runs\manual\` (gitignored), never over the committed numbers in `results\`.

## 0. What you need

| | why |
|---|---|
| Docker Desktop, running | the sharded cluster (section 3) |
| Python 3.11+ (`py -3.13` works) | the host tools: `ask.py`, the evals, scenarios |
| ~5 GB free disk | data (~65 MB), embedding caches (~170 MB), the image (~2.8 GB), the venv (~1.1 GB) |
| an Azure OpenAI key (optional) | only for the agent steps; everything else is free |

If `docker` is not found, Docker Desktop's CLI may not be on PATH:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
```

## 1. Bring the system up

### 1a. The Docker cluster (router + 3 shards)

```powershell
.\scripts\bootstrap.ps1
```

It checks prerequisites, creates `.env` from `.env.example`, builds the image, downloads FiQA and the embedding caches
into `data\` and `artifacts\`, starts the 4 containers and runs a 6-check smoke test inside a container. It took ~509 s
on a fresh Windows box; a re-run skips what is already there. `-Mode async` starts the async write mode, `-Down` stops
everything and keeps the data. More options are in [DEPLOY.md](../../DEPLOY.md). After a `git pull` that changed anything under `src/`, run it
again: it rebuilds the image and recreates all four containers on it, and keeps the data, the WALs and the active
model. (Rebuilding only the router leaves the shards on the old image, which the Docker tour's shape check reports.)

Check it:

```powershell
docker compose ps                                       # router, shard0, shard1, shard2: all "healthy"
Invoke-RestMethod http://localhost:8000/admin/status    # active_model minilm (or minilm-ft, see section 3), 3 shards, ~19.2K docs each
start http://localhost:8000/docs                         # every API in the browser, with examples (section 4)
```

### 1b. Host Python (for sections 2 and 3)

The host tools reuse the `data\` and `artifacts\` that bootstrap downloaded, so there is nothing to rebuild:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch, avoids ~2 GB of CUDA
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Every new PowerShell window needs these two lines (check marks in the output, and models from the local cache):

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'      # leave it out the very first time, so MiniLM downloads once
```

### 1c. The LLM key (only for the agent steps)

Open `.env` and fill in `AZURE_OPENAI_ENDPOINT` (it must end with `/openai/v1/`), `AZURE_OPENAI_DEPLOYMENT` (the
deployment name you chose in Foundry; mine is gpt-6-sol) and `AZURE_OPENAI_API_KEY`. `.env` is gitignored. Without a
key, use `--backend copilot` (free, through the GitHub Copilot CLI, ~40 s per call) or `--backend fake` (scripted,
only shows the plumbing).

Then check the settings before any agent step:

```powershell
.\.venv\Scripts\python scripts\llm_check.py                   # where each setting came from, and what is wrong with it; no call
.\.venv\Scripts\python scripts\llm_check.py --confirm-spend   # one tiny real call, never from the cache (a fraction of a cent)
```

The second one ends with a line like `OK: gpt-6-sol answered 'OK' in 3.4 s, 27 input + 5 output tokens`. Three things
can make a run look as if Azure was never called, and this check or the agents' own output now shows each one:

| what you see | why | what to do |
|---|---|---|
| `[from your shell (it overrides .env)]` next to a setting | `.env` is loaded with `override=False`, so a variable already set in the environment wins | `Remove-Item Env:AZURE_OPENAI_API_KEY` (and the other one it names), or remove it from your user environment variables |
| `N answered from the disk cache (.llm_cache/azure/, no request)` at the end of a run | every answer is cached by prompt, so asking the same thing again sends nothing and costs nothing now | expected; delete `.llm_cache\azure\` to force fresh calls |
| `LLM calls 0` with `--strategy multi` | a short, one-clause question passes the multi-agent's free gate: it is searched as asked, with no planner and no verifier | expected; ask a two-topic question, or use `--strategy single` |

A wrong key, endpoint or deployment no longer hides either: `agent_eval.py` and the A2A server make the same tiny
call first (`preflight: ...`), and `agent_eval.py` stops with the error and its likely cause instead of serving every
question from the no-LLM baseline.

`UnicodeEncodeError: 'ascii' codec can't encode characters in position N-M` means the key (or another Azure setting)
contains characters that are not ASCII, usually an invisible space picked up by copy and paste. The key travels in an
HTTP header, which can only carry ASCII, so nothing is sent. `llm_check.py` names the setting, the positions and the
characters, and the A2A server refuses to start with the same line. Characters at the ends are trimmed for you;
re-type or re-paste a key that has them inside.

## 2. Single process (no Docker, no HTTP)

Here the whole corpus lives in one Python process (`single_node_router()`): one shard, no network. It is the baseline
the sharded cluster has to match.

### 2a. BM25, dense and hybrid on your own question

```powershell
.\.venv\Scripts\python scripts\ask.py "what is a roth ira?" --strategy bm25
.\.venv\Scripts\python scripts\ask.py "what is a roth ira?" --strategy dense
.\.venv\Scripts\python scripts\ask.py "what is a roth ira?" --strategy hybrid
```

Each run loads the corpus, the indexes and the encoder first (~30 s), then prints the top 10 with scores and a snippet.
On "what is a roth ira?" hybrid's #1 is doc 553031 ("A Roth IRA isn't an investment, per se...") in ~25 ms. `-k 3`
prints fewer hits.

### 2b. BM25, dense and hybrid on the dev set (500 questions)

```powershell
.\.venv\Scripts\python -m src.eval --mode local --strategy bm25   --name my-bm25   --out toy_runs\manual\bm25_dev.json
.\.venv\Scripts\python -m src.eval --mode local --strategy dense  --name my-dense  --out toy_runs\manual\dense_dev.json
.\.venv\Scripts\python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.7 --name my-hybrid --out toy_runs\manual\hybrid_dev.json
```

| strategy | nDCG@10 | R@10 | warm p50 |
|---|---|---|---|
| bm25 | 0.2642 | 0.3135 | ~1 ms |
| dense (MiniLM) | 0.3881 | 0.4665 | ~19 ms |
| hybrid (linear, α 0.7) | 0.4190 | 0.4914 | ~21 ms |

Hybrid needs `--fusion linear --alpha 0.7`: the eval's own defaults are RRF and α 0.5, while the service ships linear
0.7 (tuned on dev, [16_ablations.md](../topics/16_ablations.md)). `--limit 50` gives a quick partial run. Never pass
`--split test`: the test split was used once, for the final number.

### 2c. Single agent and multi-agent on your own question

```powershell
.\.venv\Scripts\python scripts\ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy single --backend azure --confirm-spend
.\.venv\Scripts\python scripts\ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy multi  --backend azure --confirm-spend
```

The single agent prints each step: step 0 searches the question as asked (no LLM call), then the model either asks for
another search (with its query) or finishes. The multi-agent prints the plan (the free gate's "simple", the planner's
"simple", or the sub-queries it split into) and any verifier rewrites. Both end with the top 10, the LLM calls, tokens
and the estimated cost, and a line saying how many requests really went to Azure and how many answers came from the
disk cache. A compound question costs about $0.001 to $0.004. The single agent's last lines, on my first run:

```text
LLM calls 2, tokens 1906 in + 158 out, estimated $0.00254 (placeholder prices in src/config.py)
LLM azure (gpt-6-sol): 2 request(s) sent to the service, 0 answered from the disk cache (.llm_cache/azure/, no request)
```

Run it again and the second line says `0 request(s) sent ..., 2 answered from the disk cache`: same answer, no cost.
`--strategy multi` on a simple question such as "what is an index fund?" makes no LLM call at all and says so (see the
table in 1c).

### 2d. Single agent and multi-agent on the compound set

`data\compound_dev.jsonl` has 100 two-topic questions built from pairs of dev questions (`data\build_compound.py`,
seed 42). Start with 10:

```powershell
.\.venv\Scripts\python scripts\agent_eval.py --backend azure --confirm-spend --systems single --sets compound --limit 10 --out toy_runs\manual\single_compound.json
.\.venv\Scripts\python scripts\agent_eval.py --backend azure --confirm-spend --systems multi  --sets compound --limit 10 --out toy_runs\manual\multi_compound.json
```

The first line is the preflight, one tiny uncached call (`preflight: OK: gpt-6-sol answered 'OK' ...`); if it fails
the run stops there with the reason. The last lines count the requests sent and the answers taken from the cache. If a
question crashes anyway, the run serves it from the no-LLM baseline and prints `! N of 10 queries crashed ...` with
the first error, so a `tok=0` row always comes with a reason.

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

```powershell
New-Item -ItemType Directory -Force toy_runs | Out-Null
Copy-Item scenarios\SimpleScenarios\simple_template.csv toy_runs\my_scenario.csv
notepad toy_runs\my_scenario.csv
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

Run it on a fresh local cluster and open the replay when it finishes:

```powershell
.\.venv\Scripts\python scripts\scenario.py toy_runs\my_scenario.csv --start-cluster --tiny --open             # 24-doc toy corpus, ~10 s
.\.venv\Scripts\python scripts\scenario.py toy_runs\my_scenario.csv --start-cluster --shards 1 --open          # real corpus, one shard
.\.venv\Scripts\python scripts\scenario.py toy_runs\my_scenario.csv --start-cluster --mode async --open        # real corpus, 3 shards, async writes
```

`--start-cluster` starts its own router and shards as local processes on ports 19000 and 19101+, so it never touches
Docker. The run prints one line per op and a summary ending in `RESULT PASS`, and writes
`toy_runs\scenarios\<name>-<mode>-<time>\replay.html` (the path is printed at the end). To open it later:
`start toy_runs\scenarios\<run>\replay.html`. The template passes with 22 ops and 0 errors in both modes.

In the replay, press Space to play. Under the narration there is a filter bar: click persona chips (say fox and bot1)
to see only their cards in both lanes, or type a doc id or query text (`bot1-1`, `roth`) to see only the cards that
mention it. Replays recorded before the filters existed get them with
`.\.venv\Scripts\python scripts\scenario.py --replay-only toy_runs\scenarios\<run>\events.jsonl`; nothing is re-run.

The full format, every operation and the JSONL format for parallel lanes are in [scenarios.md](scenarios.md).
`--to-jsonl` prints what a CSV turns into.

### 2f. Run the simple scenarios

`scenarios\SimpleScenarios\` has eight short stories in the CSV format, each building on the one before: two users
searching, a write then a search for it, a delete, a model migration, a migration with a write before it, one with a
write during it, a rollback, and retries and stale writes (idempotency). Each file's header says what it shows and what
to look for; [scenarios/SimpleScenarios/README.md](../../scenarios/SimpleScenarios/README.md) has the overview.

One scenario, one cluster, replay opened at the end:

```powershell
.\.venv\Scripts\python scripts\scenario.py scenarios\SimpleScenarios\3_SimpleReadAfterWriteAndDelete.csv --start-cluster --tiny --open               # tiny, sync
.\.venv\Scripts\python scripts\scenario.py scenarios\SimpleScenarios\3_SimpleReadAfterWriteAndDelete.csv --start-cluster --tiny --mode async --open  # tiny, async
.\.venv\Scripts\python scripts\scenario.py scenarios\SimpleScenarios\3_SimpleReadAfterWriteAndDelete.csv --start-cluster --open                      # real corpus, sync
.\.venv\Scripts\python scripts\scenario.py scenarios\SimpleScenarios\3_SimpleReadAfterWriteAndDelete.csv --start-cluster --mode async --open         # real corpus, async
```

The tiny runs take ~13 s each (mostly starting the cluster), the real-corpus ones ~50 s. Each starts its own fresh
3-shard cluster on ports 19000/19101+ and stops it afterwards, so runs never see each other's writes and Docker is not
touched. The replay goes to `toy_runs\scenarios\<name>-<mode>-<time>\replay.html`.

All eight on all four clusters (SimpleSyncTinyCluster, SimpleAsyncTinyCluster, SimpleSyncFullCluster,
SimpleAsyncFullCluster), ~17 min:

```powershell
.\.venv\Scripts\python scripts\run_simple_scenarios.py
.\.venv\Scripts\python scripts\run_simple_scenarios.py --configs SimpleSyncTinyCluster --scenarios 4 7   # a subset
start scenarios\SimpleScenarios\results\SimpleAsyncFullCluster\2_SimpleReadAfterWrite\replay.html
```

The results go to `scenarios\SimpleScenarios\results\<cluster>\<scenario>\` (console.txt, events.jsonl, replay.html),
with a grid of every run in `results\SUMMARY.md`. My runs are checked in (console.txt, events.jsonl and replay.html for each,
plus SUMMARY.md): all 32 pass, the migrated searches are answered by bge-small and the rolled-back ones by
minilm, and scenario 8's retry and late writes are no-ops.

## 3. The sharded cluster (Docker)

Same tests, now through the router at `http://localhost:8000`: every search fans out to 3 shard containers, every
write goes to the one shard that owns the doc.

Two things about the running stack change what you see here:

* **Which model is active.** The expected ids and numbers below are for `minilm`, the bootstrap default. If you
  switched the stack to the fine-tuned model (README, "Switching the running stack to the fine-tuned model"),
  `/admin/status` says `active_model: minilm-ft`, and expect instead: hybrid top 3 for "what is a roth ira?"
  553031, 214174, 179408; dev hybrid nDCG@10 0.4469 and R@10 0.5141 (p50 ~33 ms). The writes, migration, scenario
  and shard steps behave the same on either model.
* **The security toggles.** They are all off after bootstrap. If you ran
  [demo_security_a2a_memory.md](demo_security_a2a_memory.md) and stopped halfway, some may still be on, and the
  token-less calls below would get 401 or 403. Check, and turn them all off:

```powershell
.\.venv\Scripts\python -m scripts.demo_call GET /admin/security --as admin --pick toggles_on
.\.venv\Scripts\python -m scripts.demo_call alloff
```

The first should print `= toggles_on: []`.

> **All of section 3 in one command.** `.\scripts\docker_tour.ps1` runs every step below except 3h and 3i against the
> running stack, checks each result against the numbers in this section, and writes a report with an animated replay
> per step (`toy_runs\docker_tour\<stamp>\index.html`, opened at the end). It stops and starts shard1, then deletes the docs
> it wrote and verifies the stack is back in exactly the shape it found it: containers, write mode, models, live docs
> per shard and the answers to six probe searches. About 2 to 3 minutes; needs only Docker and the stack in sync mode
> (the bootstrap default). 3i stays manual: the tour never changes the write mode. `-Azure` adds the agent step
> (3e, costs a few cents), `-Skip eval,simple` or `-Only writes,shard` pick phases (search, eval, writes, migration, simple,
> agents, shard), and `-NoOpen` skips opening the report.

### 3a. Search with each strategy over raw HTTP

```powershell
$u = 'http://localhost:8000'
foreach ($s in 'bm25','dense','hybrid') {
  $r = Invoke-RestMethod -Method Post "$u/search" -ContentType application/json -Body (@{query='what is a roth ira?'; k=3; strategy=$s} | ConvertTo-Json)
  "$s  partial=$($r.partial)  model=$($r.model)  ids=$($r.results.doc_id -join ', ')"
}
```

Expect `partial=False` and the same ids as the single process (hybrid with minilm: 553031, 488556, 214174; with minilm-ft: 553031, 214174, 179408). Add `include_text=$true`
to the body for passage texts. The body also takes `fusion` (`linear`/`rrf`), `alpha` and `rerank_k`. Each response
has `timings_ms`, the router's split of where the time went.

The same through the tools:

```powershell
.\.venv\Scripts\python scripts\ask.py "what is a roth ira?" --strategy hybrid --url $u
.\.venv\Scripts\python src\search.py --url $u --query "what is short selling?" --top-k 10 --strategy hybrid
```

### 3b. The dev set through the router

```powershell
.\.venv\Scripts\python -m src.eval --mode service --url $u --strategy hybrid --fusion linear --alpha 0.7 --name my-docker-hybrid --out toy_runs\manual\docker_hybrid_dev.json
```

Expect nDCG@10 0.4190 and R@10 0.4914 with minilm (0.4469 and 0.5141 with minilm-ft), exactly the single-process numbers (global BM25 statistics + a global merge make
3 shards rank like 1). Latency is higher: p50 ~30 ms in my reference run, against ~21 ms in one process. Swap
`--strategy bm25` or `dense` for the other two. `docker compose --profile tools run --rm eval` does the hybrid run
inside a container, but it writes `results\runs\docker-hybrid-linear-minilm.json`.

### 3c. Writes: upsert, search, delete

```powershell
$j = 'application/json'
Invoke-RestMethod -Method Post "$u/upsert" -ContentType $j -Body '{"doc_id":"my-test-1","version":1,"text":"zebra bonds are a made up product for my manual test"}'
Invoke-RestMethod -Method Post "$u/search" -ContentType $j -Body '{"query":"zebra bonds manual test","k":3,"strategy":"bm25"}' | % results
Invoke-RestMethod -Method Post "$u/delete" -ContentType $j -Body '{"doc_id":"my-test-1","version":2}'
Invoke-RestMethod -Method Post "$u/search" -ContentType $j -Body '{"query":"zebra bonds manual test","k":3,"strategy":"bm25"}' | % results
Invoke-RestMethod -Method Post "$u/upsert" -ContentType $j -Body '{"doc_id":"my-test-1","version":1,"text":"zebra again"}'
```

What I got: the upsert says `applied True, shard 1`; the first search has `my-test-1` at #1 (in sync mode it is
searchable as soon as the upsert returns); the delete is applied and the doc is gone from the second search; the last
upsert says `applied False, reason stale_or_duplicate, current_version 2`, because version 1 is older than the delete.
Writes are stored in each shard's WAL under `artifacts\docker\`, so they survive restarts. On a second try use a new
`doc_id` or a higher version.

### 3d. Model migration: migrate, cutover, rollback, finalize

```powershell
Invoke-RestMethod -Method Post "$u/admin/cutover"  -ContentType $j -Body '{"model":"bge-small"}'   # refused: 409, not ready yet
Invoke-RestMethod -Method Post "$u/admin/migrate"  -ContentType $j -Body '{"model":"bge-small"}'   # every shard starts building
(Invoke-RestMethod "$u/admin/status").shards.'0'.models                                            # repeat until bge-small ready: True
Invoke-RestMethod -Method Post "$u/admin/cutover"  -ContentType $j -Body '{"model":"bge-small"}'   # queries now use bge-small
(Invoke-RestMethod -Method Post "$u/search" -ContentType $j -Body '{"query":"what is a roth ira?"}').model   # bge-small
Invoke-RestMethod -Method Post "$u/admin/rollback" -ContentType $j -Body '{}'                       # back to the model you started on, instant
Invoke-RestMethod -Method Post "$u/admin/finalize" -ContentType $j -Body '{"model":"bge-small"}'   # drop the bge-small index
```

The build took ~1 s because it reads the cached bge-small vectors from `artifacts\`. Queries keep working the whole
time and keep using the old model until the cutover. Always end on the model you started on, as here: that is what
the eval numbers assume.
`.\.venv\Scripts\python scripts\migration.py migrate --model bge-small --wait` (then `cutover`, `rollback`,
`finalize --model bge-small`) does the same, and `scripts\query_load.py --duration 120` in a second window shows no
query fails during it.

### 3e. Agents through the router

```powershell
.\.venv\Scripts\python scripts\ask.py "Can stock brokerage firms fail, and can I print my own checks?" --strategy single --backend azure --confirm-spend --url $u
.\.venv\Scripts\python scripts\agent_eval.py --backend azure --confirm-spend --systems single,multi --sets compound --limit 10 --search service --url $u --out toy_runs\manual\agents_docker.json
```

The agents' searches now go to the cluster. Quality is the same as in section 2; latency adds the HTTP hops. Both
print the same "requests sent / answered from the disk cache" line as in section 2, and `agent_eval.py` starts with
the preflight call.

### 3f. A scenario against the cluster

```powershell
.\.venv\Scripts\python scripts\scenario.py toy_runs\my_scenario.csv --url $u --open
```

This runs against the live Docker stack. My template run: 22 ops, 0 errors, wait_visible 78 to 95 ms (the doc was there on the
first poll, so that is one search round trip). It deletes the docs it wrote and ends back on the model it started on, so the index is as it was. `kill_shard` and `restart_shard` only work with `--start-cluster`; for Docker, see
the next step.

The simple scenarios run against Docker the same way:

```powershell
.\.venv\Scripts\python scripts\scenario.py scenarios\SimpleScenarios\2_SimpleReadAfterWrite.csv --url $u --open
```

Two things differ from `--start-cluster`. Their writes stay in Docker (the doc ids contain the run id, so a rerun never
collides), and scenarios 4, 5 and 6 end with bge-small active. Put the stack back on the model you started on afterwards, as at the end
of 3d:

```powershell
Invoke-RestMethod -Method Post "$u/admin/rollback" -ContentType $j -Body '{}'
Invoke-RestMethod -Method Post "$u/admin/finalize" -ContentType $j -Body '{"model":"bge-small"}'
```

Scenario 7 ends with a rollback, so it only needs the finalize.

### 3g. Lose a shard

```powershell
docker compose stop shard1
Invoke-RestMethod -Method Post "$u/search" -ContentType $j -Body '{"query":"what is a roth ira?","k":3}' | Select partial, missing_shards
docker compose start shard1      # healthy again after ~15 s (it replays its WAL)
```

With shard 1 down, searches still answer (`partial True, missing_shards 1`, ~1.2 s because the router waits for the
shard's timeout), and a write to a doc that shard 1 owns fails with HTTP 503 instead of being lost. After the restart
`partial` is `False` again.

### 3h. The automated Docker checks

```powershell
.\.venv\Scripts\python -m scripts.docker_trace --failover        # proves each request crossed containers; stops and starts a shard
.\.venv\Scripts\python -m scripts.docker_check --mode both --keep # 38 checks across sync and async; --keep leaves the stack up
```

`docker_check` restarts the stack in each mode from a clean state: it deletes `artifacts\docker\` (the shard WALs and
the router state), so writes from section 3c are gone afterwards. Without `--keep` it stops the stack at the end.
`--eval` also runs the dev eval container (which writes into `results\runs\`).

### 3i. Async write mode and shutdown

```powershell
.\scripts\bootstrap.ps1 -Mode async    # same stack; writes ack after the WAL and apply within 1 s
.\scripts\bootstrap.ps1                # back to sync
.\scripts\bootstrap.ps1 -Down          # stop; data, caches and WALs stay
docker compose logs -f router shard0   # when something looks wrong
```

In async mode the upsert reply says `visible: False, mode: async`, and a search sent right after it may miss the doc
for a few ms. A scenario with `wait_visible` measures that gap; [scenarios.md](scenarios.md#freshness-what-the-numbers-mean)
has my numbers.

## 4. Every API from the browser

The router serves an API page (Swagger UI) at http://localhost:8000/docs. It lists all 10 endpoints, grouped as
search, writes, model migration and status, each with ready-made examples: open an endpoint, pick one from the
"Examples" dropdown (or edit the JSON), press **Execute**, and the response, its status code and the matching `curl`
command appear below. The top of the page has a short tour in the order I would try things: status, the three search
strategies, upsert, a stale write, delete, then migrate, cutover, rollback and finalize.

With Docker it is there as soon as the stack is up (section 1a). Without Docker:

```powershell
.\.venv\Scripts\python -m scripts.api_docs --local              # 1 shard + the router on :8000, real corpus (under a minute to start)
.\.venv\Scripts\python -m scripts.api_docs --local --shards 3   # 3 shard processes, like src.service.cluster
.\.venv\Scripts\python -m scripts.api_docs --local --port 8600  # another port, e.g. while Docker holds 8000
```

Ctrl+C stops it. Writes are kept in `artifacts/cluster<N>/` (`--fresh` starts empty). The examples come from
`scripts/api_docs.py`, which serves the router's own app with the examples added, so the requests behave exactly as
they do from `curl` or the scripts. The page loads Swagger UI from cdn.jsdelivr.net, so the browser needs internet
access. As in 3d, end a migration on the model you started on.

## 5. Security, A2A and agent memory

These have their own guide, [demo_security_a2a_memory.md](demo_security_a2a_memory.md), written for zsh and bash; in
Windows run it in Git Bash (its setup block sets `MSYS_NO_PATHCONV=1` for that). Every call goes through
`python -m scripts.demo_call`, which prints each request and response in colour. Its Part D runs the security cases
with a real LLM answer; run `.\.venv\Scripts\python scripts\llm_check.py --confirm-spend` first (1c), and start the A2A agent with
`--backend azure --confirm-spend`, which prints its own `azure preflight: ...` line at start-up.
