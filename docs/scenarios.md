# Scenarios: scripting the live service, checking it, replaying it

`scripts/scenario.py` plays a small file of operations (writes, searches, waits, admin actions, shard crashes) against
the running distributed service over real HTTP. For every operation it prints whether it worked and whether the
expectations held, and it records everything so I can replay the run as an animated HTML page.

I built it because the service makes promises about time and order that are easy to state and hard to see:

| promise | where it is implemented | scenario that shows it |
|---|---|---|
| a write is searchable within 1 s of its ack (sync: at once) | `store.py` (sync), `async_apply.py` (async) | `freshness_demo`, `03` |
| retries and late, older versions are harmless no-ops | `store._is_newer`, `AsyncApplier.accepted` | `idempotency` |
| 3 shards answer exactly like 1 node | `router.py` DFS phase + global merge | `01` + `02` (golden) |
| a model migration never fails a query; answers change only at cutover; rollback restores them | `router.py` MODEL MIGRATION, `store.start_build` | `04`, `05` |
| a dead shard gives partial answers, not errors; a restart loses no acked write | `router.py` partial results, `wal.py` | `06`, `07` |

The pieces: `scripts/scenario.py` (runner), `scripts/replay_html.py` (replay page generator),
`scenarios/ComplexScenarios/*.jsonl` (the catalog), `scenarios/SimpleScenarios/*.csv` (eight small stories in the simple
format, plus `simple_template.csv` to copy and edit), `scripts/run_simple_scenarios.py` (runs the eight on four
clusters) and `tests/test_scenario.py` (runs them in both modes).

## Quick start

```powershell
# fastest: tiny 24-doc corpus + hash "embedder", a fresh 3-shard cluster on ports 19000/19101+ (starts in ~10 s)
python scripts/scenario.py scenarios/ComplexScenarios/freshness_demo.jsonl --start-cluster --tiny --mode sync
python scripts/scenario.py scenarios/ComplexScenarios/freshness_demo.jsonl --start-cluster --tiny --mode async

# the real FiQA corpus (57,638 docs) + MiniLM: same command without --tiny (startup ~1-2 min)
python scripts/scenario.py scenarios/ComplexScenarios/freshness_demo.jsonl --start-cluster --mode async

# against a cluster you already run (python -m src.service.cluster, docker compose, ...)
python scripts/scenario.py scenarios/ComplexScenarios/freshness_demo.jsonl --url http://127.0.0.1:8000

# the simple CSV format, opening the replay in the browser when the run ends
python scripts/scenario.py scenarios/SimpleScenarios/simple_template.csv --start-cluster --tiny --open

# re-render the replay page from a recorded log (also how old runs get the newest page features)
python scripts/scenario.py --replay-only toy_runs/scenarios/<run>/events.jsonl
```

On Windows I set `$env:PYTHONIOENCODING='utf-8'` (check marks) and `$env:HF_HUB_OFFLINE='1'` (real corpus, models from
the local cache). The exit code is 0 if every hard expectation passed, 1 if not, and 2 for a scenario file error, so a
scenario is also a test.

Each run writes to `toy_runs/scenarios/<scenario>-<mode>-<timestamp>/` (gitignored):

| file | what |
|---|---|
| `events.jsonl` | every event: op start/end, status samples, summary (one JSON per line) |
| `replay.html` | the animated replay, one self-contained file that works offline |
| `cluster_logs/` | shard and router logs, only with `--start-cluster` (where I look when something fails) |

With `--start-cluster` the WALs and state live in `artifacts/scenario_runs/<run>/` and are deleted after the run
(`--keep-state` keeps them). The default ports are 19000 (router) and 19101+ (shards), so they never collide with the
dev cluster's 8000/8101+ or the tests' 18000-18999.

## Simple format (CSV)

For a quick story I don't want to write JSON. A `.csv` file is read in the simple format, one op per line:

```text
actor, operation, text, t_ms[, doc_id[, version]]
```

```text
fox, search, "roth ira", 10
elephant, search, "investments in mortgage", 20
ADMIN, migrate, , 30
bot1, upsert, "roth investments for IRA are as follows", 40
```

Blank lines and `#` comments are allowed, and so is a header row starting with `actor`. Text that contains a comma goes
in double quotes. Lines run in file order. `t_ms` means "do not start this line before t ms after the run started", so
an op starts at the later of `t_ms` and the end of the line before it. An empty `t_ms` means "right after the line
before". A `t_ms` smaller than the one before is allowed but gets a warning, because it can only run late.

Column 3 depends on the operation:

| operation | column 3 | column 5 (`doc_id`) |
|---|---|---|
| `search` (`search:bm25`, `search:dense`; plain `search` is hybrid) | the query (required) | optional: `doc` = a doc I expect in the answer, `!doc` = a doc I expect to be gone (after a delete). Recorded as a soft check, so a miss is reported but never fails the run |
| `upsert` (alias `update`, `write`) | the document text (required) | optional: empty gives `<actor>-<n>-{run}` (`bot1-1-a3f9c2`) |
| `delete` (alias `remove`) | empty | optional: empty = this actor's last upserted doc, else the last doc anyone upserted |
| `wait_visible` / `wait_gone` | the query; empty = the doc's upserted text | same default as `delete`. `wait_gone` waits until the doc is no longer found |
| `migrate`, `wait_ready`, `cutover` | the model; empty = `bge-small` | |
| `finalize` | the model to drop; empty = the one kept for rollback (after `cutover` + `rollback` that is the model I tried) | |
| `rollback`, `flush`, `status` | empty | |
| `kill_shard`, `restart_shard` | the shard number (needs `--start-cluster`) | |
| `sleep` | seconds | |
| `label` | the banner text (column 1 is ignored) | |

The role comes from the actor's name with the same rules as JSONL (`ADMIN`/`admin*`/`build*` are admin, `ops`/`sre`/
`chaos` are ops, `bot*`/`writer*`/`load*` are bots, any other name is a user persona such as fox). A name without a
hint gets its role from its first line. A persona that tries `migrate`, or ADMIN trying `search`, is an error that
names the line and says how to fix it; so is an unknown operation, a bad `t_ms`, or a `delete` with nothing upserted
before it.

Column 6, `version`, is only for `upsert` and `delete`: an explicit version, to write a retry (the same version
again), a late older write or a stale write after a delete (see `8_Idempotency.csv`). Empty means "the last version this
run used for that doc, plus 1", which is what a normal client does.

`scenarios/SimpleScenarios/simple_template.csv` uses every operation in an order that works end to end, and its header has the exact
commands for Windows and macOS. It passes on the tiny cluster in both modes (22 ops, 0 errors) and is part of the
tests. Two flags help with it: `--open` opens `replay.html` in the default browser when the run ends, and
`--to-jsonl` prints the JSON Lines a CSV becomes, without running anything (useful when I later want `parallel` or
`repeat`, which only JSONL has).

Under the hood each CSV line becomes one JSONL op with an `at` field (seconds since the run started). Any leaf op in a
JSONL file accepts `at` too.

## The file format

JSON Lines, one op per line. Blank lines and lines starting with `#` are comments. An op may span several lines (the
parser reads consecutive JSON values), which keeps `parallel` blocks readable. Unknown ops or fields are an error with
the line number, because a typo like `"expected"` would otherwise silently check nothing.

### Templating

| placeholder | value |
|---|---|
| `{run}` | a random 6-hex-char id per run. I put it in doc ids and distinctive tokens so re-runs against a long-lived cluster always write new docs (an old id with version 1 would be a no-op) |
| `{i}` (or the `var` you chose) | the `repeat` counter, or the current item of `each` |
| `{lane}` | the current lane name |

Unknown placeholders stay as they are. I use texts with unique tokens (`"zebraprobe{run} annuity"`) and query with the
same text: BM25 then finds the doc deterministically on the tiny and the real corpus, and so does dense search
(identical text gives an identical vector).

### Ops

| op | required | optional (default) | what it does |
|---|---|---|---|
| `config` | | `shards` (3), `mode` (sync), `models` (minilm), `tiny` (false), `description`, `actors` (see Actors) | must be the first op. Describes the cluster `--start-cluster` should start and who acts in the story. CLI flags win (`--shards`, `--mode`, `--models`, `--tiny`/`--real`). With `--url` it is only compared (warning if the shard count differs) |
| `upsert` | `doc_id`, `text` | `version` (auto: last version of this doc in this run + 1), `expect_applied`, `expect_error`, `soft` | POST /upsert |
| `delete` | `doc_id` | `version` (auto), `expect_applied`, `expect_error`, `soft` | POST /delete |
| `search` | `query` | `strategy` (hybrid), `k` (10), `fusion`, `alpha`, `save_as`, expectations, `soft` | POST /search |
| `sleep` | `seconds` | | wait |
| `wait_visible` | `query`, `doc_id` | `timeout` (2.0), `strategy` (hybrid), `k` (10), `absent` (false), `interval` (0.02) | poll search until the doc appears (or disappears with `absent`). Reports the lag from the doc's last ack in this run to the search that first saw it (an upper bound: it includes that search's round trip). Fails after `timeout` |
| `flush` | | `timeout` (10) | POST /admin/flush: wait until every acked write is applied (instant in sync mode) |
| `status` | | | GET /admin/status snapshot: docs, async queue, lag, models per shard |
| `label` | `text` | | narration: a banner in the console and the replay |
| `migrate` | `model` | `throttle_s` (0), `reencode` (false), `expect_error` | POST /admin/migrate: every shard starts building `model`'s index in the background; dual-write starts now; queries keep the active model |
| `wait_ready` | `model` | `timeout` (120), `interval` (0.1) | poll /admin/status until every shard reports `model` ready; each poll is a progress sample for the replay |
| `cutover` | `model` | `expect_error` | POST /admin/cutover: the router queries with `model` from now on (409 unless ready everywhere) |
| `rollback` | | `expect_error` | POST /admin/rollback: back to the previous model (instant) |
| `finalize` | | `model` (the one kept for rollback), `expect_error` | POST /admin/finalize: drop `model`'s index (no rollback to it afterwards) |
| `kill_shard` | `shard` | | hard-kill the shard process, like a crash. Needs `--start-cluster` |
| `restart_shard` | `shard` | `timeout` (300) | start it again and wait for /health (base corpus + WAL replay happen first). Needs `--start-cluster` |
| `repeat` | `ops` + one of `times` / `each` / `until_done` / `until_signal` | `var` (i), `start` (0) | `times: N` runs `ops` N times with `{i}` = start..; `each: [...]` runs once per item; `until_done: "lane"` loops until that lane finished; `until_signal: "name"` loops until some lane sent that signal (both run at least once; `times` then caps them) |
| `parallel` | `lanes` | | runs lanes concurrently, one thread + one keep-alive HTTP client each, and waits for all (fork/join). A lane is `{"name": "bot-1", "actor": "...", "description": "...", "ops": [...]}` (`actor` defaults to the name, `description` is shown in the replay) or a bare list (named `lane0`, `lane1`, ...). Nested lanes are named `parent.child` |
| `signal` | `name` | | coordination between lanes (no HTTP): marks the named event as happened |
| `wait_signal` | `name` | `timeout` (120) | block this lane until some lane sent `signal` `name` (fails after `timeout`). A name nobody signals is a file error |

Every op except `config`, `label`, `repeat` and `parallel` also accepts `"actor": "name"` and `"at": seconds` (do
not start before this many seconds after the run started). `finalize` without `model` drops the model kept for
rollback (the runner reads it from `/admin/status`). Every op records its own line
of the file (also inside a multi-line `parallel` block), shown on its console line and its replay card.

### Expectations

A hard check decides pass/fail. `"soft": true` on an op makes all its checks soft: they are recorded and shown in amber
but never fail the run.

| field (on) | passes when |
|---|---|
| `expect: [ids]` (search) | every id is in the results. Records the rank and `since_ack_ms`, the time this search was sent after the doc's last ack |
| `expect_absent: [ids]` (search) | no id is in the results |
| `expect_ok: true` (search) | no error, not partial, at least one hit |
| `expect_partial: true/false` (search) | `partial` has that value |
| `expect_missing_shards: [1]` (search) | exactly these shards were missing |
| `save_as: "q1"` (search) | not a check: remembers the ranked id list under that name |
| `expect_same_as: "q1"` (search) | the ranked list is identical (same ids, same order). On failure it reports the first differing rank, ids lost and ids gained |
| `expect_overlap_with: {"ref": "q1", "min": 0.6}` (search) | at least `min` of q1's ids are still in the results (order ignored) |
| `expect_applied: true/false` (write) | the shard applied the write, or treated it as a no-op (retry or older version) |
| `expect_error: true` (write, search, admin) | the call fails (HTTP error or no response), e.g. a cutover to an unbuilt model (409) or a write owned by a dead shard (503) |
| implicit, `wait_visible` / `wait_ready` / `flush` | the doc became visible, the model became ready, the queue drained in time |
| implicit, any op | an unexpected HTTP error fails the op |

Two times show up in the output, and I keep them apart. A search check like `FOUND #1 · searched 2.07 s after its ack`
says when that search ran relative to the ack (for example after a sleep or a load phase). `wait_visible` reports
`VISIBLE after 34 ms (first poll)`, the visibility lag from ack to the first search that saw the doc. That lag is the
freshness number, with a 1 s budget. A hard `expect` after `sleep 1.0` only proves the doc was visible by then.

When answers must stay identical and when they may change (scenarios 04 and 05 encode this):

| situation | hybrid / dense | bm25 | check with |
|---|---|---|---|
| repeated query, nothing changed | identical | identical | `expect_same_as` |
| 1 shard vs 3 shards | identical | identical | golden file |
| during a migration build (before cutover) | identical (queries still use the old model) | identical | `expect_same_as` |
| after cutover | may change (other embedding model) | identical (BM25 ignores the model) | `expect_overlap_with` + `expect_ok` for dense/hybrid, `expect_same_as` for bm25 |
| after rollback | identical to before the cutover | identical | `expect_same_as` |
| new docs written in between | may change, even for unrelated queries: BM25's global statistics (doc count, average length) change | may change | `expect_ok`, `expect` for the new docs |

### Golden files

`--golden-save path.json` writes every `save_as` answer of the run. `--golden-check path.json` compares this run's
`save_as` answers with the file (exact ranked lists, by name) and adds the result as hard checks (op `golden_check`, the
last row of the replay timeline). I use it for 1 shard vs 3 shards:

```powershell
python scripts/scenario.py scenarios/ComplexScenarios/01_single_node_read.jsonl --start-cluster --tiny --golden-save toy_runs/golden/read_tiny.json
python scripts/scenario.py scenarios/ComplexScenarios/02_multi_node_read.jsonl  --start-cluster --tiny --golden-check toy_runs/golden/read_tiny.json
```

A golden file only compares with the same corpus and model (tiny/hash and real/MiniLM give different answers); its
`meta` records where it came from. It stores only the `save_as` names and ranked ids, so renaming actors never
invalidates it.

### Actors

Every op is performed by an actor, and every actor has one of four roles. The replay puts clients on the left, the
cluster in the middle and privileged operators on the right.

| role | shown as | may run | typical names |
|---|---|---|---|
| `user` | an animal persona (fox, elephant, dog, owl, panda, koala, ...) | client ops: `upsert`, `delete`, `search`, `wait_visible` (+ neutral ops) | fox, elephant (any name) |
| `bot` | a robot | client ops (automated traffic: background writers/readers, synthetic probes) | bot-1, probe-bot, load-a, writer-2, reader |
| `admin` | Admin | `migrate`, `wait_ready`, `cutover`, `rollback`, `finalize`, `flush` (+ ops-manager ops, + neutral ops) | admin |
| `ops` | Ops manager | `kill_shard`, `restart_shard`, `status` (+ admin ops, + neutral ops) | ops, chaos, sre |

The split is strict both ways: clients never run admin/ops ops, and admins and ops managers never search or write. A
health probe that searches during a migration is therefore a bot (`probe-bot`). Neutral ops (`sleep`, `label`,
`signal`, `wait_signal`) are allowed for every role. A `label` or a top-level `sleep` belongs to nobody and is shown in
the middle.

I declare actors in the `config` line (the whole catalog does):

```json
{"op": "config", "shards": 3, "actors": {
  "fox":      {"role": "user",  "description": "writes docs"},
  "elephant": {"role": "user",  "description": "searches for them"},
  "bot-1":    {"role": "bot",   "description": "background writer"},
  "admin":    {"role": "admin", "description": "runs the migration"},
  "ops":      {"role": "ops",   "description": "kills and restarts a shard"}}}
```

`emoji` may override the picture. Then the actor is set at whichever level reads best:

| where | example | meaning |
|---|---|---|
| per op | `{"op": "status", "actor": "ops"}` | this op only, handy at the top level |
| per lane | `{"name": "canary", "actor": "probe-bot", "ops": [...]}` | every op of the lane and its nested lanes |
| lane name | `{"name": "bot-1", "ops": [...]}` | a lane's actor defaults to its name. I use one lane per actor, named like the actor, and lanes take turns with `signal` / `wait_signal`. A replay card names its lane only when it differs from the actor |
| top level | `{"op": "cutover", "model": "m"}` | client ops go to the first declared user (else bot, else an actor named `user`), admin-type ops to the first declared admin (else `admin`), ops-type ops to the first declared ops (else `ops`) |

Undeclared actors are inferred. First by name: `admin*`, `build*`, `migrat*`, `operator*` → admin; `ops*`, `chaos*`,
`sre*`, `oncall*` → ops; `bot*`, `load*`, `burst*`, `writer*`, `reader*`, `checker*`, `probe*`, `canary*`,
`watchdog*`, `monitor*` or the word `bot` anywhere → bot (a watchdog searches, so it is a client); an animal name →
user. Otherwise by what it does: any admin-type op → admin, any ops-type op → ops, else user (neutral ops don't count).
Users get the next free animal persona in order of first appearance; generic names (`user`, `main`, `lane2`) are
displayed as that animal.

Before anything runs, the runner rejects (exit code 2) a client actor running a privileged op (`migrate`, `wait_ready`,
`cutover`, `rollback`, `finalize`, `flush`, `kill_shard`, `restart_shard`, `status`) and an admin/ops actor running a
client op (`search`, `upsert`, `delete`, `wait_visible`). It never moves the op silently:

```text
scenario error: line 7: 'migrate' by actor 'fox' (role user, declared): admin ops must run in an
admin/ops actor. Add "actor": "admin" (or "ops") to the op, or run it in a lane whose actor has
role admin/ops in config.actors

scenario error: line 12: 'search' by actor 'ops' (role ops, declared): client ops
(search/upsert/delete/wait_visible) must run in a user/bot actor. Add "actor": "<a user or bot>" to
the op (e.g. a probe bot for health searches), or declare the lane's actor with role user/bot in
config.actors
```

Recordings from before actors existed use lane names (`main`, `admin`, `admin.build`, `admin.questions`, `watchdog`,
`chaos`, `burst-a/b/c`); the replay maps them with the same inference rules.

## Reading the console output

```text
[   0.555s] fox        UPSERT  fresh-5289e1 v1 -> shard 2 ... OK applied (ack 18.0 ms, sync: applied, searchable now)
                               expect_applied=True: got applied=True ✓
[   0.573s] elephant   SEARCH  "zebraprobe5289e1 annuity ladder freshne…" [hybrid k=10] -> 10 hits (12.7 ms, model minilm)
                               expected fresh-5289e1: FOUND (#1) (doc acked 0.555 s, search sent 0.573 s, answer back 0.586 s, difference 13 ms) ✓
[   2.646s] elephant   WAIT    fresh2-5289e1 VISIBLE after 11.6 ms (first poll) · freshness budget 1 s ✓
[   0.052s] admin      CUTOVER failed as expected: HTTP 409: refusing cutover: bge-small not ready on every shard
```

| part | meaning |
|---|---|
| `[   0.555s]` | when the op started, seconds since the run began |
| `fox` | the actor; `main` (or a lane name) for labels and fork/join lines |
| `-> shard 2` | the owner shard (`crc32(doc_id) % n`, `src/index/sharding.py`) |
| `ack 18.0 ms` | client-side latency until the ack |
| `async: queued, not yet visible` / `sync: applied, searchable now` | what the ack means in this apply mode |
| `FOUND (#1) (doc acked …, search sent …, answer back …, difference 13 ms)` | rank, then three clock times since the run started: when the write's ack came back, when this search was sent and when its answer came back. The difference is answer minus sent, the search's own round trip. How long after the ack the search was sent is a property of the script, not of the system |
| `VISIBLE after 11.6 ms (first poll)` | the visibility lag, checked against the 1 s budget. `first poll` means it was visible on the first try, so the lag is an upper bound of one round trip |
| `MISSING ~ soft, allowed` | a soft check missed (expected in async mode right after a write) |

The summary at the end:

| line | meaning |
|---|---|
| `expectations` | checks that held, hard and soft (split in brackets); hard checks that failed (they fail the run); soft checks that missed (allowed) |
| `writes`, `searches` | counts and client-side p50/p95; `(N partial)` = searches a shard missed |
| `answer checks` | `same_as`, `overlap`, `golden` passed/total |
| `expected docs` | searches that expected a doc, split by when they were sent after its ack: `<1 s` and `>=1 s`. This is when the search ran, not the lag. A search sent 3 s after the ack only shows the doc was visible by then |
| `visibility lag` | sync mode only: 0 for every acked write, because sync applies the write before it acks |
| `wait_visible` | each `wait_visible`'s measured lag (ack → the first poll that saw it, budget 1 s) and the max |
| `admin timeline` | migrate/cutover/rollback/finalize/kill/restart with times (`!` = failed) |
| `RESULT` | PASS / FAIL (exit code 0 / 1) |

Other flags: `-q` prints only labels, admin ops, failures and the summary; `--no-color`; `--no-sample` /
`--sample-ms 150` control the background status sampler that feeds the replay's cluster panel and charts;
`--async-max-lag` / `--async-batch-max` set `ASYNC_MAX_LAG_S` / `ASYNC_BATCH_MAX` for `--start-cluster`;
`--shard-of ID ... --shards 3` prints owner shards (that is how I picked the ids in 06/07).

## Reading the replay page

`replay.html` opens paused at t = 0; Space plays it. A sticky dock at the top holds the controls and the cluster stage:
clients (users and bots) on the left, the router and one box per shard in the middle, admin and ops on the right. Below
it is a vertical timeline with one row per op, and at the bottom a scoreboard, the freshness and answer-check panels and
the charts.

| area | what it shows |
|---|---|
| header | scenario, SYNC/ASYNC badge, PASS/FAIL, date, cluster info, model, run id, the actor list |
| How to read this | the architecture (clients → router → N shards, admin/ops → router `/admin/*`); flow diagrams for search (fan-out to all shards: stats phase, search phase, merge + fuse), write (router → the one owner shard `crc32(doc_id) % N` → WAL → ack; sync applies before the ack, async within ~1 s) and admin (migrate → wait_ready → cutover → rollback/finalize, kill/restart); the actor legend (role, how it was decided, a one-line summary of its recorded ops) and the lanes table. The status sampler is the runner's own thread (GET /admin/status every 150 ms), not part of the scenario |
| controls | play/pause (Space), prev/next event (←/→), restart (Home), speed 0.25x to 4x, scrubber, auto-scroll, cluster stage on/off. Markers: purple = label, orange = admin/ops, red = failed check, amber = soft miss |
| cluster stage | a chip per actor showing what it is doing; the router's active-model badge (flips at cutover/rollback); per shard: docs, writes routed, async queue depth / oldest age / lag p95, one bar per embedding model (purple = building, green = ready, ★ = active). A killed shard turns grey and striped, "DOWN" |
| packet animations | a search goes actor → router → all shards and back, then shows the outcome (`3 hits · fresh-1 FOUND ✓`, `partial: shard 1 missing`). An upsert/delete goes to the owner shard only; in async mode a doc chip sits on the shard as "queued" until the status samples show it applied. Migrate fills progress bars, cutover/rollback flip the model badge, kill_shard greys the shard |
| timeline | client cards left, admin/ops cards right, the cluster's reaction in the middle ("router → shard 1 (owner)", "router → all 3 shards (stats + search), merge + fuse ⇒ 2 hits"). Labels are full-width banners; fork/join, top-level sleeps and unknown op types are middle rows. A card appears when the request is sent and fills in when the response arrives. Search cards list the top hits (expected doc green if found, red if missing), the model that answered and any same/overlap check. Each card shows the actor, its lane when that differs, and the op's line in the file |
| scoreboard | checks passed/failed, soft misses, writes, ack p50/p95, searches, search p50/p95, errors, partial searches, all "so far" at the playhead |
| filters | under the narration in the dock: one chip per actor (clients and admin & ops, multi-select, all/none) and a box for a doc id or query text. A card stays if its actor is selected and the text matches its doc id, its query, its upserted text, a doc id in its results or a doc it expected. Banners stay; fork/join rows hide while a filter is on. The counter says "showing N of M ops". Playback and scrubbing keep working, and the freshness panel follows the text box |
| freshness | "was the new doc found?": every search that expected a recently written doc, found or missing, with doc acked at, search sent at, answer back at and the difference, plus every `wait_visible` lag |
| answer checks | every same-as / overlap / golden comparison, with the models involved |
| charts | docs in the index, queued writes, oldest queued age (dashed red line = the 1000 ms budget) |

URL options: `#t=5.2` opens paused at 5.2 s (I use it for screenshots), `?autoplay=1` starts playing, `?howto=0`
collapses the explanation.

### A search card, line by line

```text
🤖 probe-bot bot                                                  line 105 · 105.3 ms   ← client-measured round trip
🔎 Search "what is an etf"  hybrid k=10
10 hits  minilm  all 3 shards answered ✓                                                ← partial = false
[▇▇ ▇▇▇▇▇▇▇▇▇ ▇▇▇▇▇▇ ▇▇▇▇▇▇▇ ▇▇▇▇▇ ▏]                                                    ← the time split, to scale
client↔router 7.2 · stats (DFS) 32.0 · embed wait 24.3 · shards compute 23.5 · network/queue 17.2 · merge + fuse 0.8 · other 0.3
per shard compute: s0 7.5 ms, s1 23.5 ms, s2 8.2 ms · encoder 51.0 ms (overlaps stats)
```

"All 3 shards answered" means the answer is not partial: every shard replied in time. Every card shown is a finished
request (an in-flight one has a spinner). With a dead shard the card says `PARTIAL: shard 1 missing`. Pages generated
before this wording say "complete ✓"; `--replay-only` re-renders them.

The time split comes from the router (`timings_ms` in every `/search` response, measured in `Router.search`). The
segments are sequential and add up exactly to the card's time:

| segment | what it is | typical, real corpus, hybrid (scenario 04) |
|---|---|---|
| client↔router | HTTP hop client → router → client (client time − router time) | ~3 ms |
| stats (DFS) | phase 1: every shard returns term counts → global BM25 statistics. 0 for dense-only | ~15 ms |
| embed wait | waiting for the query embedding after the stats phase. The encoder runs in parallel with phase 1, so only its overhang shows (`encoder` = its full time) | ~0 to 15 ms |
| shards compute | phase 2: the slowest shard's own BM25 + faiss time (`took_ms` from each shard; shards run in parallel) | ~4 to 5 ms |
| network/queue | the rest of the phase-2 round trip: HTTP, JSON, thread hand-offs. With a dead shard it includes the wait until that shard's timeout | ~8 to 12 ms |
| merge + fuse | phases 3 and 4 at the router | < 1 ms |
| fetch/rerank | phase 5 (only with `include_text` / `rerank_k`) | 0 here |
| other | router bookkeeping | < 0.5 ms |

This is the sharding-latency finding, per query: the shards' own search work is a few ms, and most of a query's time
is round trips (stats phase + network) and the query encoder. In the card above shard 1 took 23.5 ms instead of ~8
because it was building the bge-small index at that moment, and the split shows that at a glance. Recordings made
before `timings_ms` was added to `/search` have no bar.

Everything on the page is derived from the recorded events (op type, actor, lane, request, response, status samples),
so any scenario file renders with no per-scenario code: unknown lane names, any number of shards, nested lanes, unknown
op types. The page embeds the events as JSON and replays them on a `requestAnimationFrame` clock (only while playing).
Scrubbing backwards resets and re-applies all events up to the new time without animations, which was simpler than
undoing them. Packets use the Web Animations API. The JavaScript in `scripts/replay_html.py` is commented.

## Simple scenarios

`scenarios/SimpleScenarios/` holds eight short CSV stories, each one step beyond the last. They use the personas
Elephant, Fox and Owl and the admin ADMIN, and every write uses a doc id with `{run}` in it.

| file | what happens | what it shows |
|---|---|---|
| `1_SimpleRead` | Elephant and Fox search | the baseline: fan-out, merge, the time split |
| `2_SimpleReadAfterWrite` | + Owl upserts a doc, Fox searches for it | read-your-writes: sync finds it at once; async may miss it for a few ms |
| `3_SimpleReadAfterWriteAndDelete` | + Owl deletes it, Fox searches three times expecting it gone (`!doc`) | deletes become visible the same way writes do |
| `4_ReadMigrateRead` | search, ADMIN `migrate` / `wait_ready` / `cutover` to bge-small, search again | answers switch models only at cutover |
| `5_ReadWriteMigrateRead` | Owl writes before the migration; Fox finds it with bge-small after cutover | the backfill embeds existing docs with the new model |
| `6_ReadWriteMigrateWriteDuringMigrateRead` | + Elephant writes between migrate and cutover; Fox finds both | dual-write: during a migration every write is embedded with both models |
| `7_ReadWriteMigrateWriteDuringMigrateRollbackRead` | + ADMIN rolls back; Fox finds both with minilm | rollback is instant and loses nothing |
| `8_Idempotency` | 3 with explicit versions: a retry, a late v1, a late v2 after the delete, then v4 | retries and stale writes are no-ops; a tombstone blocks resurrection |

`scripts/run_simple_scenarios.py` runs all eight on four clusters, each run on its own fresh 3-shard cluster
(`--start-cluster`): SimpleSyncTinyCluster and SimpleAsyncTinyCluster (`--tiny`), SimpleSyncFullCluster and
SimpleAsyncFullCluster (the real corpus). It writes `scenarios/SimpleScenarios/results/<cluster>/<scenario>/` with the
console output (`console.txt`), the events and the replay, all checked in, plus `results/SUMMARY.md`, a grid of
all 32 runs. [scenarios/SimpleScenarios/README.md](../scenarios/SimpleScenarios/README.md) reads my results.

## The scenario catalog

These are the JSONL files in `scenarios/ComplexScenarios/`. All run in both modes and declare their actors. "Tiny" means `--start-cluster --tiny`. Each takes 0.1 to 7 s (06: ~23 s
on Windows) plus ~10 to 15 s cluster start.

| file | cluster | actors | what happens | what to look for |
|---|---|---|---|---|
| `freshness_demo.jsonl` | any | fox, elephant, ops | fox upserts; elephant searches at once (bm25 + hybrid, soft), then 1 s later (hard); fox deletes, same again; then elephant's `wait_visible` measures the lag | sync: the immediate search always finds the doc, `0 ms after ack`. Async: it may miss (soft, amber), since the ack only meant "in the WAL". After 1 s both modes find it. On tiny the hash embedder applies in microseconds, so async usually finds it anyway; on the real corpus (MiniLM) the gap is real |
| `idempotency.jsonl` | any | fox (flaky network), elephant, ops | fox sends the same (doc, version) twice; v5 then a late v4; a versioned delete; an older upsert after the delete; re-create with v7. Elephant checks what is visible | every `expect_applied` holds in both modes: retries and stale writes are no-ops, and a tombstone blocks resurrection. Async works because the version check uses the latest accepted version, not the latest applied one |
| `01_single_node_read.jsonl` | 1 shard | elephant, ops | four questions (hybrid, dense, bm25, hybrid), saved with `save_as`, then 3 more rounds | identical ranked lists every time. `--golden-save` keeps them |
| `02_multi_node_read.jsonl` | 3 shards | elephant, ops | the same questions, same names | identical within the run; with `--golden-check` (golden from 01) identical to the single node, thanks to global BM25 stats + global merge before fusion. Search latency vs 01 shows the fan-out cost |
| `03_multi_node_read_write.jsonl` | 3 shards | bot-1, bot-2 (writers), bot-3 (reader), fox (freshness checker), admin, ops | bot-1/bot-2: 3 bursts of 20 upserts, then `wait_visible` on each burst's last doc (timeout 1 s); bot-3 searches `until_done` fox (`expect_ok`); fox: write → soft search → sleep 1 → hard search, 3 times; admin flushes, fox finds the bots' last docs | every write visible within 1 s while reads keep working. Async: queued writes and lag on the shard boxes and charts |
| `04_multi_node_read_migrate.jsonl` | 3 shards, minilm | elephant, admin, probe-bot, ops | elephant's baseline answers; admin's early cutover is refused (409); admin migrates and waits while elephant demands identical answers (`repeat until_signal "built"`); cutover (dense/hybrid overlap ≥ 0.3/0.2, bm25 identical); rollback (identical again); cutover + finalize; rollback now refused. Admin and elephant take turns with `signal` / `wait_signal`; probe-bot searches every 50 ms throughout | zero failed or partial queries across every flip; the model badge flips; the bge-small bars fill on every shard; `minilm → bge-small` overlaps and identical answers after rollback |
| `05_multi_node_read_write_migrate.jsonl` | 3 shards, minilm | fox, elephant, bot-1 (writer), bot-2 (reader), admin, ops | fox writes before the migration; bot-1 writes during the build; after cutover elephant's dense search must find both (backfill + dual-write); rollback identical to the quiet point; cutover + finalize while bot-1 writes; everything still found | dual-write: docs written while bge-small was building are found by dense search with bge-small; after finalize bge-small alone finds docs from every phase |
| `06_shard_failure.jsonl` | 3 shards, needs `--start-cluster` | fox, elephant, bot-1 (reader), ops | one lane per actor, in turn via signals: ops kills shard 1 while bot-1 searches; elephant's searches are partial (`missing_shards: [1]`) and its docs absent; fox's write owned by shard 1 fails (503 after retries), her write to shard 0 works; ops restarts it; answers identical; the failed write is absent until fox retries | the greyed-out shard, amber bot-1 cards during the outage, the restart card with the number of WAL records replayed. Outage searches take ~4 s on Windows (connecting to a dead port is slow) |
| `07_async_burst_crash.jsonl` | 3 shards, needs `--start-cluster` | bot-1..3, ops, elephant | the bots write 30 docs owned by shard 1 as fast as possible; ops kills it right after the last ack and restarts it; elephant searches every doc | every acked doc is present. This matters in async mode: acked-but-unapplied writes survive because the ack comes after the WAL fsync and the restart re-embeds them. Sync is a control run. On tiny the queue is usually empty at the kill; on the real corpus writes can be caught in flight |

On the tiny cluster both "models" are hash embedders with different seeds, so the backfill is instant but answers do
change at cutover. On the real corpus `bge-small` comes from the embedding cache `artifacts/emb_bge-small.npy` (the
backfill reads cached vectors, so it is fast too; queries and dual-writes need the model in the local Hugging Face
cache).

## Results from my runs

Tiny corpus, every catalog scenario in both modes, all PASS (from my verification run of the tool; other runs differ
slightly):

| scenario | hard checks (sync / async) | notes |
|---|---|---|
| freshness_demo | 5 / 5 | immediate searches found the doc in both modes (hash apply takes µs) |
| idempotency | 13 / 13 | 9 writes, 4 applied, 5 no-ops as expected |
| 01_single_node_read (`--golden-save`) | 28 / 28 | same_as 12/12, search p50 ~4 ms |
| 02_multi_node_read (`--golden-check`) | 32 / 32 | same_as 12/12, golden 4/4 (3 shards == 1 shard), search p50 ~11 ms |
| 03_multi_node_read_write | 68 / 68 | 123 writes; every burst's last doc visible after 26 to 39 ms (first poll). Fox's final searches were sent ~2 s after the ack only because they ran after the load phase |
| 04_multi_node_read_migrate | 99 / 99 | same_as 33/33, overlap 6/6; probe-bot never saw a failed query |
| 05_multi_node_read_write_migrate | 41 / 41 | dual-written docs found by dense search after cutover and finalize |
| 06_shard_failure | 26 / 26 | 7 partial searches (p95 ~4 s), 5 soft misses by the bot-1 reader during the outage |
| 07_async_burst_crash | 30 / 30 | 30/30 acked docs present after kill + restart |

Real corpus (57,638 docs, MiniLM; cluster start ~1 to 1.5 min):

| scenario | result |
|---|---|
| `freshness_demo`, async | PASS. Elephant's immediate BM25 search, sent 0 ms after the ack, missed fox's doc (soft: the doc was still in its shard's queue being embedded). The immediate hybrid search, sent 15 ms after the ack, found it at #1, because the router embeds the query first and that gave the applier time. 1 s later: found. Same for the delete. `wait_visible`: visible after 37 ms (first poll) |
| `04_multi_node_read_migrate`, sync | PASS, 87 checks: same_as 25/25 (identical during the build and after rollback); overlap after cutover q1 0.70, q2 0.30, q4 0.40 (a different embedding model really changes dense answers); BM25 identical throughout; zero failed or partial queries, probe-bot included |

## Freshness: what the numbers mean

The promise is about visibility lag: the time from a write's ack until a search can find the doc. In sync mode it is 0,
since the shard applies the write before it acks. In async mode back-pressure keeps it under 1 s: a writer waits when
the shard predicts its queue would take more than 0.5 s to drain (`ASYNC_MAX_LAG_S`, `src/service/async_apply.py`).

The replay and the console show when each search was sent and answered, and those times are easy to misread as a lag.
A search sent 3 s after the ack that finds the doc says the doc was visible by then; it says nothing about the first
3 s. The direct measurement is `wait_visible`, which polls from the ack until the first search sees the doc.

What I measured, across the 29 runs I have recorded (local clusters on the tiny and the real corpus, and Docker):

| measurement | result |
|---|---|
| every `wait_visible` probe (38, both modes) | all 38 saw the doc on their first poll, so each number is one search round trip, an upper bound on the lag: 9.8 to 38.9 ms on local clusters, 78 to 95 ms through Docker |
| async under write load (`results/write_load.json`, 8 writers) | lag p95 466 ms, max 583 ms |
| searches sent ≥ 1 s after an ack (up to 6.5 s) | 81, all found the doc |
| searches sent < 1 s after an ack | 39, 38 found. The miss was a soft check in async mode, sent 0.1 ms after the ack (allowed: async promises 1 s, not instant) |

No measurement went over 1 s. The bound comes from back-pressure, so it is a measured property under the load I tried,
not a hard real-time guarantee under unbounded overload.

## Writing a scenario

I start each phase with a `label`, which becomes the narration in the replay, and declare the actors in
`config.actors`. Against a long-lived cluster I put `{run}` in doc ids, and I query with the doc's exact text when I
expect to find it. In async mode, checks right after a write are soft; hard checks come after `sleep 1.0`,
`wait_visible` or `flush`. `expect_same_as` only holds across stretches where nothing changes the ranking (no writes,
same model). Continuous traffic is a parallel lane with `{"op": "repeat", "until_done": "<other lane>", "ops": [...]}`,
and turn-taking between actors is one lane per actor with `signal` / `wait_signal` (see 04 and 06). For ids on a given
shard: `python scripts/scenario.py --shard-of a1 a2 a3 --shards 3`. For a straight sequence with no parallel lanes, the
simple CSV format above is quicker to write.
