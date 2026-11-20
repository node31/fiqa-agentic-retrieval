# Scenarios: script the live service, check it, replay it

`scripts/scenario.py` plays a small file of operations (writes, searches, waits, admin actions,
shard crashes) against the **running** distributed service over real HTTP. For every operation it
prints whether it worked and whether the expectations held, and it records everything so you can
**replay** the run as an animated HTML page.

Why it exists: the service makes promises about *time and order* that are easy to state and hard to
*see*:

| promise | where it is implemented | scenario that shows it |
|---|---|---|
| a write is searchable within 1 s of its ack (sync: at once) | `store.py` (sync), `async_apply.py` (async) | `freshness_demo`, `03` |
| retries and late, older versions are harmless no-ops | `store._is_newer`, `AsyncApplier.accepted` | `idempotency` |
| 3 shards answer exactly like 1 node | `router.py` DFS phase + global merge (ADR 0003) | `01` + `02` (golden) |
| a model migration never fails a query; answers change only at cutover; rollback restores them | `router.py` MODEL MIGRATION, `store.start_build` | `04`, `05` |
| a dead shard gives *partial* answers, not errors; a restart loses no acked write | ADR 0006, `wal.py` | `06`, `07` |

The files: `scripts/scenario.py` (runner), `scripts/replay_html.py` (replay page generator),
`scenarios/*.jsonl` (examples), `tests/test_scenario.py` (runs them in both modes).

---

## 1. Quick start

```powershell
# fastest: tiny 24-doc corpus + hash "embedder", a fresh 3-shard cluster on ports 19000/19101+ (starts in ~10 s)
python scripts/scenario.py scenarios/freshness_demo.jsonl --start-cluster --tiny --mode sync
python scripts/scenario.py scenarios/freshness_demo.jsonl --start-cluster --tiny --mode async

# the real FiQA corpus (57,638 docs) + MiniLM: same command without --tiny (startup ~1-2 min)
python scripts/scenario.py scenarios/freshness_demo.jsonl --start-cluster --mode async

# against a cluster you already run (python -m src.service.cluster, docker compose, ...)
python scripts/scenario.py scenarios/freshness_demo.jsonl --url http://127.0.0.1:8000

# re-render the replay page from a recorded log
python scripts/scenario.py --replay-only toy_runs/scenarios/<run>/events.jsonl
```

On Windows set `$env:PYTHONIOENCODING='utf-8'` (check marks) and `$env:HF_HUB_OFFLINE='1'` (real
corpus, models from the local cache). The exit code is **0 if every hard expectation passed, 1 if not,
2 for a scenario file error**, so a scenario is also a test.

Each run writes to `toy_runs/scenarios/<scenario>-<mode>-<timestamp>/` (gitignored):

| file | what |
|---|---|
| `events.jsonl` | every event: op start/end, status samples, summary (one JSON per line) |
| `replay.html` | the animated replay: one self-contained file, open it in any browser (works offline) |
| `cluster_logs/` | shard and router logs, only with `--start-cluster` (look here when something fails) |

With `--start-cluster` the WALs/state live in `artifacts/scenario_runs/<run>/` and are deleted after
the run (`--keep-state` keeps them). Default ports **19000 (router) / 19101+ (shards)**, never the dev
cluster's 8000/8101+ or the tests' 18000-18999.

---

## 2. The file format

JSON Lines: one op per line. Blank lines and lines starting with `#` are comments. An op may span
several lines (the parser reads consecutive JSON values), which keeps `parallel` blocks readable.
Unknown ops or fields are an **error** with the line number (a typo like `"expected"` would otherwise
silently check nothing).

### 2.1 Templating

Any string may contain placeholders, replaced when the op runs:

| placeholder | value |
|---|---|
| `{run}` | a random 6-hex-char id per run. Put it in doc ids and distinctive tokens so re-runs against a long-lived cluster always write NEW docs (an old id with version 1 would be a no-op) |
| `{i}` (or the `var` you chose) | the `repeat` counter, or the current item of `each` |
| `{lane}` | the current lane name |

Unknown placeholders stay as they are. Use texts with unique tokens (`"zebraprobe{run} annuity"`)
and query with the same text: BM25 then finds the doc deterministically on the tiny AND the real
corpus, and dense search does too (identical text = identical vector).

### 2.2 Ops

| op | required | optional (default) | what it does |
|---|---|---|---|
| `config` | | `shards` (3), `mode` (sync), `models` (minilm), `tiny` (false), `description`, `actors` (2.5) | must be the FIRST op. Describes the cluster `--start-cluster` should start, and who acts in the story. CLI flags win (`--shards`, `--mode`, `--models`, `--tiny`/`--real`). With `--url` it is only compared (warning if the shard count differs) |
| `upsert` | `doc_id`, `text` | `version` (auto: last version of this doc in this run + 1), `expect_applied`, `expect_error`, `soft` | POST /upsert |
| `delete` | `doc_id` | `version` (auto), `expect_applied`, `expect_error`, `soft` | POST /delete |
| `search` | `query` | `strategy` (hybrid), `k` (10), `fusion`, `alpha`, `save_as`, expectations (2.3), `soft` | POST /search |
| `sleep` | `seconds` | | wait |
| `wait_visible` | `query`, `doc_id` | `timeout` (2.0), `strategy` (hybrid), `k` (10), `absent` (false), `interval` (0.02) | poll search until the doc appears (or disappears with `absent`). Reports the **lag** = time from the doc's last ack in this run to the search that first saw it (an upper bound: it includes that search's round-trip). Fails after `timeout` |
| `flush` | | `timeout` (10) | POST /admin/flush: wait until every acked write is applied (instant in sync mode) |
| `status` | | | GET /admin/status snapshot: docs, async queue, lag, models per shard |
| `label` | `text` | | narration: a banner in the console and the replay |
| `migrate` | `model` | `throttle_s` (0), `reencode` (false), `expect_error` | POST /admin/migrate: every shard starts building `model`'s index in the background; dual-write starts now; queries keep the active model |
| `wait_ready` | `model` | `timeout` (120), `interval` (0.1) | poll /admin/status until every shard reports `model` ready; each poll is a progress sample for the replay |
| `cutover` | `model` | `expect_error` | POST /admin/cutover: the router queries with `model` from now on (409 unless ready everywhere) |
| `rollback` | | `expect_error` | POST /admin/rollback: back to the previous model (instant) |
| `finalize` | `model` | `expect_error` | POST /admin/finalize: drop `model`'s index (no rollback to it afterwards) |
| `kill_shard` | `shard` | | hard-kill the shard process (like a crash). **Needs `--start-cluster`** |
| `restart_shard` | `shard` | `timeout` (300) | start it again and wait for /health (base corpus + WAL replay happen first). Needs `--start-cluster` |
| `repeat` | `ops` + one of `times` / `each` / `until_done` / `until_signal` | `var` (i), `start` (0) | `times: N` runs `ops` N times with `{i}` = start..; `each: [...]` runs once per item (`{i}` = the item); `until_done: "lane"` loops until that lane finished; `until_signal: "name"` loops until some lane sent that signal (both at least once; `times` then caps them) |
| `parallel` | `lanes` | | runs lanes concurrently, one thread + one keep-alive HTTP client each, and waits for all (fork/join). A lane is `{"name": "bot-1", "actor": "...", "description": "...", "ops": [...]}` (`actor` defaults to the name, `description` is shown in the replay) or a bare list (named `lane0`, `lane1`, ...). Nested lanes are named `parent.child` |

| `signal` | `name` | | coordination between lanes (no HTTP): marks the named event as happened. Lanes waiting for it continue |
| `wait_signal` | `name` | `timeout` (120) | block this lane until some lane sent `signal` `name` (fails after `timeout`). A name nobody signals is a file error. Use it for turn-taking: "admin cut over" → "elephant may check the answers now" → "elephant checked" → "admin rolls back" |

Every op except `config`, `label`, `repeat` and `parallel` also accepts **`"actor": "name"`**: who
performs it (section 2.5). Every op records its **own** line of the file (also inside a multi-line
`parallel` block), shown on its console line and its replay card.

### 2.3 Expectations (checks)

A **hard** check decides pass/fail. Add `"soft": true` to an op to make all its checks soft: they are
recorded and shown (amber) but never fail the run.

| field (on) | passes when |
|---|---|
| `expect: [ids]` (search) | every id is in the results. Records the rank and `since_ack_ms` = when this search was SENT after the doc's last ack (not a visibility lag, see below) |
| `expect_absent: [ids]` (search) | no id is in the results |
| `expect_ok: true` (search) | the search is "healthy": no error, not partial, at least one hit |
| `expect_partial: true/false` (search) | `partial` has that value |
| `expect_missing_shards: [1]` (search) | exactly these shards were missing |
| `save_as: "q1"` (search) | (not a check) remembers the ranked id list under that name |
| `expect_same_as: "q1"` (search) | the ranked list is IDENTICAL (same ids, same order). On failure: first differing rank, ids lost, ids gained |
| `expect_overlap_with: {"ref": "q1", "min": 0.6}` (search) | at least `min` of q1's ids are still in the results (order ignored) |
| `expect_applied: true/false` (write) | the shard applied the write / treated it as a no-op (retry or older version) |
| `expect_error: true` (write, search, admin) | the call FAILS (HTTP error or no response). E.g. a cutover to an unbuilt model (409), a write owned by a dead shard (503) |
| (implicit) `wait_visible`, `wait_ready`, `flush` | the doc became visible / the model became ready / the queue drained in time |
| (implicit) any op | an unexpected HTTP error fails the op |

**Two different times (do not mix them up).**

| you see | it means | it is NOT |
|---|---|---|
| search check: `FOUND #1 · searched 2.07 s after its ack` | the search happened to RUN 2.07 s after the doc's ack (e.g. after a sleep or a load phase) and found the doc | how long the doc took to become visible |
| `wait_visible`: `VISIBLE after 34 ms (first poll)` | the **visibility lag**: time from the ack to the first search that saw the doc. THE freshness number; budget 1 s | |

So to measure freshness use `wait_visible`; a hard `expect` after `sleep 1.0` only proves "visible
by then".

**When answers must be identical, and when they may change** (encoded in scenarios 04/05):

| situation | hybrid / dense | bm25 | check with |
|---|---|---|---|
| repeated query, nothing changed | identical | identical | `expect_same_as` |
| 1 shard vs 3 shards | identical | identical | golden file |
| during a migration build (before cutover) | identical (queries still use the old model) | identical | `expect_same_as` |
| after cutover | may CHANGE (other embedding model) | identical (BM25 ignores the model) | `expect_overlap_with` + `expect_ok` for dense/hybrid, `expect_same_as` for bm25 |
| after rollback | identical to before the cutover | identical | `expect_same_as` |
| new docs were written in between | may change, even for unrelated queries: BM25's global statistics (doc count, average length) change | may change | `expect_ok`, `expect` for the new docs |

### 2.4 Golden files (compare across runs or clusters)

`--golden-save path.json` writes every `save_as` answer of the run; `--golden-check path.json` compares
this run's `save_as` answers with the file (exact ranked lists, by name) and adds the result as hard
checks (op `golden_check`, the last row of the replay's timeline). Typical use: 1 shard vs 3 shards:

```powershell
python scripts/scenario.py scenarios/01_single_node_read.jsonl --start-cluster --tiny --golden-save toy_runs/golden/read_tiny.json
python scripts/scenario.py scenarios/02_multi_node_read.jsonl  --start-cluster --tiny --golden-check toy_runs/golden/read_tiny.json
```

A golden file is only comparable with the same corpus and model (tiny/hash vs real/MiniLM give
different answers); its `meta` records where it came from. It stores only the `save_as` names and the
ranked ids (no lane or actor names), so renaming actors never invalidates a golden file.

### 2.5 Actors: who does what

A scenario reads like a story: every op is performed by an **actor**, and every actor has one of four
**roles**. The replay puts clients on the LEFT, the cluster in the MIDDLE and privileged operators
on the RIGHT, so you always see who did what.

| role | shown as | may run | typical names |
|---|---|---|---|
| `user` | an animal persona: 🦊 fox, 🐘 elephant, 🐶 dog, 🦉 owl, 🐼 panda, 🐨 koala, ... | client ops: `upsert`, `delete`, `search`, `wait_visible` (+ neutral ops) | fox, elephant (any name) |
| `bot` | 🤖 | client ops (automated traffic: background writers/readers, synthetic probes/canaries) | bot-1, probe-bot, load-a, writer-2, reader |
| `admin` | 🛠️ Admin | admin ops: `migrate`, `wait_ready`, `cutover`, `rollback`, `finalize`, `flush` (+ ops-manager ops, + neutral ops) | admin |
| `ops` | 🧑‍🔧 Ops manager | ops-manager ops: `kill_shard`, `restart_shard`, `status` (+ admin ops, + neutral ops) | ops, chaos, sre |

The split is **strict both ways**: clients never run admin/ops ops, and admins/ops managers never
search or write. A health probe that searches during a migration is therefore a bot (🤖 probe-bot),
shown on the left with the other clients. **Neutral ops** (`sleep`, `label`, `signal`,
`wait_signal`) are allowed for every role. A `label` and a top-level `sleep` belong to nobody: they
are the narration / the story pausing, shown in the middle.

**Declaring actors** (recommended; the whole catalog does it) in the `config` line:

```json
{"op": "config", "shards": 3, "actors": {
  "fox":      {"role": "user",  "description": "writes docs"},
  "elephant": {"role": "user",  "description": "searches for them"},
  "bot-1":    {"role": "bot",   "description": "background writer"},
  "admin":    {"role": "admin", "description": "runs the migration"},
  "ops":      {"role": "ops",   "description": "kills and restarts a shard"}}}
```

(`emoji` may override the picture.) Then say who acts, at the level that reads best:

| where | example | meaning |
|---|---|---|
| per op | `{"op": "status", "actor": "ops"}` | this op only. Handy at the top level of the file |
| per lane | `{"name": "canary", "actor": "probe-bot", "ops": [...]}` | every op of the lane (and of its nested lanes) |
| lane name | `{"name": "bot-1", "ops": [...]}` | a lane's actor defaults to its own (short) name. **Recommended**: one lane per actor, named like the actor (the catalog does this); lanes take turns with `signal` / `wait_signal`. A replay card names its lane only when it differs from the actor |
| top level | `{"op": "cutover", "model": "m"}` | no lane, no `actor`: client ops go to the first declared user (else bot, else an actor named `user`), admin-type ops to the first declared admin (else `admin`), ops-type ops to the first declared ops (else `ops`) |

**Undeclared actors** (old files, quick custom files) are inferred, in this order:

1. the name: `admin*`, `build*`, `migrat*`, `operator*` → admin; `ops*`, `chaos*`, `sre*`, `oncall*`
   → ops; `bot*`, `load*`, `burst*`, `writer*`, `reader*`, `checker*`, `probe*`, `canary*`,
   `watchdog*`, `monitor*` or the word `bot` anywhere → bot (a watchdog SEARCHES, so it is a client);
   an animal name (fox, owl, ...) → user;
2. else what it does: any admin-type op → admin; any ops-type op → ops; otherwise → user
   (neutral ops do not count).

Users get the next free animal persona in order of first appearance (deterministic); generic names
(`user`, `main`, `lane2`) are displayed as that animal.

**Validation (fail fast, both ways).** Before anything runs (exit code 2), the runner rejects
a client actor (user, bot) running a privileged op (`migrate`, `wait_ready`, `cutover`, `rollback`,
`finalize`, `flush`, `kill_shard`, `restart_shard`, `status`), and an admin/ops actor running a client
op (`search`, `upsert`, `delete`, `wait_visible`). It never silently moves the op:

```text
scenario error: line 7: 'migrate' by actor 'fox' (role user, declared): admin ops must run in an
admin/ops actor. Add "actor": "admin" (or "ops") to the op, or run it in a lane whose actor has
role admin/ops in config.actors

scenario error: line 12: 'search' by actor 'ops' (role ops, declared): client ops
(search/upsert/delete/wait_visible) must run in a user/bot actor. Add "actor": "<a user or bot>" to
the op (e.g. a probe bot for health searches), or declare the lane's actor with role user/bot in
config.actors
```

**Old lane names** (before actors existed): `main` was the file's top-level sequence, `admin` the
migration lane, `admin.build` / `admin.questions` its nested build and check lanes, `watchdog` the
health-query loop, `chaos` the shard killer, `burst-a/b/c` the burst writers. In the catalog every
lane is now named after its actor: 🛠️ `admin` runs the whole migration, 🐘 `elephant` the answer
checks (taking turns with the admin through signals), 🤖 `probe-bot` the continuous health searches
(it searches, so it is a bot), 🧑‍🔧 `ops` the shard kill/restart, 🤖 `bot-1..3` the burst writers.
Top-level ops (lane `main`) carry an explicit `"actor"`.

---

## 3. Reading the console output

```text
[   0.555s] fox        UPSERT  fresh-5289e1 v1 -> shard 2 ... OK applied (ack 18.0 ms, sync: applied, searchable now)
                               expect_applied=True: got applied=True ✓
[   0.573s] elephant   SEARCH  "zebraprobe5289e1 annuity ladder freshne…" [hybrid k=10] -> 10 hits (12.7 ms, model minilm)
                               expected fresh-5289e1: FOUND (#1), searched 0 ms after its ack ✓
[   2.646s] elephant   WAIT    fresh2-5289e1 VISIBLE after 11.6 ms (first poll) · freshness budget 1 s ✓
[   0.052s] admin      CUTOVER failed as expected: HTTP 409: refusing cutover: bge-small not ready on every shard
```

| part | meaning |
|---|---|
| `[   0.555s]` | when the op STARTED, seconds since the run began |
| `fox` | the actor (2.5) that performed the op; `main` (or a lane name) for labels / fork-join lines |
| `-> shard 2` | the owner shard (`crc32(doc_id) % n`, `src/index/sharding.py`) |
| `ack 18.0 ms` | client-side latency until the ack |
| `async: queued, not yet visible` / `sync: applied, searchable now` | what the ack means in this apply mode |
| `FOUND (#1), searched 0 ms after its ack` | rank, and WHEN this search was sent relative to the doc's ack. Not a visibility lag: a search after a 2 s load phase says "searched 2.07 s after its ack" even if the doc was visible after 30 ms |
| `VISIBLE after 11.6 ms (first poll)` | `wait_visible`: the **visibility lag** (ack → first search that saw the doc), checked against the 1 s freshness budget. `first poll` = it was already visible on the first try (the lag is then an upper bound: one search round-trip) |
| `MISSING ~ soft, allowed` | a soft check missed (expected in async mode right after a write) |

The summary at the end:

| line | meaning |
|---|---|
| `expectations` | hard checks passed / failed; soft ones found / missed |
| `writes`, `searches` | counts and client-side p50/p95 latency; `(N partial)` = searches a shard missed |
| `answer checks` | `same_as`, `overlap`, `golden` passed/total |
| `freshness` | searches that expected a doc, split by when they were SENT after its ack: `<1 s` and `>=1 s`. The guarantee is about the `>=1 s` bucket: it must be N/N in both modes |
| `visibility lags` | each `wait_visible`'s measured lag: THE freshness number (budget 1 s) |
| `admin timeline` | migrate/cutover/rollback/finalize/kill/restart with times (`!` = failed) |
| `RESULT` | PASS / FAIL (= exit code 0 / 1) |

Flags: `-q` prints only labels, admin ops, failures and the summary (handy for long scenarios);
`--no-color`; `--no-sample` / `--sample-ms 150` control the background status sampler (it feeds the
replay's cluster panel and charts); `--async-max-lag` / `--async-batch-max` set `ASYNC_MAX_LAG_S` /
`ASYNC_BATCH_MAX` for `--start-cluster`; `--shard-of ID ... --shards 3` prints owner shards (how the
ids in 06/07 were picked).

---

## 4. Reading the replay page

Open `replay.html` in any browser (one self-contained file, works offline). It opens paused at
t = 0; press Space (or `play`). The page reads like a story of **who does what**, top to bottom:

```text
+--------------------------------------------------------------------------------------------+
| header: scenario, SYNC/ASYNC, PASS/FAIL, date, cluster, model, actors (🦊 fox · 🛠️ admin …) |
| ▸ How to read this (open by default): architecture diagram, SEARCH / WRITE / ADMIN flows,   |
|   actor legend + lanes table generated from THIS recording                                 |
+==================================== sticky dock ===========================================+
| controls: prev · play · next · restart · speed · time · auto-scroll · cluster stage         |
| scrubber + markers                                       narration (latest label)          |
|  CLIENTS: USERS & BOTS   |  router (queries → model)  |  ADMIN & OPS                      |
|  🦊 fox  🐘 elephant  🤖  |  shard0  shard1  shard2 …  |  🛠️ admin  🧑‍🔧 ops  📡 sampler  |
+============================================================================================+
|  client cards (left)     |  t = 0.012 s  router → shard 1 (owner)  |  admin/ops cards (right) |
|  …                       |  t = 0.025 s  fan-out to 3 shards        |  …                       |
+--------------------------------------------------------------------------------------------+
| scoreboard · freshness · answer checks · charts                                            |
+--------------------------------------------------------------------------------------------+
```

| area | what you see |
|---|---|
| header | scenario, SYNC/ASYNC badge, PASS/FAIL, date, cluster info, model, run id, the actor list |
| How to read this | (a) the architecture: clients → router → N shards, admin/ops → router `/admin/*`; (b) three mini flow diagrams: **SEARCH** (router fans out to ALL shards: stats phase, search phase, merge + fuse), **WRITE** (router → the ONE owner shard `crc32(doc_id) % N` → WAL → ack; sync: applied before the ack, async: queued, applied within ~1 s), **ADMIN** (migrate → wait_ready → cutover → rollback/finalize; kill/restart a shard); (c) the **actor legend**: emoji, name, role, how the role was decided (declared / from its name / from its ops) and a one-line "what this actor does in this scenario" counted from the recorded ops (e.g. "writes 60 docs, waits 3 times for docs to become visible"), plus the **lanes** table (each thread, its actor, its plan, its description). The 📡 status sampler is the runner's own background thread (GET /admin/status every 150 ms) that feeds the shard boxes and charts; it is not part of the scenario. `?howto=0` starts it collapsed |
| controls (sticky) | play/pause (Space), prev/next event (←/→), restart (Home), speed 0.25x-4x, scrubber, auto-scroll on/off, cluster stage on/off. Markers: purple = label, orange = admin/ops, red = failed check, amber = soft miss; click to jump |
| cluster stage (sticky, always visible) | LEFT box: one chip per user/bot showing what it does right now ("searching 'otterprobe'", "upsert down1 → shard 1"). MIDDLE: the router (active model badge, flips with an animation at cutover/rollback) and one box per shard (docs, writes routed, async queue depth / oldest age / lag p95, one bar per embedding model: purple = building, green = ready, ★ = active; a killed shard turns grey and striped, "DOWN"). RIGHT box: admin and ops chips |
| packet animations | driven only by the event type: **search** = a packet from the actor's chip to the router labelled `SEARCH hybrid "…"`, fan-out to ALL shards and back, then the outcome (`3 hits · fresh-1 FOUND ✓`, `partial: shard 1 missing`); **upsert/delete** = actor → router → the OWNER shard only (highlighted) and the ack back; in async mode a doc chip sits on the shard as "queued" until the status samples show it applied; **admin/ops** ops come from the right: migrate fills progress bars on every shard, cutover/rollback flip the router's model badge, kill_shard greys the shard, restart brings it back |
| vertical timeline | one row per op at its start time; client cards on the LEFT, admin/ops cards on the RIGHT (never on the left), the cluster's reaction in the MIDDLE ("router → shard 1 (owner)", "router → all 3 shards (stats + search), merge + fuse ⇒ 2 hits", "partial: shard 1 missing", "(🤖 bot-1 pauses)"). Labels are full-width banners; fork/join, top-level sleeps and unknown ops are middle rows (an unknown op type gets a generic card with its request/response). A card appears when the request is SENT (spinner) and fills in when the response ARRIVES; search cards list the top hits with the expected doc green (found) or red (missing), the model that answered and any same/overlap check. Each card shows the actor (plus its role when the name does not already say it: `🦊 fox user`, but just `🛠️ admin`), its lane only when that is not the actor's own lane, and the op's own line in the scenario file. The column follows the op type, strictly: search/upsert/delete/wait_visible on the left, admin/ops ops on the right; neutral ops (sleep, signal, wait_signal) follow their actor. During playback the page **auto-scrolls** so the current row stays in view (highlighted); click any row to jump there |
| scoreboard (bottom) | checks passed/failed, soft misses, writes, ack p50/p95, searches, search p50/p95, errors, partial searches, all "so far" at the playhead |
| freshness | every search that expected a recently written doc (found/missing, time since ack, `<1 s` / `≥1 s`) and every `wait_visible` lag |
| answer checks | every same-as / overlap / golden comparison, with the models involved |
| charts | docs in the index, queued writes, oldest queued age (dashed red line = the 1000 ms freshness budget), with a playhead |

URL options (append to the file URL): `#t=5.2` opens paused at 5.2 s (used for screenshots),
`?autoplay=1` starts playing, `?howto=0` collapses the explanation section.

### 4.1 A search card, line by line

```text
🤖 probe-bot bot                                                  line 105 · 105.3 ms   ← client-measured round trip
🔎 Search "what is an etf"  hybrid k=10
10 hits  minilm  all 3 shards answered ✓                                                ← partial = false
[▇▇ ▇▇▇▇▇▇▇▇▇ ▇▇▇▇▇▇ ▇▇▇▇▇▇▇ ▇▇▇▇▇ ▏]                                                    ← the time split, to scale
client↔router 7.2 · stats (DFS) 32.0 · embed wait 24.3 · shards compute 23.5 · network/queue 17.2 · merge + fuse 0.8 · other 0.3
per shard compute: s0 7.5 ms, s1 23.5 ms, s2 8.2 ms · encoder 51.0 ms (overlaps stats)
```

- **"all 3 shards answered"** (replay pages generated before this change say "complete ✓"; re-render them with
  `--replay-only`) means the answer is **not partial**: every shard
  replied in time. It does *not* mean "the request finished": every card you see is a finished request (an
  in-flight one shows a spinner). With a dead shard the card says `PARTIAL: shard 1 missing` instead.
- **The time split** comes from the router (`timings_ms` in every `/search` response, measured in
  `Router.search`). The segments are **sequential** and add up exactly to the card's time:

| segment | what it is | typical, real corpus, hybrid (scenario 04) |
|---|---|---|
| client↔router | HTTP hop client → router → client (= client time − router time) | ~3 ms |
| stats (DFS) | phase 1: every shard returns term counts → global BM25 statistics. 0 for dense-only | ~15 ms |
| embed wait | waiting for the query embedding *after* the stats phase. The encoder runs **in parallel** with phase 1, so only its overhang shows (`encoder` = its full time) | ~0–15 ms |
| shards compute | phase 2: the **slowest** shard's own BM25 + faiss time (`took_ms` reported by each shard; all shards run in parallel) | ~4–5 ms |
| network/queue | the rest of the phase-2 round-trip: HTTP, JSON, thread hand-offs. With a dead shard it also contains the wait until that shard's timeout | ~8–12 ms |
| merge + fuse | phases 3-4 at the router | < 1 ms |
| fetch/rerank | phase 5 (only with `include_text` / `rerank_k`) | 0 here |
| other | router bookkeeping | < 0.5 ms |

What this teaches (it is ch07's "sharding latency" finding, per query): the shards' actual search work is a few ms; most of a
query's time is **round-trips** (stats phase + network) and the query encoder. In the card above, shard 1 took 23.5 ms
instead of ~8 because it was building the bge-small index at that moment (the migration), and the split shows it at a glance.
Recordings made before this field existed have no bar (`timings_ms` was added to `/search` later).

Everything is derived from the recorded events (op type, actor, lane, request, response, status
samples), so ANY scenario file renders and animates with no per-scenario code: unknown lane names,
any number of shards, nested lanes, unknown op types. Old recordings without actor fields are shown
with the same role inference as 2.5.

Replay internals (for the curious): the page embeds the events as JSON and replays them on a clock
driven by `requestAnimationFrame` (only while playing, so an idle page costs nothing). Scrubbing
backwards resets and re-applies all events up to the new time without animations, which is simpler
than undoing them. Packets use the Web Animations API. The JavaScript is commented; read
`scripts/replay_html.py`.

---

## 5. The scenario catalog

All run in both modes and declare their actors. "Tiny" = `--start-cluster --tiny`. Each takes
0.1-7 s (06: ~23 s on Windows) plus ~10-15 s cluster start.

| file | cluster | actors | what happens | what to observe |
|---|---|---|---|---|
| `freshness_demo.jsonl` | any | 🦊 fox, 🐘 elephant, 🧑‍🔧 ops | fox upserts; elephant searches at once (bm25 + hybrid, soft) → 1 s later (hard); fox deletes → same; then elephant's `wait_visible` measures the lag | **sync**: the immediate search always finds the doc, `0 ms after ack`. **async**: it MAY miss (soft, amber): the ack only meant "in the WAL". After 1 s both modes find it. On tiny the hash embedder applies in microseconds, so async usually finds it anyway; on the real corpus (MiniLM) the gap is real (see §6) |
| `idempotency.jsonl` | any | 🦊 fox (flaky network), 🐘 elephant, 🧑‍🔧 ops | fox sends the same (doc, version) twice; v5 then a late v4; a versioned delete; an older upsert after the delete; re-create with v7. Elephant checks what is visible | every `expect_applied` holds in both modes: retries and stale writes are no-ops, a tombstone blocks resurrection. In async mode this works because the version check uses the latest ACCEPTED version, not the latest applied one |
| `01_single_node_read.jsonl` | 1 shard | 🐘 elephant, 🧑‍🔧 ops | elephant asks four questions (hybrid, dense, bm25, hybrid), remembers them (`save_as`), asks 3 more rounds | identical ranked lists every time (determinism). `--golden-save` keeps them |
| `02_multi_node_read.jsonl` | 3 shards | 🐘 elephant, 🧑‍🔧 ops | the same questions, same names | identical within the run; with `--golden-check` (golden from 01) identical to the single node: global BM25 stats + global merge before fusion. Compare the search latency with 01 (fan-out cost) |
| `03_multi_node_read_write.jsonl` | 3 shards | 🤖 bot-1, bot-2 (writers), 🤖 bot-3 (reader), 🦊 fox (freshness checker), 🛠️ admin, 🧑‍🔧 ops | bot-1/bot-2: 3 bursts of 20 upserts, then `wait_visible` on each burst's last doc (timeout 1 s); bot-3 searches `until_done` fox (`expect_ok`); fox: write → soft search → sleep 1 → hard search, 3×; admin flushes, fox finds the bots' last docs | every write visible within 1 s while reads keep working. **async**: queued writes and lag on the shard boxes and charts |
| `04_multi_node_read_migrate.jsonl` | 3 shards, minilm | 🐘 elephant, 🛠️ admin, 🤖 probe-bot, 🧑‍🔧 ops | elephant's baseline answers; admin's early cutover is refused (409); admin migrates + waits while elephant demands IDENTICAL answers (`repeat until_signal "built"`); cutover (dense/hybrid: overlap ≥ 0.3/0.2, bm25 identical); rollback (identical again); cutover + finalize; rollback now refused. Admin and elephant take turns with `signal` / `wait_signal`; probe-bot searches every 50 ms the whole time | zero failed or partial queries across every flip; the router's model badge flips; the bge-small bars fill on every shard; the answer-checks panel shows `minilm → bge-small` overlaps and identical answers after rollback |
| `05_multi_node_read_write_migrate.jsonl` | 3 shards, minilm | 🦊 fox, 🐘 elephant, 🤖 bot-1 (writer), 🤖 bot-2 (reader), 🛠️ admin, 🧑‍🔧 ops | fox writes before the migration; bot-1 writes DURING the build; admin cuts over: elephant's dense search must find both (backfill + dual-write); rollback identical to the quiet point; cutover + finalize while bot-1 writes; everything still found | **dual-write**: docs written while bge-small was being built are found by dense search with bge-small; after finalize bge-small alone finds docs from every phase |
| `06_shard_failure.jsonl` | 3 shards, needs `--start-cluster` | 🦊 fox, 🐘 elephant, 🤖 bot-1 (reader), 🧑‍🔧 ops | one lane per actor, in turn via signals: ops kills shard 1 while bot-1 searches: elephant's searches are partial (`missing_shards: [1]`), its docs absent; fox's write owned by shard 1 fails (503 after retries), her write to shard 0 works; ops restarts it; answers identical; the failed write is absent until fox retries | the greyed-out shard, amber (soft) bot-1 cards during the outage, the restart card with the number of WAL records replayed. Outage searches take ~4 s on Windows (connecting to a dead port is slow) |
| `07_async_burst_crash.jsonl` | 3 shards, needs `--start-cluster` | 🤖 bot-1..3, 🧑‍🔧 ops, 🐘 elephant | the bots write 30 docs owned by shard 1 as fast as possible; ops kills it right after the last ack and restarts it; elephant searches every doc | every acked doc is present. Meaningful in **async** mode: acked-but-unapplied writes survive because the ack comes after the WAL fsync and the restart re-embeds them. In sync mode it is a control run. On tiny the queue is usually empty at the kill; on the real corpus writes can be caught in flight |

Migration on the tiny cluster: both "models" are hash embedders with different seeds, so the backfill
is instant but answers really change at cutover. On the real corpus `bge-small` comes from the
embedding cache `artifacts/emb_bge-small.npy` (the backfill reads cached vectors, so it is fast too;
queries and dual-writes need the model in the local Hugging Face cache).

---

## 6. Example results

Tiny corpus, every catalog scenario in both modes (from the verification run of this tool; your
numbers will differ slightly). All **PASS**:

| scenario | hard checks (sync / async) | notes |
|---|---|---|
| freshness_demo | 5 / 5 | immediate searches found the doc in both modes (hash apply takes µs) |
| idempotency | 13 / 13 | 9 writes, 4 applied, 5 no-ops as expected |
| 01_single_node_read (`--golden-save`) | 28 / 28 | same_as 12/12, search p50 ~4 ms |
| 02_multi_node_read (`--golden-check`) | 32 / 32 | same_as 12/12, **golden 4/4** (3 shards == 1 shard), search p50 ~11 ms |
| 03_multi_node_read_write | 68 / 68 | 123 writes; every burst's last doc VISIBLE after 26-39 ms (first poll). Fox's final searches say "searched ~2 s after its ack": they simply ran after the load phase, it is not a lag |
| 04_multi_node_read_migrate | 99 / 99 | same_as 33/33, overlap 6/6; admin and elephant take turns via signals, probe-bot never saw a failed query |
| 05_multi_node_read_write_migrate | 41 / 41 | dual-written docs found by dense search after cutover and finalize |
| 06_shard_failure | 26 / 26 | 7 partial searches (p95 ~4 s), 5 soft misses by the bot-1 reader during the outage |
| 07_async_burst_crash | 30 / 30 | 30/30 acked docs present after kill + restart |

Real corpus (57,638 docs, MiniLM; cluster start ~1-1.5 min):

| scenario | result |
|---|---|
| `freshness_demo`, **async** | PASS. Elephant's immediate **BM25** search, sent 0 ms after the ack, **MISSED** fox's doc (soft: the doc was still in its shard's queue being embedded); the immediate **hybrid** search, sent 15 ms after the ack, FOUND it at #1 (the router embeds the query first, which gave the applier time). Searched 1 s later: found. Same for the delete. `wait_visible`: VISIBLE after 37 ms (first poll) |
| `04_multi_node_read_migrate`, sync | PASS, 87 checks: same_as 25/25 (identical during the build and after rollback), overlap after cutover: q1 0.70, q2 0.30, q4 0.40 (a different embedding model really changes dense answers); BM25 identical throughout; zero failed or partial queries (probe-bot included) |

---

## 7. Writing your own scenario: tips

* Start with a `label` per phase: it becomes the narration in the replay.
* Declare your actors in `config.actors` (users as animals, bots, admin, ops) and give client ops in
  admin/ops lanes an explicit `"actor"`; the runner refuses admin/ops ops by a user or bot.
* Use `{run}` in doc ids when you target a long-lived cluster.
* Query with the doc's exact text when you expect it (deterministic in BM25 and dense).
* For async mode use `"soft": true` on checks right after a write, and hard checks after `sleep 1.0`,
  `wait_visible` or `flush`.
* Use `expect_same_as` only across stretches where nothing changes the ranking (no writes, same model).
* Continuous traffic: `{"op": "repeat", "until_done": "<other lane>", "ops": [...]}` in a parallel lane.
* Turn-taking between actors: one lane per actor; `signal` when something happened, `wait_signal` before
  reacting to it (see 04 and 06).
* Need ids on a given shard? `python scripts/scenario.py --shard-of a1 a2 a3 --shards 3`.
