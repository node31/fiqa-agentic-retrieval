# Simple scenarios

Eight short stories in the CSV scenario format, each one step beyond the last. I use them to see one behaviour of
the service at a time before reading the longer JSONL scenarios in [../ComplexScenarios/](../ComplexScenarios/).
The format is explained in [docs/guides/scenarios.md](../../docs/guides/scenarios.md#simple-format-csv); `simple_template.csv` is a
copy-and-edit starting point that uses every operation.

| file | what happens | what to look for |
|---|---|---|
| `1_SimpleRead.csv` | Elephant searches "what is short selling?", Fox "should I pay off my mortgage early?" | the baseline: every search fans out to the 3 shards and is merged; the time split on each search card |
| `2_SimpleReadAfterWrite.csv` | + Owl upserts a doc, Fox searches for it | read-your-writes: Fox finds it at #1 |
| `3_SimpleReadAfterWriteAndDelete.csv` | + Owl deletes it, Fox searches three times expecting it gone (`!doc`) | the delete is visible to the next search |
| `4_ReadMigrateRead.csv` | the two searches, ADMIN `migrate`, `wait_ready`, `cutover` to bge-small, the two searches again | answers come from minilm until the cutover and from bge-small after it |
| `5_ReadWriteMigrateRead.csv` | Owl writes before the migration; after cutover Fox searches for it | bge-small finds a doc written when only minilm existed: the backfill embedded it |
| `6_ReadWriteMigrateWriteDuringMigrateRead.csv` | + Elephant writes between `migrate` and `cutover`; Fox searches for both docs | dual-write: during a migration each write is embedded with both models |
| `7_ReadWriteMigrateWriteDuringMigrateRollbackRead.csv` | + ADMIN rolls back; Fox searches for both docs again | after the rollback minilm answers and still finds both: nothing is lost |
| `8_Idempotency.csv` | 3 with explicit versions (column 6): a retry of v1, a late v1 after v2, a late v2 after the delete (v3), then v4 | the retry and both late writes are no-ops; the delete's tombstone stops the late v2 from bringing the doc back |

Each file's header has the run commands. In short (`python` = `.\.venv\Scripts\python` on Windows):

```bash
python scripts/scenario.py scenarios/SimpleScenarios/2_SimpleReadAfterWrite.csv --start-cluster --tiny --open
python scripts/scenario.py scenarios/SimpleScenarios/2_SimpleReadAfterWrite.csv --start-cluster --mode async --open
python scripts/run_simple_scenarios.py          # all eight on the four clusters below, ~17 min
```

## The four clusters

`scripts/run_simple_scenarios.py` runs each scenario on its own fresh 3-shard cluster (`--start-cluster`: local
processes on ports 19000/19101+, stopped after the run), so no run sees another run's writes or models.

| folder in `results/` | corpus | writes |
|---|---|---|
| `SimpleSyncTinyCluster` | the 24-doc test corpus with hash "embedders" (cluster start ~12 s) | sync: applied before the ack |
| `SimpleAsyncTinyCluster` | the same | async: acked once in the WAL, applied within 1 s |
| `SimpleSyncFullCluster` | the real FiQA corpus, 57,638 docs, MiniLM and bge-small (cluster start ~45 s) | sync |
| `SimpleAsyncFullCluster` | the same | async |

Each run leaves `results/<cluster>/<scenario>/` with `console.txt`, `events.jsonl` and `replay.html`, all checked in
(the cluster logs stay local). `results/SUMMARY.md` is the grid of all 32 runs. GitHub shows HTML as source, so to
watch a replay, download its `replay.html` (or clone the repo) and open it in a browser. It is one self-contained
file that works offline.

## Reading the scoreboard: hard and soft checks

Column 5 on a search (`doc` = expect it, `!doc` = expect it gone) is a soft check: the console says FOUND, MISSING or
ABSENT and the replay colours the hit, but a miss never fails the run. A hard check does fail the run when it misses. In a
CSV those are `wait_ready` (the new model must be ready on every shard before the timeout) and `wait_visible` /
`wait_gone`; in JSONL also `expect` without `"soft": true`, `expect_applied`, `expect_error`, `same_as` and the rest.

"Expectations passed" counts every check that held, hard or soft, so scenario 8 says
`expectations 5 passed (0 hard, 5 soft), 0 failed`. "Failed" counts only hard checks, and a soft check that missed is
listed separately as a soft miss. I kept the search checks soft because in async mode a search sent a few ms after a
write may legitimately miss it.

## What my runs showed

All 32 runs passed (`results/SUMMARY.md`). Every soft check held except one, on the async real-corpus cluster.

- **Models.** In 4 to 7 the searches before the cutover were answered by minilm and those after it by bge-small; in 7
  the two after the rollback by minilm again. The model is on every search line of the console and every search
  card.
- **Writes during a migration.** On the real corpus in sync mode, Elephant's write in scenario 6 took 343 ms to ack,
  against 31 ms for Owl's write before the migration, because each shard was loading bge-small and backfilling at
  that moment and a sync write waits for its embedding. The `migrate` call itself took 360 ms for the same reason. In
  async mode the same write acked in 7 ms (the ack only waits for the WAL). On the tiny corpus all of these are a few
  ms.
- **Async freshness.** In async scenario 6 on the real corpus, Fox's search for Owl's doc was sent 0 ms after the ack
  and missed it: the doc was still queued for embedding. The same search 1.1 s later found it at #1. That miss is the
  async contract working as documented ("searchable within 1 s", not "at once"), and it is why the search checks are
  soft. Every other search right after a write or a delete, in both async clusters, already saw the change: a search
  spends its first ~30 ms embedding the query and collecting BM25 statistics before it reaches the shards, which
  usually gives the applier enough time. In sync mode no search missed.
- **Idempotency.** In scenario 8 the retry, the late v1 and the late v2 all came back
  `no-op (stale_or_duplicate, current vN)`, on every cluster: 7 writes, 4 applied.

## Against the Docker stack

`--url http://localhost:8000` instead of `--start-cluster` runs a scenario against the running Docker stack. The
writes stay there (the ids contain the run id, so reruns never collide), and 4, 5 and 6 leave bge-small active. The
manual test guides ([Windows](../../docs/guides/manual_tests_windows.md), [macOS](../../docs/guides/manual_tests_mac.md)) show how to
put it back on minilm.
