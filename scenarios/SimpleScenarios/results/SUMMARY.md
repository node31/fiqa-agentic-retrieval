# Simple scenarios: recorded runs

Written by `python scripts/run_simple_scenarios.py` on 2026-11-21. Each cell: result · soft checks passed/total · which model answered the searches, in order. A soft check is a search that expected a doc (found) or expected it gone (`!doc`); a miss is reported but never fails the run. The console output of every run is in `<cluster>/<scenario>/console.txt`; rerun a scenario to get its `replay.html`.

| scenario | SimpleSyncTinyCluster | SimpleAsyncTinyCluster | SimpleSyncFullCluster | SimpleAsyncFullCluster |
|---|---|---|---|---|
| 1_SimpleRead | PASS · soft 0/0 · minilm×2 | PASS · soft 0/0 · minilm×2 | PASS · soft 0/0 · minilm×2 | PASS · soft 0/0 · minilm×2 |
| 2_SimpleReadAfterWrite | PASS · soft 1/1 · minilm×3 | PASS · soft 1/1 · minilm×3 | PASS · soft 1/1 · minilm×3 | PASS · soft 1/1 · minilm×3 |
| 3_SimpleReadAfterWriteAndDelete | PASS · soft 4/4 · minilm×6 | PASS · soft 4/4 · minilm×6 | PASS · soft 4/4 · minilm×6 | PASS · soft 4/4 · minilm×6 |
| 4_ReadMigrateRead | PASS · soft 0/0 · minilm×2 → bge-small×2 | PASS · soft 0/0 · minilm×2 → bge-small×2 | PASS · soft 0/0 · minilm×2 → bge-small×2 | PASS · soft 0/0 · minilm×2 → bge-small×2 |
| 5_ReadWriteMigrateRead | PASS · soft 2/2 · minilm×3 → bge-small×1 | PASS · soft 2/2 · minilm×3 → bge-small×1 | PASS · soft 2/2 · minilm×3 → bge-small×1 | PASS · soft 2/2 · minilm×3 → bge-small×1 |
| 6_ReadWriteMigrateWriteDuringMigrateRead | PASS · soft 3/3 · minilm×3 → bge-small×2 | PASS · soft 3/3 · minilm×3 → bge-small×2 | PASS · soft 3/3 · minilm×3 → bge-small×2 | PASS · soft 2/3 · minilm×3 → bge-small×2 |
| 7_ReadWriteMigrateWriteDuringMigrateRollbackRead | PASS · soft 5/5 · minilm×3 → bge-small×2 → minilm×2 | PASS · soft 5/5 · minilm×3 → bge-small×2 → minilm×2 | PASS · soft 5/5 · minilm×3 → bge-small×2 → minilm×2 | PASS · soft 5/5 · minilm×3 → bge-small×2 → minilm×2 |
| 8_Idempotency | PASS · soft 5/5 · minilm×7 · 3 no-op writes | PASS · soft 5/5 · minilm×7 · 3 no-op writes | PASS · soft 5/5 · minilm×7 · 3 no-op writes | PASS · soft 5/5 · minilm×7 · 3 no-op writes |

## Soft misses

Searches whose expectation did not hold. In async mode a search sent a few ms after a write can miss it (the write is acked once it is in the WAL and applied within 1 s); in sync mode there should be none.

- SimpleAsyncFullCluster / 6_ReadWriteMigrateWriteDuringMigrateRead, line 25: Fox's search expected `owl-annuity-61a3ef`, missed it

## Wall time per cluster (including cluster start)

- SimpleSyncTinyCluster: 105 s
- SimpleAsyncTinyCluster: 104 s
- SimpleSyncFullCluster: 383 s
- SimpleAsyncFullCluster: 421 s
