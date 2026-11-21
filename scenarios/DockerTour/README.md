# Docker tour scenarios

These six scenarios are the scripted part of `scripts/docker_tour.ps1` / `scripts/docker_tour.sh`. The tour runs
the Docker steps of the manual tests (section 3 of [manual_tests_windows.md](../../docs/guides/manual_tests_windows.md)
and [manual_tests_mac.md](../../docs/guides/manual_tests_mac.md)) against the running stack, checks each result, and
writes an animated `replay.html` for every step. It also runs the eight [simple scenarios](../SimpleScenarios/README.md),
the CLI tools and the dev eval. At the end it puts the stack back in the shape it found it and verifies that, field by
field.

| file | guide | what it does and checks |
|---|---|---|
| `1_search_strategies.jsonl` | 3a | one question with bm25, dense and hybrid; every answer complete; hybrid finds 553031, 488556, 214174 |
| `2_writes.jsonl` | 3c | upsert, found at #1 right away (sync), delete, gone, then a late version 1 is a no-op |
| `3_migration.jsonl` | 3d | cutover refused, then migrate to bge-small, cut over, roll back, finalize, while probe-bot searches non-stop |
| `4_shard1_before.jsonl` | 3g | a doc on shard 1 and one on shard 0, before the outage |
| `5_shard1_down.jsonl` | 3g | with shard1 stopped: partial answers, the shard-1 write fails, the shard-0 write works |
| `6_shard1_back.jsonl` | 3g | shard1 restarted: complete answers, its doc is back (WAL replay), the failed write is retried |

`{onN:name}` in a doc id means "a doc that lives on shard N": `scripts/docker_tour.py` replaces it with an id that
hashes to that shard (`crc32(doc_id) % 3`), the same id in every file of one tour. `{run}` is the scenario runner's
usual per-run id. Steps 4 to 6 need the tour: they use pinned ids, and the wrapper scripts stop and start shard1 between
them. The tour needs the stack in sync mode and never changes it, so guide step 3i (the async mode) stays manual. Steps 1
to 3 also run on their own:

```bash
python scripts/scenario.py scenarios/DockerTour/2_writes.jsonl --url http://localhost:8000 --open
```
