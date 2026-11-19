# ch09: Rebuild the whole project yourself

Guide: `docs/from_zero/09_rebuild_it_yourself.md` (18 milestones, M0–M17, each with a "why", a skeleton,
and the numbers to match).

The toy exercises (ch01–ch08) live here, inside the repo. The **full rebuild is better done in a fresh, empty repo**
of your own (e.g. `my-fiqa/`, layout in the guide), for two reasons:
1. You'll write the code from an empty folder, with no reference code one `Ctrl+P` away.
2. You can compare *your* numbers with *ours* (`results/`) as an independent check.

**Shortcut:** your ch01–ch08 files are the seeds of milestones M1–M7, M10, M12 and M16. Copy them into the new repo as your starting point.

Track your progress here:

| milestone | done? | your number | ours |
|---|---|---|---|
| M1 loader stats | ☐ | | 57,638 / 500 / 648 |
| M3 BM25 dev nDCG@10 | ☐ | | 0.264 |
| M6 dense MiniLM dev nDCG@10 | ☐ | | 0.388 |
| M7 hybrid linear α=0.7 dev nDCG@10 | ☐ | | 0.419 |
| M8 hybrid warm p95 (single node) | ☐ | | 24.5 ms |
| M9 test split (once) | ☐ | | 0.397 |
| M10/M11 3-shard nDCG equals single node | ☐ | | 0.4190 = 0.4190 |
| M13 failed queries during migration | ☐ | | 0 |
| M14 Docker eval nDCG@10 | ☐ | | 0.4190 |
| M16 compound Recall@10 (multi-agent) | ☐ | | 0.335 (baseline 0.204) |
