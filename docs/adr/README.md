# Architecture decision records

Each record says why I made one major choice: the context, my decision, what I rejected and why, and what it cost, with measured numbers where I have them.

| ADR | Decision |
|---|---|
| [0001](0001-own-bm25.md) | I wrote BM25 myself on scipy sparse matrices |
| [0002](0002-dense-models.md) | MiniLM serves first, bge-small is the migration target |
| [0003](0003-merge-then-fuse.md) | The router fuses after the global merge |
| [0004](0004-dfs-global-stats.md) | Global BM25 statistics via a DFS round trip |
| [0005](0005-router-side-embedding.md) | The router embeds the query once |
| [0006](0006-partial-results.md) | Timeouts and partial results, flagged to the caller |
| [0007](0007-sync-apply-wal.md) | Synchronous apply behind an fsynced WAL |
| [0008](0008-json-action-protocol.md) | A plain-JSON action protocol instead of native function calling |
| [0009](0009-agent-roles.md) | Code orchestrator, planner LLM, parallel search workers, verifier LLM |
| [0010](0010-slot-allocation.md) | Round-robin interleave of sub-query lists for the 10 slots |
| [0011](0011-llm-backends-and-cost.md) | Free backends for development, Azure only for final numbers |
| [0012](0012-linear-fusion-default.md) | Linear fusion with alpha=0.7 as the default hybrid |
| [0013](0013-async-apply-mode.md) | Async apply as a switchable second write mode, with the sync vs async load-test results |
