# Architecture Decision Records

Short records of *why* each major choice was made. Format: context, decision, alternatives, consequences (with measured numbers where they exist).

- [0001-own-bm25](0001-own-bm25.md): Write BM25 ourselves on scipy sparse matrices
- [0002-dense-models](0002-dense-models.md): MiniLM primary, bge-small as the migration target
- [0003-merge-then-fuse](0003-merge-then-fuse.md): Fuse at the router after the global merge
- [0004-dfs-global-stats](0004-dfs-global-stats.md): Global BM25 statistics via a DFS round trip
- [0005-router-side-embedding](0005-router-side-embedding.md): Embed the query once at the router
- [0006-partial-results](0006-partial-results.md): Timeouts + partial results, flagged to the caller
- [0007-sync-apply-wal](0007-sync-apply-wal.md): Synchronous apply behind a fsynced WAL
- [0008-json-action-protocol](0008-json-action-protocol.md): Plain-JSON action protocol instead of native function calling
- [0009-agent-roles](0009-agent-roles.md): Code orchestrator + planner LLM + parallel search workers + verifier LLM
- [0010-slot-allocation](0010-slot-allocation.md): Round-robin interleave of sub-query lists for the 10 slots
- [0011-llm-backends-and-cost](0011-llm-backends-and-cost.md): Free backends for development, Azure only for final numbers
- [0012-linear-fusion-default](0012-linear-fusion-default.md): Linear fusion alpha=0.7 as the default hybrid
