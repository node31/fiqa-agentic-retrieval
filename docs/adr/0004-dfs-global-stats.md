# ADR 0004: Global BM25 statistics via a DFS round trip

**Status:** accepted

## Context
Shard-local IDF/avgdl make scores incomparable across shards.

## Decision
Phase 1: router asks all shards for N, total length, df(query terms); sums; phase 2 sends global stats with the query.

## Alternatives considered
Local stats (ES query_then_fetch default); periodically cached global df at the router.

## Consequences
Measured: local-IDF nDCG@10 0.2653 vs global 0.2642 (noise); local saves 4 ms (6.8 vs 10.7 ms p50). Kept DFS for exact single-node equivalence (1% rule) and robustness to skewed shards. A cached-df variant is the obvious production optimisation (drill).
