# ADR 0004: Global BM25 statistics via a DFS round trip

Status: accepted

## Context
With shard-local IDF and avgdl, BM25 scores from different shards are not comparable.

## Decision
In phase 1 the router asks every shard for N, total length and df of the query terms, and sums them. In phase 2 it sends the global stats with the query.

## Alternatives considered
Local stats (the Elasticsearch query_then_fetch default), and a global df cache at the router, refreshed periodically.

## Consequences
I measured local IDF at nDCG@10 0.2653 vs 0.2642 global, which is noise, and local saves 4 ms (6.8 vs 10.7 ms p50). I kept DFS anyway because it gives exact single-node equivalence (the 1% rule) and stays correct on skewed shards. A cached-df variant is the obvious production optimisation; I have not built it.
