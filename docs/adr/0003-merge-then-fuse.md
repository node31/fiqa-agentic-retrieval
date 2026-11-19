# ADR 0003: Fuse at the router after the global merge

**Status:** accepted

## Context
Hybrid needs BM25 and dense lists combined; shards each see a slice.

## Decision
Shards return top-100 BM25 and top-100 dense separately; the router merges each globally, then fuses.

## Alternatives considered
Fuse on each shard and merge fused scores (cheaper on the network).

## Consequences
+ exactly equals single-node (verified: 0.0% delta). − 2x the candidate payload per shard (~2 x 100 x [id, score] ~ 6 KB). Per-shard min-max normalisation would inflate each shard's local winners (toy 13).
