# ADR 0003: The router fuses after the global merge

Status: accepted

## Context
Hybrid search combines a BM25 list and a dense list, and each shard sees only a slice of the corpus.

## Decision
Each shard returns its top-100 BM25 and top-100 dense lists separately. The router merges each list globally, then fuses.

## Alternatives considered
Fusing on each shard and merging the fused scores is cheaper on the network. I rejected it because per-shard min-max normalisation inflates each shard's local winners, so fused scores from different shards are not comparable.

## Consequences
Results equal single-node exactly (verified: 0.0% difference). The cost is 2x the candidate payload per shard (about 2 × 100 × [id, score], roughly 6 KB).
