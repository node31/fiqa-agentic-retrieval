# ADR 0005: Embed the query once at the router

**Status:** accepted

## Context
Dense search needs the query vector on every shard.

## Decision
Router encodes (in a thread, overlapped with the DFS phase) and sends the vector + model id.

## Alternatives considered
Each shard encodes the query itself.

## Consequences
+ 1 encode instead of N; the router controls which model a request uses (migration safety: model captured once per request). − the router needs the models in memory and is a CPU hot spot under concurrency (visible in the load test p50 of ~140 ms).
