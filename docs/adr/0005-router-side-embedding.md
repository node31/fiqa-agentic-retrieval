# ADR 0005: The router embeds the query once

Status: accepted

## Context
Dense search needs the query vector on every shard.

## Decision
The router encodes the query in a thread, overlapped with the DFS phase, and sends the vector plus the model id to the shards.

## Alternatives considered
Each shard encodes the query itself. I rejected it because it repeats the same encode N times.

## Consequences
One encode instead of N, and the router controls which model a request uses: it captures the model once per request, which keeps migration safe. The cost is that the router holds the models in memory and becomes a CPU hot spot under concurrency (visible in the load test p50 of ~140 ms).
