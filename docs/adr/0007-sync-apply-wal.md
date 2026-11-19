# ADR 0007: Synchronous apply behind a fsynced WAL

**Status:** accepted

## Context
B2 needs idempotent writes, lexical/dense consistency, a freshness guarantee and recovery.

## Decision
Version check -> embed outside lock -> write lock -> re-check -> WAL append+fsync -> apply to all indexes -> ack. Crash-only recovery: restart = base corpus + WAL replay.

## Alternatives considered
Async apply with a refresh interval (higher write throughput); apply-then-log.

## Consequences
Guarantee: searchable by the first query after the ack. Cost: write latency = one embedding per maintained model (~15-30 ms) + fsync. Apply-then-log can lose acked writes (toy 17 exercise).
