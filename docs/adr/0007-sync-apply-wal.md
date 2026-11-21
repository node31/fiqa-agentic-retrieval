# ADR 0007: Synchronous apply behind an fsynced WAL

Status: accepted

## Context
B2 needs idempotent writes, consistency between the lexical and dense indexes, a freshness guarantee and crash recovery.

## Decision
The write path is: version check, embed outside the lock, take the write lock, re-check the version, append to the WAL and fsync, apply to all indexes, ack. Recovery is crash-only: a restart loads the base corpus and replays the WAL.

## Alternatives considered
Async apply with a refresh interval gives higher write throughput. I later built it as a second, switchable mode for comparison ([ADR 0013](0013-async-apply-mode.md)); sync stays the default. I rejected apply-then-log because a crash between the ack and the log write loses an acked write.

## Consequences
A write is searchable by the first query after its ack. The cost is write latency: one embedding per maintained model (about 15 to 30 ms) plus the fsync.
