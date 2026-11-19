# ADR 0006: Timeouts + partial results, flagged to the caller

**Status:** accepted

## Context
Shards can be slow or down; the assignment asks us to decide and document.

## Decision
Every shard call has a 0.5 s timeout. Default policy `partial`: return merged results from responsive shards with `partial: true, missing_shards: [...]`. Policy `fail` returns 503. Eval harness treats partial as an error.

## Alternatives considered
Always fail; silently return partial results.

## Consequences
Search UX prefers a degraded answer over an error; callers needing completeness can choose `fail`. Never silent: the flag is in every response. Tested by killing a shard in tests/test_integration_service.py.
