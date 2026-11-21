# ADR 0006: Timeouts and partial results, flagged to the caller

Status: accepted

## Context
Shards can be slow or down, and the assignment asks me to decide what happens and document it.

## Decision
Every shard call has a 0.5 s timeout. The default policy `partial` returns merged results from the shards that answered, with `partial: true, missing_shards: [...]`. Policy `fail` returns 503. The eval harness treats a partial result as an error.

## Alternatives considered
Always failing, which turns one slow shard into an outage, and silently returning partial results, which hides missing data from the caller.

## Consequences
Search users get a degraded answer instead of an error, and callers that need completeness can choose `fail`. The flag is in every response, so a partial result is never silent. `tests/test_integration_service.py` tests this by killing a shard.
