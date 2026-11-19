# ADR 0011: Free backends for development, Azure only for final numbers

**Status:** accepted

## Context
Azure tokens cost real money; development needs many LLM calls.

## Decision
fake (tests/toys/dry-run), copilot CLI (free dev evaluation), azure (final; --confirm-spend, --max-usd, disk cache, per-query Budget).

## Alternatives considered
Develop directly against Azure.

## Consequences
Near-zero spend during development; dev numbers on a different model are labelled as such and kept in a separate results file. Copilot backend latency is meaningless (CLI start-up), so latency/cost claims come from the Azure run.
