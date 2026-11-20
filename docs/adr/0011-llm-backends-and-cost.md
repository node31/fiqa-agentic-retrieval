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

**Measured outcome:** the whole final Azure eval (600 system-query runs) cost ≈ 655K tokens, $0.79 at placeholder prices
($1 / $4 per 1M input / output tokens); the `--dry-run` estimate was $0.47, because the fake policy makes fewer calls. The
dev model (haiku) led to a different routing decision than gpt-6-sol, so dev numbers guide the plumbing, not the final call.
