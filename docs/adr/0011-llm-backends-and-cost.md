# ADR 0011: Free backends for development, Azure only for final numbers

Status: accepted

## Context
Azure tokens cost real money, and development needs many LLM calls.

## Decision
I use three backends. `fake` serves tests and dry runs. `copilot` (the Copilot CLI) gives free dev evaluation. `azure` produces the final numbers and requires `--confirm-spend`, with `--max-usd`, a disk cache and a per-query `Budget` as guards.

## Alternatives considered
Developing directly against Azure, which would spend money on every iteration.

## Consequences
Spend during development is near zero. Dev numbers come from a different model, so I label them as dev and keep them in a separate results file. Copilot backend latency is meaningless (CLI start-up), so every latency and cost claim comes from the Azure run.

Measured outcome: the whole final Azure eval (600 system-query runs) cost ≈ 655K tokens, $0.79 at placeholder prices ($1 / $4 per 1M input / output tokens). The `--dry-run` estimate was $0.47, because the fake policy makes fewer calls. The dev model (haiku) led to a different routing decision than gpt-6-sol, so I used dev numbers to get the plumbing right and made the final call on the Azure numbers.
