# ADR 0009: Code orchestrator, planner LLM, parallel search workers, verifier LLM

Status: accepted

## Context
C3 requires at least 2 sub-agents with distinct roles, guardrails and tolerance of partial failure.

## Decision
The orchestrator is code: it owns the budget, timeouts and slots. The planner LLM decides whether and how to split a question, and a free gate skips it for simple questions. The workers are parallel tool calls with no LLM. One batched verifier call checks the results and can trigger one re-search.

## Alternatives considered
| Option | Why I rejected it |
|---|---|
| LLM orchestrator | It costs calls and can loop; budgets, timeouts and slots must be deterministic and testable |
| An LLM per worker | One more call per sub-query for little gain: the planner already wrote good queries |
| No verifier | Nothing would catch the main failure, a sub-query that retrieved off-topic results |

## Consequences
The happy path makes at most 2 LLM calls (planner and verifier), and 0 for questions the gate marks simple. The control flow is deterministic and unit-testable. Workers without an LLM save one call per sub-query.

Measured outcome (Azure gpt-6-sol, `results/agent_eval.json`): on compound questions multi used 925 tokens and p95 4.7 s, vs the single agent's 3,380 tokens and 9.3 s. Quality was lower (R@10 0.355 vs 0.400, Coverage 0.32 vs 0.40): the single agent's final LLM rerank helps with a strong model, and in this design the LLM never ranks. So multi ships as the compound path under a latency or cost budget, and it is not the default (DESIGN.md, "Agent evaluation and routing").
