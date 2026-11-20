# ADR 0009: Code orchestrator + planner LLM + parallel search workers + verifier LLM

**Status:** accepted

## Context
C3 requires >= 2 sub-agents with distinct roles, guardrails and partial-failure tolerance.

## Decision
Orchestrator in code (budget, timeouts, slots); planner decides/decomposes (skipped by a free gate); workers are parallel tool calls; one batched verifier call can trigger one re-search.

## Alternatives considered
LLM orchestrator; an LLM per worker; no verifier.

## Consequences
Max 2 LLM calls on the happy path (planner + verifier), 0 for gated-simple questions; deterministic control flow is unit-testable. Workers without an LLM save a call per sub-query.

**Measured outcome (Azure gpt-6-sol, `results/agent_eval.json`):** on compound questions multi used 925 tokens and p95 4.7 s,
vs the single agent's 3,380 tokens and 9.3 s. Quality was lower (R@10 0.355 vs 0.400, Coverage 0.32 vs 0.40): the single
agent's final LLM rerank helps with a strong model, and in this design the LLM never ranks. So multi ships as the
budget-constrained compound path, not the default (DESIGN.md §7).
