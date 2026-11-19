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
