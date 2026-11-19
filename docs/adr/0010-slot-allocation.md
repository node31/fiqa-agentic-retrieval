# ADR 0010: Round-robin interleave of sub-query lists for the 10 slots

**Status:** accepted

## Context
Several sub-queries compete for one top-10; scores across sub-queries are not comparable.

## Decision
Interleave A1,B1,A2,B2...; dedupe; fill leftovers from the full-question search.

## Alternatives considered
Score merge; RRF across sub-lists; planner-weighted quotas.

## Consequences
Guarantees Coverage@10 whenever every worker found something (toy 25); costs each half its ranks 6-10. Planner capped at 3 sub-queries so each keeps >= 3 slots.
