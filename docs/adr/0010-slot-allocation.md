# ADR 0010: Round-robin interleave of sub-query lists for the 10 slots

Status: accepted

## Context
Several sub-queries compete for one top 10, and scores from different sub-queries are not comparable.

## Decision
I interleave the lists (A1, B1, A2, B2, ...), dedupe, and fill leftover slots from the search on the full question.

## Alternatives considered
A score merge, RRF across sub-lists, and planner-weighted quotas. I rejected the score merge because scores from different sub-queries are not comparable, so the easy half takes all 10 slots and Coverage@10 drops. The interleave is simpler than RRF or quotas and guarantees every half a share of the slots.

## Consequences
Coverage@10 is guaranteed whenever every worker found something. Each half loses its ranks 6 to 10. I cap the planner at 3 sub-queries so each keeps at least 3 slots.
