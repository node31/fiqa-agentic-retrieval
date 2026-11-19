# ADR 0012: Linear fusion alpha=0.7 as the default hybrid

**Status:** accepted

## Context
Textbook RRF hurt on FiQA (0.373 vs dense 0.388).

## Decision
Linear min-max fusion, alpha (dense weight) = 0.7, tuned on dev only.

## Alternatives considered
RRF k=60; weighted RRF (alpha 0.9 gave 0.406).

## Consequences
Best dev nDCG@10 0.419 (+8% over dense). Risk: tuned and reported on the same split (optimistic) - the single test-split run quantifies it.
