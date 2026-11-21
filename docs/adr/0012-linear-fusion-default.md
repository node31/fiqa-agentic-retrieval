# ADR 0012: Linear fusion with alpha=0.7 as the default hybrid

Status: accepted

## Context
Textbook RRF hurt on FiQA: 0.373 nDCG@10 vs 0.388 for dense alone.

## Decision
Linear min-max fusion with alpha (the dense weight) = 0.7, tuned on dev only.

## Alternatives considered
RRF with k=60 (0.373), and weighted RRF, whose best setting (alpha 0.9) gave 0.406. Both scored below linear fusion.

## Consequences
Best dev nDCG@10 is 0.419, 8% over dense. The risk is that I tuned and reported on the same split, which is optimistic. The single test-split run measures how optimistic: nDCG@10 0.3968. The full sweep is in [16_ablations.md](../16_ablations.md).
