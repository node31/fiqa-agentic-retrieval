# ADR 0002: MiniLM primary, bge-small as the migration target

**Status:** accepted

## Context
Two of {MiniLM, mpnet, bge-small, bge-base} are required. 50 ms p95 budget on CPU, through a router.

## Decision
all-MiniLM-L6-v2 serves first; B3 migrates to bge-small-en-v1.5. Both 384-d, max_seq_length 256.

## Alternatives considered
mpnet/bge-base (768-d, 12 layers, ~2x slower encode on CPU); bge-small as primary.

## Consequences
Dense: bge-small +0.010 nDCG over MiniLM (0.398 vs 0.388), but +12 ms per query (30.6 vs 18.8 ms p50). In hybrid the gap vanishes (0.416 vs 0.419). MiniLM leaves headroom for the HTTP fan-out (svc3 p95 40.5 ms). The migration is therefore a demonstration of mechanics, not a quality win, which is stated honestly in DESIGN.md.
