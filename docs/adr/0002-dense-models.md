# ADR 0002: MiniLM serves first, bge-small is the migration target

Status: accepted

## Context
The assignment requires two of {MiniLM, mpnet, bge-small, bge-base}. My budget is 50 ms p95 on CPU, through a router.

## Decision
all-MiniLM-L6-v2 serves first, and in B3 I migrate to bge-small-en-v1.5. Both are 384-d with max_seq_length 256.

## Alternatives considered
I rejected mpnet and bge-base because they are 768-d with 12 layers and encode about 2x slower on CPU. I also rejected bge-small as the primary model (see the numbers below).

## Consequences
Dense alone, bge-small beats MiniLM by 0.010 nDCG (0.398 vs 0.388) but costs 12 ms more per query (30.6 vs 18.8 ms p50). In hybrid the gap vanishes (0.416 vs 0.419). MiniLM leaves headroom for the HTTP fan-out (3 shards: p95 40.5 ms). So the migration demonstrates the mechanics and brings no quality gain, and DESIGN.md says so.
