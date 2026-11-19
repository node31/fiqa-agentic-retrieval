# ch05: Fusion yourself (min-max, linear, RRF)

Guide: `docs/from_zero/05_hybrid_and_reranking.md`.

> 🎯 **Why:** combining BM25 and dense is the cheapest quality gain in the project (+0.031 nDCG for ~2 ms). Getting
> fusion wrong (adding raw scores) silently gives you "BM25 in disguise". Reproducing toy 08's numbers proves you
> got it right.

## Steps
1. `PAPER.md`: the fusion tables by hand.
2. Implement in `my_fusion.py`: `minmax` → `linear` → `rrf`.
3. `python -m pytest exercises\ch05_fusion`.
4. `python exercises\ch05_fusion\my_fusion.py` prints toy 08's example with your code.

## Rules (so your output matches the reference exactly)
- Lists are `[(doc_id, score), ...]`, best first. Outputs are sorted by score descending, **ties by doc_id ascending**.
- `minmax`: if every score is equal, map them all to 1.0 (avoid dividing by zero).
- `linear`: `α·dense_norm + (1−α)·bm25_norm`; a doc missing from a list contributes 0 for that list.
- `rrf`: `Σ weight_i / (k + rank_i)`, ranks start at 1, default k = 60, default weights all 1.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch05_fusion` | toy 08's numbers + 200 random lists vs the reference | seconds |
| full | `python -m pytest exercises\ch05_fusion --full` | your `linear(α=0.7)` on the real BM25 and dense top-100 lists (500 dev queries) gives the shipped **0.419**, top-10s identical to the reference; your RRF gives **0.373**, *below* dense alone (0.388) | ~1 min first time (cached) |
