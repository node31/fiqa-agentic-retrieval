# ch02: Write the metrics yourself

Guide: `docs/from_zero/02_measuring_quality_and_speed.md`.

> 🎯 **Why:** every decision in the project was made by comparing these numbers. If your ruler is wrong, every
> decision built on it is wrong, so you'll test your ruler against hand-computed answers *and* the reference.

## Steps
1. `PAPER.md`: compute every number by hand first.
2. Implement in `my_metrics.py`, in order: `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `coverage_at_k`, `percentile`, `evaluate`.
3. `python -m pytest exercises\ch02_metrics` (or one at a time: `-k recall`, `-k ndcg`, ...).
4. `python exercises\ch02_metrics\my_metrics.py` runs the sanity demo: an oracle run (should be 1.0) and a random run (≈ 0).

## What the checker tests
- the worked examples from the guide (§2.2–§2.5) and toy 07 (k = 5: nDCG 0.613 / 0.501 / 0.000)
- edge cases: duplicate ids in a ranking, a query with no gold docs, k larger than the list
- 300 random rankings compared with the reference `src/metrics.py` (catches subtle off-by-ones)

## Hints
- Dedupe the ranking (keep the first occurrence) before cutting to k.
- nDCG: gain = `2**rel - 1`, discount = `1 / log2(rank + 1)` with ranks starting at 1.
  IDCG uses **all** gold docs (sorted by rel, cut to k), not only the ones you found.
- `percentile`: nearest rank, `sorted(v)[ceil(p/100 * n) - 1]` (at least index 0).

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch02_metrics` | hand examples, edge cases, 300 random rankings vs the reference | seconds |
| full | `python -m pytest exercises\ch02_metrics --full` | your `evaluate()` on a REAL run (reference BM25, 500 dev queries) equals the reference to 1e-9 and gives BM25's 0.264 | ~1 min the first time (builds the reference BM25 run, cached), then ~10 s |
