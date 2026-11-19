# ch02 worksheet: answers

> Fill in `exercises/ch02_metrics/PAPER.md` yourself **before** reading this.
> Run `python exercises\solutions\ch02_metrics\my_metrics.py` to see A, B and D computed by code.

**A.** Gold {A, B, C}, ranking [X, A, Y, Z, B, …], C missing.

| | working | value |
|---|---|---|
| Recall@10 | 2 found of 3 | **0.667** |
| RR | first hit (A) at rank 2 → 1/2 | **0.5** |
| DCG | A at rank 2 (0.631) + B at rank 5 (0.387) | **1.018** |
| IDCG | best case: 3 hits at ranks 1, 2, 3 = 1 + 0.631 + 0.5 | **2.131** |
| nDCG@10 | 1.018 / 2.131 | **0.478** |

**B.** Gold {A, B, C, D}, ranking [X, A, Y, B, Z, C, …], k = 5. Only ranks 1–5 count, so C (rank 6) is invisible.
- Recall@5 = 2/4 = **0.5**. RR@5 = 1/2 = **0.5**.
- DCG = 0.631 (A, rank 2) + 0.431 (B, rank 4) = 1.062.
- IDCG@5 with 4 gold docs = 1 + 0.631 + 0.5 + 0.431 = 2.562 (4 hits at ranks 1–4, **not** 5: there are only 4 gold docs).
- nDCG@5 = 1.062 / 2.562 = **0.414**.

**C.** Re-ordering the same top 10:
- **Recall@10 can't change**: it's a set count.
- **MRR can change** (where is the first hit?) and **nDCG can change** (every hit's position matters).

This is why nDCG is the headline metric: it rewards putting the right docs *first*.

**D.** Latencies 20 21 19 22 20 23 21 20 95 24 → sorted 19 20 20 20 21 21 22 23 24 95.
- mean = 285/10 = **28.5 ms**. No query actually took ~28 ms; the 95 ms outlier drags the mean up.
- p50 = value at position ceil(0.5·10) = 5 → **21 ms**.
- p95 = value at position ceil(0.95·10) = 10 → **95 ms**.

For the dashboard, show **p50 and p95** (the assignment's limit is on p95 ≤ 50 ms). p50 = the typical user, p95 = the
unlucky 1-in-20. The mean hides both. (With only 10 samples p95 is just the max; you need hundreds of queries for a stable p95.)

**E.** The fan-out waits for the slowest shard, so a query is fast only if all 3 shards are fast: 0.9³ = 0.729.
**1 − 0.9³ = 0.271 → 27% of queries are slow**, almost 3× the single-shard rate. This is *tail-latency amplification*.
It's why our 3-shard p95 (40.5 ms) is worse than a single node's (24.5 ms) even though each shard does less work.

**F.** Halves a = {a1}, b = {b1, b2}. Top-3 [a1, x, b2]: both halves represented → **Coverage@3 = 1.0**.
Top-2 [a1, x]: half b missing → **0.0**. Coverage is all-or-nothing per question; averaged over 100 questions
it becomes "the share of two-part questions where *both* parts got at least one good doc".
