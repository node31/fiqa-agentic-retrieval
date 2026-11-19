# ch02 worksheet (pen and paper first)

Discount table to use: rank 1 → 1.000, 2 → 0.631, 3 → 0.500, 4 → 0.431, 5 → 0.387, 6 → 0.356.

**A.** Gold = {A, B, C}. Ranking = [X, A, Y, Z, B, ...] (C not in the top 10).
- Recall@10 = ____   RR = ____   DCG = ____   IDCG = ____   nDCG@10 = ____

**B.** Gold = {A, B, C, D}. Ranking = [X, A, Y, B, Z, C, ...]. Use k = 5.
- Recall@5 = ____   RR@5 = ____   nDCG@5 = ____

**C.** The same 10 docs, re-ordered. Which of Recall@10 / MRR / nDCG can change? ____________

**D.** Latencies (ms): 20 21 19 22 20 23 21 20 95 24
- mean = ____   p50 = ____   p95 = ____   Which one would you put on a dashboard, and why? ________

**E.** 3 shards, each slow 10% of the time. What fraction of fan-out queries are slow? ____

**F.** Compound question, halves a = {a1}, b = {b1, b2}. Top-3 = [a1, x, b2]. Coverage@3 = ____. Top-2? ____

---
✅ Answers: [`solutions/ch02_metrics/ANSWERS.md`](../solutions/ch02_metrics/ANSWERS.md). Open it only *after* you've written yours.
