# ch05 worksheet (pen and paper first)

BM25:  A 18.2   B 11.0   C 10.7   D 2.1
Dense: C 0.83   E 0.81   A 0.52   B 0.50

1. Naive sum of raw scores, ranked: ____________   Why is this "BM25 in disguise"? ____________
2. Min-max BM25: A ___ B ___ C ___ D ___      Min-max dense: C ___ E ___ A ___ B ___
3. Linear, α = 0.7 (weight on dense): C ___ E ___ A ___ B ___ D ___   → ranking ____________
4. Linear, α = 0.5: winner ____ with score ____
5. RRF (k = 60): A ___ C ___ B ___ E ___ D ___   Who ties? ____  How is the tie broken? ____
6. RRF with k = 1 instead of 60: does the A/C tie survive? ____
7. On FiQA, plain RRF (0.373) scored BELOW dense alone (0.388). Explain in one sentence: ____________

---
✅ Answers: [`solutions/ch05_fusion/ANSWERS.md`](../solutions/ch05_fusion/ANSWERS.md). Open it only *after* you've written yours.
