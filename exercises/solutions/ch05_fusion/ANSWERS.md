# ch05 worksheet: answers

> Fill in `exercises/ch05_fusion/PAPER.md` yourself **before** reading this.
> Run `python exercises\solutions\ch05_fusion\my_fusion.py` to print every number below.

BM25: A 18.2 · B 11.0 · C 10.7 · D 2.1 Dense: C 0.83 · E 0.81 · A 0.52 · B 0.50

**1. Naive sum.** A 18.72, C 11.53, B 11.50, D 2.10, E 0.81 → **A, C, B, D, E**.
This is exactly the BM25 order with tiny nudges. BM25 scores are on a 0–20 scale and cosines on 0–1, so the dense
signal is a rounding error. E, the dense model's #2, comes last. Different scales must be normalised before adding.

**2. Min-max.**
- BM25 (min 2.1, max 18.2, range 16.1): A **1.0**, B (11.0 − 2.1)/16.1 = **0.553**, C (10.7 − 2.1)/16.1 = **0.534**, D **0.0**.
- Dense (min 0.50, max 0.83, range 0.33): C **1.0**, E (0.81 − 0.50)/0.33 = **0.939**, A 0.02/0.33 = **0.061**, B **0.0**.

**3. Linear, α = 0.7** (score = 0.7·dense + 0.3·bm25; a doc missing from a list gets 0 from it):

| doc | dense part | BM25 part | total |
|---|---|---|---|
| C | 0.7·1.0 | 0.3·0.534 | **0.860** |
| E | 0.7·0.939 | 0.3·0 (not in BM25) | **0.658** |
| A | 0.7·0.061 | 0.3·1.0 | **0.342** |
| B | 0.7·0 | 0.3·0.553 | **0.166** |
| D | 0.7·0 (not in dense) | 0.3·0 | **0.0** |

Ranking **C, E, A, B, D**. C is good in *both* lists, so it wins clearly.

**4. α = 0.5:** **C wins with 0.767** (then A 0.530, E 0.470, B 0.276, D 0). With more weight on BM25, A climbs above E.

**5. RRF, k = 60** (1/(60 + rank) per list, summed):

| doc | BM25 part | dense part | total |
|---|---|---|---|
| A | 1/61 (rank 1) | 1/63 (rank 3) | **0.03227** |
| C | 1/63 (rank 3) | 1/61 (rank 1) | **0.03227** |
| B | 1/62 (rank 2) | 1/64 (rank 4) | **0.03175** |
| E | – | 1/62 (rank 2) | **0.01613** |
| D | 1/64 (rank 4) | – | **0.01562** |

**A and C tie** (ranks 1+3 vs 3+1). Our code breaks ties by **doc id** (A before C), so the order is
deterministic and reproducible. RRF ignores score *sizes*: A's huge 18.2 counts the same as any other #1.

**6. k = 1:** A = 1/2 + 1/4 = 0.75 and C = 1/4 + 1/2 = 0.75: **the tie survives** for every k. Swapping two ranks
between two lists always gives the same sum. k only changes how much rank 1 beats rank 2 (small k = very top-heavy).

**7.** RRF weights both lists equally and throws away the scores, so the much weaker BM25 list (0.264) gets the same say as
dense (0.388) and drags it down. Linear fusion with α = 0.7 lets dense dominate while still adding BM25's exact-match hits → 0.419.
(Check it yourself on the real data: `python -m pytest exercises\ch05_fusion --full`.)
