# 05 · Hybrid retrieval: fusing BM25 and dense

**Code:** `src/retrieval/hybrid.py` · **Toys:** 08, 13 · **Results:** `results/ablation_fusion_*.json`

## 1. Intuition

BM25 and dense fail differently. BM25 misses paraphrases; dense misses exact rare terms (tickers,
form numbers, names) and can drift to "topically similar but wrong" passages. If a passage is
ranked high by **both**, that's strong evidence. Fusion turns two ranked lists into one.

## 2. The math

**RRF** (rank-based): score(d) = Σ_lists w_i/(k + rank_i(d)), with k = 60. Docs missing from a list
contribute 0 from it. Scale-free.

**Linear** (score-based): normalise each list with min-max, s' = (s − min)/(max − min), then
score(d) = α·dense'(d) + (1 − α)·bm25'(d). Needs α tuned.

**Worked example** (toy 08): BM25 [A 18.2, B 11.0, C 10.7, D 2.1], dense [C 0.83, E 0.81, A 0.52, B 0.50].
- Naive raw sum: A 18.7, B 11.5, C 11.5, … (BM25's scale dominates; E, dense's #2, ends last).
- RRF: C = 1/63 + 1/61 = 0.0323 → **C first** (both systems like it).
- Linear α = 0.5: bm25' = [A 1, B 0.56, C 0.54, D 0], dense' = [C 1, E 0.94, A 0.06, B 0] → C 0.77, A 0.53, E 0.47…

## 3. Diagram

```
BM25 top-100 ──┐
               ├── fuse (RRF or α-linear) ──> top-10
dense top-100 ─┘        ↑ at the ROUTER, after the global merge (ADR 0003)
```

## 4. How this repo implements it

- `rrf(lists, k, weights)`, `minmax`, `linear`, `fuse(method, alpha)`; deterministic tie-break.
- In `fuse`, **α is always the weight on dense**: linear uses α/(1 − α), RRF uses list weights (1 − α, α).
  α = 0.5 RRF is the textbook unweighted RRF.
- Candidate depth 100 per list (`config.CANDIDATE_DEPTH`).

## 5. Run the toys

```bash
python examples/08_fusion_rrf_vs_linear.py
python examples/13_merge_then_fuse.py
python scripts/ablations.py --model minilm      # the real sweep on dev (~1 min)
```

## 6. Observe it in the real system: dev sweep, MiniLM (`results/ablation_fusion_minilm.json`)

| method | α (dense weight) | nDCG@10 | R@10 | MRR@10 |
|---|---|---|---|---|
| dense only | – | 0.388 | 0.467 | 0.453 |
| BM25 only | – | 0.264 | 0.314 | 0.322 |
| RRF | 0.5 (standard) | **0.373** ⚠ below dense | 0.442 | 0.449 |
| RRF | 0.9 | 0.406 | 0.481 | 0.479 |
| linear | 0.5 | 0.402 | 0.467 | 0.473 |
| **linear** | **0.7 (chosen)** | **0.419** | **0.491** | **0.487** |
| linear | 0.9 | 0.401 | 0.472 | 0.469 |

**Counterintuitive:** textbook RRF *hurts* (−0.015 vs dense alone). RRF gives a strong and a weak
retriever equal votes. When BM25 is 12 points worse, its votes are mostly noise. Linear fusion with
α = 0.7 respects both the *quality gap* (via α) and score *magnitudes* (a close cosine race stays close),
and gains **+0.031 nDCG (+8% relative)** over dense. bge-small shows the same shape (best: linear 0.7 → 0.416).

⚠ α was tuned on dev and these numbers are reported on dev, so they're optimistic. That's why the single test run matters ([DESIGN.md](../DESIGN.md)).

## 7. Pitfalls and interview questions

- **Fusing per shard, then merging** is wrong for score-based fusion (toy 13, [ADR 0003](adr/0003-merge-then-fuse.md)).
- **Min-max is outlier-sensitive**: one huge BM25 score squashes the rest to ~0.
- **Depth matters**: a doc missing from one list gets 0 from it. Deeper lists = fairer fusion, but more cost.
- *Q: When is RRF the right default?* When you can't tune (no labels), or when the systems are of similar quality.

## 8. Exercises

1. (Concept) Why is RRF invariant to multiplying one system's scores by 1000?
2. (Predict) What happens to linear fusion if BM25 returns only 3 docs for a query?
3. (Do) Implement z-score fusion and add it to the sweep. Does it beat min-max?
4. (Do) Find the α at which linear hybrid drops below dense alone (both sides of the peak).

Solutions: [solutions/05_hybrid.md](solutions/05_hybrid.md)

## 9. Further reading

- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion* (SIGIR 2009)
- Bruch, Gai, Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval* (2022): explains why
  tuned convex combination tends to beat RRF, which is what we observed.
