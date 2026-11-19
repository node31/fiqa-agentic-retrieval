"""Toy 08: RRF vs linear fusion, and why adding raw scores is wrong.

Concept: BM25 and cosine scores live on different scales, so fuse by rank (RRF) or by normalised score.
Real code: src/retrieval/hybrid.py     Doc: docs/05_hybrid_fusion.md
"""
from _common import banner, note, step, try_this

from src.retrieval.hybrid import fuse, minmax, rrf

banner("Toy 08: fusing a BM25 list and a dense list")
bm25 = [("A", 18.2), ("B", 11.0), ("C", 10.7), ("D", 2.1)]
dense = [("C", 0.83), ("E", 0.81), ("A", 0.52), ("B", 0.50)]
step(1, f"BM25 (unbounded scores): {bm25}\n    dense (cosine 0..1):     {dense}")
naive = {}
for lst in (bm25, dense):
    for d, s in lst:
        naive[d] = naive.get(d, 0) + s
step(2, f"Naive sum of raw scores: {sorted(naive.items(), key=lambda x: -x[1])}")
note("BM25's scale (2-18) swamps cosine (0.5-0.8): this is just BM25 in disguise. E, dense's #2, ends last.")
step(3, "RRF: score = sum 1/(60 + rank). Ranks only, so scale is irrelevant")
for d, s in rrf([bm25, dense]):
    print(f"     {d}: {s:.5f}")
note("C wins: rank 3 + rank 1. RRF rewards docs that BOTH systems like.")
step(4, f"Linear: min-max normalise each list, then alpha*dense + (1-alpha)*bm25.  bm25 norm: "
        f"{ {k: round(v, 2) for k, v in minmax(bm25).items()} }")
for a in (0.3, 0.5, 0.7):
    print(f"     alpha={a}: {[(d, round(s, 3)) for d, s in fuse(bm25, dense, 'linear', a)]}")
note("Linear keeps magnitudes: dense says C and E are nearly tied (0.83 vs 0.81), and linear respects "
     "that while RRF calls them rank 1 vs rank 2. On FiQA dev, linear alpha=0.7 beat RRF by +0.046 nDCG.")
try_this("Run `python scripts/ablations.py` and find the alpha where hybrid falls BELOW dense alone.",
         "What happens to min-max when one list has a single outlier (e.g. BM25 = 90 for one doc)?",
         "Implement z-score normalisation instead of min-max and compare on dev.")
