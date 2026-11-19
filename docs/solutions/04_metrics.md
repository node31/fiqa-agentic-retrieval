# Solutions: metrics

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Yes: relevant = {g1, g2, g3, g4}, ranked = [g1, x, x, …]. MRR = 1, Recall@10 = 1/4 = 0.25.

**2.** With exactly 1 relevant doc at rank r ≤ 10: nDCG = 1/log2(r+1), MRR = 1/r. Both decrease with r, nDCG more slowly (rank 2: 0.63 vs 0.5). They are equal only at rank 1.

**3.** def precision_at_k(ranked, rel, k=10): top = _dedupe(ranked)[:k]; return sum(1 for d in top if rel.get(d,0) > 0) / k, plus a doctest and a hand-computed unit test.

**4.** Store `[d for d, _ in ...][:100]` in the run. If BM25's Recall@100 is much higher than its Recall@10, it's a ranking problem (a reranker or fusion helps); if Recall@100 is also low, it's a vocabulary-mismatch retrieval problem (dense helps).
