# Solutions: hybrid

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** RRF uses only ranks; multiplying scores by a positive constant preserves every rank, so the fused result is unchanged (tested in test_rrf_ignores_score_scale).

**2.** Min-max over only 3 scores: one doc gets 1.0, one 0.0, one in between. Docs missing from the BM25 list get 0, the same as BM25's worst returned doc. Short lists distort the normalisation; deeper candidate lists make it fairer.

**3.** See docs/solutions/drills/drill_02.md.

**4.** From results/ablation_fusion_minilm.json: linear drops below dense (0.388) for α ≤ ~0.45 (α = 0.3 → 0.336) and stays above it all the way to α = 0.9 (0.401). RRF needs α ≥ 0.6 to beat dense.
