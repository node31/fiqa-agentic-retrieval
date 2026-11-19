"""Toy 07: Recall@k, MRR and nDCG@k computed by hand for 3 queries, then checked against src/metrics.py.

Concept: what each metric rewards, and how they disagree.
Real code: src/metrics.py     Doc: docs/04_ir_metrics.md
"""
import math

from _common import banner, note, step, try_this

from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k

banner("Toy 07: IR metrics by hand (k = 5)")
cases = {
    "q1: gold at rank 1": (["g1", "x", "x2", "x3", "x4"], {"g1": 1, "g2": 1}),
    "q2: both gold at ranks 4,5": (["x", "x2", "x3", "g1", "g2"], {"g1": 1, "g2": 1}),
    "q3: gold missed": (["x", "x2", "x3", "x4", "x5"], {"g1": 1}),
}
for name, (ranked, rel) in cases.items():
    step(name, f"ranked = {ranked}, relevant = {sorted(rel)}")
    hits = [i for i, d in enumerate(ranked, 1) if d in rel]
    r = len(hits) / len(rel)
    m = 1 / hits[0] if hits else 0
    dcg = sum(1 / math.log2(i + 1) for i in hits)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), 5) + 1))
    print(f"     hit ranks = {hits}")
    print(f"     Recall@5 = {len(hits)}/{len(rel)} = {r:.3f}            (src: {recall_at_k(ranked, rel, 5):.3f})")
    print(f"     MRR@5    = 1/{hits[0] if hits else 'inf'} = {m:.3f}          (src: {mrr_at_k(ranked, rel, 5):.3f})")
    print(f"     DCG = sum 1/log2(rank+1) = {dcg:.3f}; IDCG = {idcg:.3f}; nDCG@5 = {dcg / idcg:.3f}"
          f"  (src: {ndcg_at_k(ranked, rel, 5):.3f})")
note("q1 vs q2: q1 has HALF the recall of q2 but a much higher MRR. Recall asks 'did we find them all?', "
     "MRR asks 'how soon is the first good one?', and nDCG blends both with a log discount on rank.")
note("For RAG, Recall@10 matters most (the LLM reads all 10), while for a human search box MRR matters.")
try_this("Make q2's gold appear at ranks 1 and 2. Which metric changes the most?",
         "Give g1 relevance 2 (graded). Recompute nDCG by hand with gain 2^rel - 1.",
         "Why does tests/test_metrics.py check that IDCG uses ALL relevant docs? Break it and see.")
