"""Toy 25: sharing 10 result slots between sub-queries. Interleave vs score-merge vs RRF.

Concept: slot allocation decides Coverage@10 for compound questions.
Real code: src/agent/orchestrator.py::interleave   Doc: docs/topics/13_multi_agent_orchestration.md
"""
from _common import banner, note, step, try_this

from src.agent.orchestrator import interleave
from src.metrics import coverage_at_k, recall_at_k
from src.retrieval.hybrid import rrf

banner("Toy 25: two sub-queries, one 'easy' (high scores) and one 'hard' (low scores)")
A = [(f"a{i}", 0.90 - 0.01 * i) for i in range(10)]  # shorting: confident, high cosine
B = [(f"b{i}", 0.60 - 0.01 * i) for i in range(10)]  # mortgage: harder, lower cosine
gold = {"a": ["a0", "a3"], "b": ["b1", "b6"]}
rel = {d: 1 for g in gold.values() for d in g}
strategies = {
    "score merge": [d for d, _ in sorted(A + B, key=lambda x: -x[1])[:10]],
    "RRF": [d for d, _ in rrf([A, B], top=10)],
    "interleave": interleave([[d for d, _ in A], [d for d, _ in B]], 10),
}
for name, top in strategies.items():
    step(name, f"{top}")
    print(f"     Coverage@10={coverage_at_k(top, gold):.2f}  Recall@10={recall_at_k(top, rel):.2f}")
note("Score merge gives all 10 slots to the easy half: zero coverage. Scores from different sub-queries are "
     "NOT comparable (different queries, different score distributions).")
note("RRF over two lists IS an interleave (rank r of each list gets the same 1/(60+r)), with ties broken by doc id. "
     "Explicit interleave makes the order deterministic and the intent obvious.")
note("Cost of fairness: each half keeps only its top 5, so b6 (rank 7 in B) is lost. With 3 sub-queries "
     "each gets ~3 slots. That's why the planner is capped at 3 sub-queries.")
try_this("Weight slots by planner confidence (e.g. 7/3). When is that better than 5/5?",
         "Should the original-question list get slots BEFORE the sub-queries? Test on compound dev.",
         "Implement 'interleave then rerank the 10 with a cross-encoder'. Coverage preserved?")
