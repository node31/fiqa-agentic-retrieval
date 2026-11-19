"""Information-retrieval metrics: nDCG@k, Recall@k, MRR@k, Coverage@k.

WHAT
    Pure functions that score ONE ranked list against ONE query's qrels, plus
    ``mean`` helpers. No numpy, no magic: every line maps to a formula in
    docs/04_ir_metrics.md.

WHY OUR OWN
    Libraries (pytrec_eval, ranx) are great, but metrics are where silent bugs
    hide (ties, k-cutoffs, ideal DCG computed over the wrong set). Writing them
    once by hand, with unit tests against hand-computed values, is the fastest
    way to really know what "nDCG@10 = 0.37" means.

DEFINITIONS (for one query; ``ranked`` = our list, ``rel`` = qrels)
    Recall@k   = |top-k ∩ relevant| / |relevant|
    MRR@k      = 1 / rank of the FIRST relevant doc in top-k (0 if none)
    DCG@k      = Σ_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
    nDCG@k     = DCG@k / IDCG@k, where IDCG is DCG of the perfect ordering
    Coverage@k = 1 if top-k has >= 1 relevant doc from EVERY "half" (Part C)

PITFALLS
    * IDCG must be computed from ALL relevant docs (truncated at k), not only
      the relevant docs we happened to retrieve. Otherwise nDCG is inflated.
    * Duplicated doc ids in a ranked list would double count; we dedupe.
    * Queries with no relevant docs are undefined for recall; callers skip them.

SEE ALSO
    examples/07_metrics_by_hand.py, tests/test_metrics.py
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for d in ranked:
        if d not in seen:  # WHY: keep FIRST occurrence; later ones are noise
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Fraction of relevant docs that appear in the top-k.

    >>> recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, k=2)
    0.3333333333333333
    """
    relevant = {d for d, r in rel.items() if r > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for d in _dedupe(ranked)[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant doc within the top-k.

    >>> mrr_at_k(["x", "a"], {"a": 1})
    0.5
    """
    for i, d in enumerate(_dedupe(ranked)[:k], start=1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Normalised discounted cumulative gain with exponential gain (2^rel - 1).

    >>> round(ndcg_at_k(["x", "a"], {"a": 1}), 4)   # 1/log2(3) / 1
    0.6309
    """
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1)
              for i, d in enumerate(_dedupe(ranked)[:k], start=1))
    # PITFALL: ideal ordering uses ALL judged-relevant docs, not just retrieved ones.
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage_at_k(ranked: Sequence[str], halves: Mapping[str, Iterable[str]], k: int = 10) -> float:
    """1.0 if the top-k contains at least one gold doc from EVERY half, else 0.0.

    ``halves`` maps a half name ("a"/"b") to that half's gold doc ids. Used for
    the compound query set in Part C, where one question merges two topics.

    >>> coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1", "b2"]}, k=3)
    1.0
    >>> coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1"]}, k=2)
    0.0
    """
    top = set(_dedupe(ranked)[:k])
    return 1.0 if all(top & set(golds) for golds in halves.values()) else 0.0


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]],
                 k: int = 10) -> dict[str, float]:
    """Average nDCG@k / Recall@k / MRR@k over the queries present in BOTH run and qrels."""
    qids = [q for q in run if q in qrels and any(r > 0 for r in qrels[q].values())]
    if not qids:
        return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "n": 0}
    n = len(qids)
    return {
        f"ndcg@{k}": sum(ndcg_at_k(run[q], qrels[q], k) for q in qids) / n,
        f"recall@{k}": sum(recall_at_k(run[q], qrels[q], k) for q in qids) / n,
        f"mrr@{k}": sum(mrr_at_k(run[q], qrels[q], k) for q in qids) / n,
        "n": n,
    }
