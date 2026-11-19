"""SOLUTION for ch02: IR metrics. Standard library only.

Try the exercise yourself first (exercises/ch02_metrics/my_metrics.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch02_metrics
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each id, in order.
    WHY: a buggy system that returns the same correct doc 10 times must not get 10x credit."""
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """|relevant ∩ top-k| / |relevant|."""
    relevant = {d for d, r in rel.items() if r > 0}       # WHY r > 0: qrels may list judged-NOT-relevant docs as 0
    if not relevant:
        return 0.0
    hits = sum(1 for d in _dedupe(ranked)[:k] if d in relevant)
    return hits / len(relevant)                           # PITFALL: divide by ALL relevant docs, not by k


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """1 / rank of the first relevant doc (ranks start at 1)."""
    for rank, d in enumerate(_dedupe(ranked)[:k], start=1):   # PITFALL: start=1, or rank 1 gives 1/0
        if rel.get(d, 0) > 0:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    # rank 1 → /log2(2)=1, rank 2 → /log2(3)=1.585, ...  (the "discount" table in the guide)
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """DCG of our ranking / DCG of the best possible ranking."""
    gains = [2 ** rel.get(d, 0) - 1 for d in _dedupe(ranked)[:k]]           # rel 1 → gain 1, rel 0 → gain 0
    # WHY the ideal uses ALL relevant docs (cut to k): a system that finds 1 of 3 gold docs must score < 1.
    ideal = sorted((2 ** r - 1 for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def coverage_at_k(ranked: Sequence[str], halves: Mapping[str, Iterable[str]], k: int = 10) -> float:
    """1.0 if every half of a compound question has >= 1 gold doc in the top-k, else 0.0."""
    top = set(_dedupe(ranked)[:k])
    return 1.0 if all(top & set(gold) for gold in halves.values()) else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the value at position ceil(p/100 · n) in sorted order (1-based)."""
    if not values:
        raise ValueError("percentile of an empty list")
    s = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(s)))   # WHY max(1, ..): p=0 would give position 0
    return s[rank - 1]                            # -1: Python lists are 0-based


def evaluate(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]], k: int = 10) -> dict:
    """Average each metric over queries that are in both the run and qrels and have a relevant doc."""
    qids = [q for q in run if q in qrels and any(r > 0 for r in qrels[q].values())]
    n = len(qids)
    if n == 0:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, "n": 0}
    return {f"ndcg@{k}": sum(ndcg_at_k(run[q], qrels[q], k) for q in qids) / n,
            f"recall@{k}": sum(recall_at_k(run[q], qrels[q], k) for q in qids) / n,
            f"mrr@{k}": sum(mrr_at_k(run[q], qrels[q], k) for q in qids) / n,
            "n": n}


if __name__ == "__main__":
    # The worksheet's examples, so you can see every number.
    print("A nDCG@10:", round(ndcg_at_k(["X", "A", "Y", "Z", "B"], {"A": 1, "B": 1, "C": 1}), 3))     # 0.478
    print("B nDCG@5 :", round(ndcg_at_k(["X", "A", "Y", "B", "Z", "C"], dict.fromkeys("ABCD", 1), 5), 3))  # 0.415
    lat = [20, 21, 19, 22, 20, 23, 21, 20, 95, 24]
    print("D p50/p95:", percentile(lat, 50), percentile(lat, 95))                                      # 21 95
