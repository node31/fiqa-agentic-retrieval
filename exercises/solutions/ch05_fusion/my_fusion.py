"""SOLUTION for ch05: hybrid fusion. Standard library only.

Try the exercise yourself first (exercises/ch05_fusion/my_fusion.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch05_fusion
"""
from __future__ import annotations

from typing import Sequence

Ranked = list[tuple[str, float]]


def _sorted(scores: dict[str, float], top: int | None) -> Ranked:
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))   # score desc, then doc id asc (deterministic ties)
    return out[:top] if top else out


def minmax(lst: Ranked) -> dict[str, float]:
    """Rescale this list's scores to [0, 1]. WHY: BM25 is 0..~20, cosine is ~0.2..0.9; raw sums let BM25 win."""
    if not lst:
        return {}
    hi = max(s for _, s in lst)
    lo = min(s for _, s in lst)
    if hi == lo:                                   # PITFALL: division by zero when every score is equal
        return {d: 1.0 for d, _ in lst}
    return {d: (s - lo) / (hi - lo) for d, s in lst}


def linear(bm25: Ranked, dense: Ranked, alpha: float = 0.7, top: int | None = None) -> Ranked:
    """alpha · dense + (1 - alpha) · bm25, both min-maxed. A doc missing from a list gets 0 from it."""
    nb, nd = minmax(bm25), minmax(dense)
    union = set(nb) | set(nd)                      # WHY the union: a doc found by only ONE retriever still counts
    return _sorted({d: alpha * nd.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in union}, top)


def rrf(lists: Sequence[Ranked], k: int = 60, weights: Sequence[float] | None = None,
        top: int | None = None) -> Ranked:
    """score(d) = Σ w_i / (k + rank_i(d)). Uses only RANKS, so score scales don't matter at all."""
    weights = list(weights) if weights is not None else [1.0] * len(lists)
    scores: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        for rank, (doc, _score) in enumerate(lst, start=1):    # the raw score is ignored on purpose
            # WHY k=60: dampens the gap between rank 1 and rank 2 (1/61 vs 1/62), so agreement between
            # lists matters more than being #1 in one list. k=60 is Cormack et al.'s default.
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return _sorted(scores, top)


if __name__ == "__main__":
    bm25 = [("A", 18.2), ("B", 11.0), ("C", 10.7), ("D", 2.1)]
    dense = [("C", 0.83), ("E", 0.81), ("A", 0.52), ("B", 0.50)]
    print("minmax bm25 :", {d: round(s, 3) for d, s in minmax(bm25).items()})
    print("minmax dense:", {d: round(s, 3) for d, s in minmax(dense).items()})
    for a in (0.3, 0.5, 0.7):
        print(f"linear α={a}:", [(d, round(s, 3)) for d, s in linear(bm25, dense, a)])
    print("rrf k=60   :", [(d, round(s, 5)) for d, s in rrf([bm25, dense])])
    print("rrf k=1    :", [(d, round(s, 4)) for d, s in rrf([bm25, dense], k=1)])
