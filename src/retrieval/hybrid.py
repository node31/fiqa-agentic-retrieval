"""Hybrid retrieval: fuse a BM25 ranking and a dense ranking into one list.

WHAT
    Two principled fusion methods:
      * RRF (Reciprocal Rank Fusion): uses only RANKS.
            score(d) = Σ_lists  w_i / (k + rank_i(d))         k = 60
      * Linear interpolation: min-max normalise each list's SCORES to [0, 1], then
            score(d) = α · dense_norm(d) + (1 - α) · bm25_norm(d)

WHY BOTH
    BM25 scores are unbounded (5, 17, 42...), while cosine scores sit in
    [0.2, 0.8]. Adding raw scores lets BM25 dominate. RRF sidesteps scale
    entirely. Linear fusion keeps score *magnitudes* (a dense score of 0.81 vs
    0.80 is "tied", while BM25 42 vs 7 is not), but it needs a tuned α. We
    sweep α on dev in the ablation. See docs/05_hybrid_fusion.md and
    examples/08_fusion_rrf_vs_linear.py.

DISTRIBUTED CORRECTNESS (docs/adr/0003-merge-then-fuse.md)
    Fusion must run on the GLOBALLY merged BM25 list and the globally merged
    dense list, i.e. at the router, AFTER merging shards. Fusing on each shard
    and then merging is wrong for linear fusion, because min-max uses each
    shard's local min/max. examples/13_merge_then_fuse.py demonstrates it.

PITFALLS
    * Docs missing from one list: RRF gives them 0 from that list, and linear
      gives them the normalised minimum (0). Both are deliberate choices.
    * A list where all scores are equal makes min-max divide by zero, so we map it to 1.0.
"""
from __future__ import annotations

from typing import Sequence

from src.config import DEFAULT_ALPHA, RRF_K

Ranked = list[tuple[str, float]]


def _sorted(scores: dict[str, float], k: int | None) -> Ranked:
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # deterministic ties
    return out[:k] if k else out


def rrf(lists: Sequence[Ranked], k: int = RRF_K, weights: Sequence[float] | None = None,
        top: int | None = None) -> Ranked:
    """Reciprocal Rank Fusion.

    >>> rrf([[("a", 9.0), ("b", 1.0)], [("b", .9), ("c", .1)]], k=1)
    [('b', 0.8333333333333333), ('a', 0.5), ('c', 0.3333333333333333)]
    """
    weights = weights or [1.0] * len(lists)
    scores: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        for rank, (doc, _) in enumerate(lst, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return _sorted(scores, top)


def minmax(lst: Ranked) -> dict[str, float]:
    """Scale scores to [0, 1] within one list.

    >>> minmax([("a", 10.0), ("b", 5.0), ("c", 0.0)])
    {'a': 1.0, 'b': 0.5, 'c': 0.0}
    """
    if not lst:
        return {}
    hi, lo = max(s for _, s in lst), min(s for _, s in lst)
    if hi == lo:
        return {d: 1.0 for d, _ in lst}
    return {d: (s - lo) / (hi - lo) for d, s in lst}


def linear(bm25: Ranked, dense: Ranked, alpha: float = DEFAULT_ALPHA, top: int | None = None) -> Ranked:
    """α · dense + (1-α) · bm25 after min-max normalisation of each list."""
    nb, nd = minmax(bm25), minmax(dense)
    docs = set(nb) | set(nd)
    return _sorted({d: alpha * nd.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in docs}, top)


def fuse(bm25: Ranked, dense: Ranked, method: str = "rrf", alpha: float = DEFAULT_ALPHA,
         top: int | None = None) -> Ranked:
    """Dispatch to RRF or linear fusion (the router calls this after merging shards)."""
    if method == "rrf":
        return rrf([bm25, dense], top=top)
    if method == "linear":
        return linear(bm25, dense, alpha, top=top)
    raise ValueError(f"unknown fusion method {method!r}")


def merge_shard_lists(lists: Sequence[Ranked], depth: int) -> Ranked:
    """Merge per-shard top-k lists into the global top-``depth`` (scatter-gather "gather").

    Correct only if every shard scored with the SAME statistics, so scores are
    comparable (global IDF for BM25, same model for dense). If each shard
    returns its top-``depth``, the global top-``depth`` is guaranteed to be in
    the union.

    >>> merge_shard_lists([[("a", 3.0)], [("b", 5.0), ("c", 1.0)]], depth=2)
    [('b', 5.0), ('a', 3.0)]
    """
    best: dict[str, float] = {}
    for lst in lists:
        for d, s in lst:
            best[d] = max(s, best.get(d, float("-inf")))  # WHY max: defensive vs dupes
    return _sorted(best, depth)
