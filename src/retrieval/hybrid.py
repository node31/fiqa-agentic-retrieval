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

DISTRIBUTED CORRECTNESS (docs/08_sharding_scatter_gather.md)
    Fusion must run on the GLOBALLY merged BM25 list and the globally merged
    dense list, i.e. at the router, AFTER merging shards. Fusing on each shard
    and then merging is wrong for linear fusion, because min-max uses each
    shard's local min/max. examples/13_merge_then_fuse.py demonstrates it.

PITFALLS
    * Docs missing from one list: RRF gives them 0 from that list, and linear
      gives them the normalised minimum (0). Both are deliberate choices.
    * A list where all scores are equal makes min-max divide by zero, so we map it to 1.0.

HOW TO READ THIS FILE
    The router's order of operations is:
      1. every shard returns its own BM25 top list and dense top list;
      2. merge_shard_lists() turns the per-shard lists into ONE global BM25 list and
         ONE global dense list (the "gather" step);
      3. fuse() combines those two global lists into the final ranking, via rrf()
         or linear() (which uses minmax()).
    So read the file bottom-up for the data flow, or top-down for the building blocks.

    Shapes: a ``Ranked`` list is [(doc_id, score), ...], best first. Fused scores
    are only meaningful for ORDERING within one query; they are not probabilities.

    Shipped config (src/config.py): linear fusion, α = 0.7. nDCG@10 on FiQA dev:
    hybrid 0.419 vs dense alone 0.388 vs BM25 alone 0.264. (nDCG@10 is a 0-1
    ranking-quality score that rewards putting relevant passages near the top.)
"""
from __future__ import annotations

from typing import Sequence

from src.config import DEFAULT_ALPHA, RRF_K

Ranked = list[tuple[str, float]]  # [(doc_id, score), ...], best first


def _sorted(scores: dict[str, float], k: int | None) -> Ranked:
    # dict -> list sorted by score (high to low), ties by doc id (A-Z), cut to k.
    # PITFALL: k=None AND k=0 both mean "no cut" (0 is falsy), not "empty list".
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # deterministic ties
    return out[:k] if k else out


def rrf(lists: Sequence[Ranked], k: int = RRF_K, weights: Sequence[float] | None = None,
        top: int | None = None) -> Ranked:
    """Reciprocal Rank Fusion.

    >>> rrf([[("a", 9.0), ("b", 1.0)], [("b", .9), ("c", .1)]], k=1)
    [('b', 0.8333333333333333), ('a', 0.5), ('c', 0.3333333333333333)]
    """
    # Reading the doctest (k=1): "b" is rank 2 in list 1 and rank 1 in list 2, so
    # 1/(1+2) + 1/(1+1) = 0.333 + 0.5 = 0.833; "a" = 1/(1+1); "c" = 1/(1+2).
    # The input scores (9.0, .9, ...) are ignored; only positions count.
    # With the real k = 60 the differences are gentle: rank 1 -> 1/61 ≈ 0.0164,
    # rank 10 -> 1/70 ≈ 0.0143. WHY a large k: a doc ranked well in BOTH lists beats
    # a doc ranked first in only one (one list's #1 cannot dominate the fusion).
    # weights: one multiplier per list (default all 1.0); fuse() uses (1-α, α).
    weights = weights or [1.0] * len(lists)
    scores: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        # rank starts at 1 (best). A doc absent from a list simply gets nothing from it.
        for rank, (doc, _) in enumerate(lst, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return _sorted(scores, top)


def minmax(lst: Ranked) -> dict[str, float]:
    """Scale scores to [0, 1] within one list.

    >>> minmax([("a", 10.0), ("b", 5.0), ("c", 0.0)])
    {'a': 1.0, 'b': 0.5, 'c': 0.0}
    """
    # Min-max normalisation: (s - lo) / (hi - lo) maps the list's best score to 1.0
    # and its worst to 0.0, so lists with different scales become comparable.
    # Doctest: b = (5 - 0) / (10 - 0) = 0.5. Note lo is the lowest score IN THE LIST
    # (e.g. the 100th result), not the lowest possible score.
    if not lst:
        return {}
    hi, lo = max(s for _, s in lst), min(s for _, s in lst)
    if hi == lo:
        # WHY 1.0 (not 0.0): every doc in the list was retrieved, so treat them all
        # as the list's best rather than erasing this list's vote entirely.
        return {d: 1.0 for d, _ in lst}
    return {d: (s - lo) / (hi - lo) for d, s in lst}


def linear(bm25: Ranked, dense: Ranked, alpha: float = DEFAULT_ALPHA, top: int | None = None) -> Ranked:
    """α · dense + (1-α) · bm25 after min-max normalisation of each list."""
    # nb / nd: doc_id -> score in [0, 1] for the BM25 / dense list.
    nb, nd = minmax(bm25), minmax(dense)
    # Score the UNION of both lists; a doc missing from one list gets 0 there.
    # Example with α = 0.7: bm25_norm = 1.0, dense_norm = 0.5 -> 0.7·0.5 + 0.3·1.0 = 0.65,
    # while a doc only in the dense list with dense_norm = 0.9 -> 0.7·0.9 = 0.63.
    docs = set(nb) | set(nd)
    return _sorted({d: alpha * nd.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in docs}, top)


def fuse(bm25: Ranked, dense: Ranked, method: str = "rrf", alpha: float = DEFAULT_ALPHA,
         top: int | None = None) -> Ranked:
    """Dispatch to RRF or linear fusion (the router calls this after merging shards).

    ``alpha`` is the weight on DENSE for both methods: linear uses α·dense + (1-α)·bm25,
    and RRF uses list weights (1-α, α). At α = 0.5 weighted RRF ranks exactly like plain RRF.
    """
    if method == "rrf":
        return rrf([bm25, dense], weights=[1 - alpha, alpha], top=top)
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
    # WHY the union is enough: any doc in the global top-depth beats all but at most
    # depth-1 docs overall, so it is also in the top-depth of its own shard.
    # Raw scores are compared directly here (no normalisation): that is only valid
    # because all shards share the same IDF / the same embedding model.
    best: dict[str, float] = {}
    for lst in lists:
        for d, s in lst:
            best[d] = max(s, best.get(d, float("-inf")))  # WHY max: defensive vs dupes
    return _sorted(best, depth)
