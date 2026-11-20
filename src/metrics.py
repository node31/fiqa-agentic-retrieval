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
    examples/07_metrics_by_hand.py, tests/test_metrics.py,
    docs/from_zero/02_measuring_quality_and_speed.md (gentle version)

HOW TO READ THIS FILE
    Vocabulary used below:
      gold / relevant  a passage a human judged as answering the query.
      qrels            "query relevance judgements": the answer key.
                       Shape: {query_id: {doc_id: grade}}; grade > 0 = relevant
                       (FiQA only uses grade 1). Docs not listed = not relevant.
      ranked           our system's answer for ONE query: [doc_id, ...] best first.
      run              answers for MANY queries: {query_id: ranked}.
      @k               "only look at the first k results" (we use k=10).
    Read top to bottom: _dedupe (cleanup) -> four per-query metrics ->
    evaluate_run (the average over all queries, what src/eval.py reports).

    Mini example used in the comments below: gold = {"a", "b"},
    ranked = ["x", "a", "y", "b"], k = 10.
      Recall@10 = 2/2 = 1.0   (both gold docs are somewhere in the top 10)
      MRR@10    = 1/2 = 0.5   (first gold doc is at rank 2)
      nDCG@10   = (1/log2(3) + 1/log2(5)) / (1/log2(2) + 1/log2(3))
                = (0.6309 + 0.4307) / (1 + 0.6309) = 0.6509
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    # Example: ["a", "x", "a", "b"] -> ["a", "x", "b"]. Without this, a system
    # that returns the same gold doc twice would score 2 "hits" for 1 doc.
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

    "Of everything we SHOULD have found, how much did we find?" Order inside
    the top-k doesn't matter. (Doctest above: top-2 = [a, x] -> 1 of 3 gold.)
    """
    # The gold set: judged docs with grade > 0 (a grade-0 line means "judged, NOT relevant").
    relevant = {d for d, r in rel.items() if r > 0}
    # WHY 0.0 and not an error: recall is undefined with nothing to find
    # (0/0); evaluate_run skips such queries anyway, so this is just a safe guard.
    if not relevant:
        return 0.0
    # Dedupe BEFORE cutting at k, so a duplicate can't push a real result out of the top-k.
    hits = sum(1 for d in _dedupe(ranked)[:k] if d in relevant)
    # Denominator = ALL gold docs (also the ones we missed), not just k.
    return hits / len(relevant)


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant doc within the top-k.

    >>> mrr_at_k(["x", "a"], {"a": 1})
    0.5

    MRR = "Mean Reciprocal Rank" (the MEAN part happens in evaluate_run).
    It answers "how far down must a user scroll to see the first good answer?":
    rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.333, not in top-k -> 0.0.
    Only the FIRST gold doc counts; the others don't change the score.
    """
    # start=1 so `i` is the human rank (1 = top result), not the 0-based index.
    for i, d in enumerate(_dedupe(ranked)[:k], start=1):
        # rel.get(d, 0): docs absent from qrels are treated as not relevant.
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Normalised discounted cumulative gain with exponential gain (2^rel - 1).

    >>> round(ndcg_at_k(["x", "a"], {"a": 1}), 4)   # 1/log2(3) / 1
    0.6309

    The idea, in four steps:
      * gain      each result is worth (2^grade - 1): grade 1 -> 1, grade 2 -> 3,
                  not relevant (grade 0) -> 0. Exponential, so "very relevant"
                  counts much more than "somewhat relevant" (FiQA only has grade 1).
      * discount  a result at rank i is divided by log2(i + 1): rank 1 -> /1,
                  rank 2 -> /1.585, rank 3 -> /2, rank 10 -> /3.46. Lower = worth less.
      * DCG       ("discounted cumulative gain") = sum of gain/discount over the top-k.
      * nDCG      DCG divided by IDCG ("ideal DCG": the DCG of the best possible
                  ordering), so a perfect ranking scores 1.0 whatever the number
                  of gold docs. Unlike recall, nDCG rewards putting gold docs HIGH.
    """
    # DCG of OUR ranking. Non-gold docs add (2^0 - 1) = 0, so only gold docs contribute.
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1)
              for i, d in enumerate(_dedupe(ranked)[:k], start=1))
    # PITFALL: ideal ordering uses ALL judged-relevant docs, not just retrieved ones.
    # The ideal list is just the grades of all gold docs, best grade first, cut at k
    # (e.g. 3 gold docs of grade 1 -> [1, 1, 1]). Doc ids don't matter here.
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:k]
    # IDCG: the same DCG formula applied to that perfect list.
    # e.g. [1, 1, 1] -> 1/1 + 1/1.585 + 1/2 = 2.131. Finding just one of the
    # three gold docs at rank 1 gives nDCG = 1 / 2.131 = 0.469, not 1.0.
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    # idcg == 0 only when the query has no gold doc at all: return 0 instead of dividing by zero.
    return dcg / idcg if idcg > 0 else 0.0


def coverage_at_k(ranked: Sequence[str], halves: Mapping[str, Iterable[str]], k: int = 10) -> float:
    """1.0 if the top-k contains at least one gold doc from EVERY half, else 0.0.

    ``halves`` maps a half name ("a"/"b") to that half's gold doc ids. Used for
    the compound query set in Part C, where one question merges two topics.

    >>> coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1", "b2"]}, k=3)
    1.0
    >>> coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1"]}, k=2)
    0.0

    WHY a separate metric: for a compound question ("X and also Y") plain
    recall can look fine while the answer only covers X. Coverage is
    all-or-nothing per query: did EACH sub-question get at least one gold doc?
    (Second doctest: top-2 = [a1, x], so half "b" is not covered -> 0.0.)
    """
    top = set(_dedupe(ranked)[:k])
    # `top & set(golds)` is the set intersection: non-empty (truthy) when this
    # half has >= 1 gold doc in the top-k. all() requires that for every half.
    return 1.0 if all(top & set(golds) for golds in halves.values()) else 0.0


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]],
                 k: int = 10) -> dict[str, float]:
    """Average nDCG@k / Recall@k / MRR@k over the queries present in BOTH run and qrels.

    Shapes:
        run   = {query_id: [doc_id, ...]}        our ranked answers, best first
        qrels = {query_id: {doc_id: grade}}      the answer key
    Returns e.g. ``{"ndcg@10": 0.37, "recall@10": 0.45, "mrr@10": 0.41, "n": 500}``
    (illustrative numbers) where ``n`` = how many queries were averaged (always check it!).
    Each metric is a plain (macro) average: every query counts equally, no
    matter how many gold docs it has.
    """
    # Keep only queries we can score: we answered them, they have an answer key,
    # and that key contains at least one relevant doc (otherwise recall = 0/0).
    qids = [q for q in run if q in qrels and any(r > 0 for r in qrels[q].values())]
    if not qids:
        # PITFALL: these keys say "@10" even if k != 10 (the normal path below uses f"@{k}").
        return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "n": 0}
    n = len(qids)
    # Score each query on its own, then take the mean over queries.
    return {
        f"ndcg@{k}": sum(ndcg_at_k(run[q], qrels[q], k) for q in qids) / n,
        f"recall@{k}": sum(recall_at_k(run[q], qrels[q], k) for q in qids) / n,
        f"mrr@{k}": sum(mrr_at_k(run[q], qrels[q], k) for q in qids) / n,
        "n": n,
    }
