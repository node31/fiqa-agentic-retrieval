"""ch02: IR metrics, written by you. Standard library only.

Run the checker:   python -m pytest exercises\\ch02_metrics
Run the demo:      python exercises\\ch02_metrics\\my_metrics.py

The arguments, in plain words (every metric scores ONE query at a time):

  ranked : Sequence[str]       YOUR system's answer for the query: doc ids, best first.
                               e.g. ["9", "1", "22", "2"]          (a list of strings)
  rel    : Mapping[str, int]   The ANSWER KEY for the same query, i.e. that query's row of the qrels:
                               {doc_id: relevance score}.
                               e.g. {"1": 1, "2": 1}               (a dict: docs "1" and "2" are correct)
  k      : int                 Only look at the first k of `ranked` (k=10 → "@10").

  Where `rel` comes from: your ch01 load_split() returns qrels = {query_id: {doc_id: score}}. For query "q1":
      qrels["q1"]  ==  {"1": 1, "2": 1}      ← this inner dict is `rel`
  A doc that isn't in `rel` is NOT relevant. Use rel.get(doc_id, 0) to get 0 for those.

  The type hints: `Sequence[str]` = "a list (or tuple) of strings"; `Mapping[str, int]` = "a dict from str to int",
  read-only (the function looks things up and never changes it). In practice: ranked is a list, rel is a dict.

  Worked example: ranked=["9", "1", "22", "2"], rel={"1": 1, "2": 1}, k=10
      position 1 "9"  → rel.get("9", 0) = 0 → wrong
      position 2 "1"  → rel.get("1", 0) = 1 → correct
      position 3 "22" → 0                  → wrong
      position 4 "2"  → 1                  → correct
      recall = 2 correct found / 2 correct in rel = 1.0 ;  first correct at rank 2 → MRR = 1/2 = 0.5
"""
from __future__ import annotations

import math  # noqa: F401
from typing import Iterable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """Fraction of the relevant docs (rel > 0) that appear in the top-k of `ranked`. 0.0 if no relevant docs.

    Example: recall_at_k(["X", "A", "Y", "Z", "B"], {"A": 1, "B": 1, "C": 1}) == 2/3
    """
    raise NotImplementedError("TODO (ch02): implement recall_at_k")


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """1 / (rank of the first relevant doc in the top-k), ranks start at 1. 0.0 if none.

    Example: mrr_at_k(["X", "A"], {"A": 1}) == 0.5
    """
    raise NotImplementedError("TODO (ch02): implement mrr_at_k")


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """DCG / IDCG with gain 2**rel - 1 and discount 1/log2(rank + 1). 0.0 if IDCG == 0.

    Example: ndcg_at_k(["X", "A"], {"A": 1}) == 1/log2(3) ≈ 0.6309
    """
    raise NotImplementedError("TODO (ch02): implement ndcg_at_k")


def coverage_at_k(ranked: Sequence[str], halves: Mapping[str, Iterable[str]], k: int = 10) -> float:
    """1.0 if the top-k contains at least one gold doc from EVERY half, else 0.0.

    Example: coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1", "b2"]}, k=3) == 1.0
    """
    raise NotImplementedError("TODO (ch02): implement coverage_at_k")


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, no interpolation. p50 of [1..10] is 5; p95 of 20 values is the 19th smallest."""
    raise NotImplementedError("TODO (ch02): implement percentile")


def evaluate(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]], k: int = 10) -> dict:
    """Average nDCG / Recall / MRR over queries that are in BOTH run and qrels and have >= 1 relevant doc.

    Return {"ndcg@10": ..., "recall@10": ..., "mrr@10": ..., "n": number_of_queries}  (keys use k).
    """
    raise NotImplementedError("TODO (ch02): implement evaluate")


if __name__ == "__main__":
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import TINY, use_chapter

    use_chapter("ch01_data")
    from loader import load_jsonl, load_split  # your ch01 code

    corpus = load_jsonl(TINY / "corpus.jsonl")
    _, qrels = load_split(TINY, "dev")
    oracle = {q: list(g) for q, g in qrels.items()}
    rnd = {q: random.Random(0).sample(sorted(corpus), 10) for q in qrels}
    print("oracle run (expect 1.0):", evaluate(oracle, qrels))
    print("random run (expect low):", evaluate(rnd, qrels))
