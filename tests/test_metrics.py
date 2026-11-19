"""Unit tests for src/metrics.py against hand-computed values.

Each expected number is derived in a comment so a reader can re-check it with
pen and paper (that is the point of these tests).
"""
import math

import pytest

from src.metrics import coverage_at_k, evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k

REL = {"a": 1, "b": 1, "c": 1}  # three relevant docs


def test_recall_counts_only_top_k():
    # top-3 = [a, x, b] -> 2 of 3 relevant found
    assert recall_at_k(["a", "x", "b", "c"], REL, k=3) == pytest.approx(2 / 3)


def test_recall_dedupes_repeated_ids():
    # 'a' twice must not count as two hits
    assert recall_at_k(["a", "a", "x"], REL, k=3) == pytest.approx(1 / 3)


def test_mrr_first_relevant_rank():
    assert mrr_at_k(["x", "y", "b"], REL) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y"], REL) == 0.0
    # outside cutoff -> 0
    assert mrr_at_k(["x", "y", "b"], REL, k=2) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b", "c"], REL) == pytest.approx(1.0)
    # ranked = [x, a]: DCG = 1/log2(3); IDCG (3 relevant) = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(["x", "a"], REL) == pytest.approx(dcg / idcg)


def test_ndcg_idcg_uses_all_relevant_not_only_retrieved():
    # PITFALL guard: retrieving only 'a' at rank 1 is NOT perfect when 3 are relevant.
    assert ndcg_at_k(["a"], REL) < 1.0


def test_ndcg_graded_relevance():
    rel = {"a": 2, "b": 1}
    # ranked [b, a]: DCG = (2^1-1)/1 + (2^2-1)/log2(3) = 1 + 3/1.585
    # ideal  [a, b]: IDCG = 3/1 + 1/log2(3)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], rel) == pytest.approx(expected)


def test_coverage_requires_every_half():
    halves = {"a": ["a1", "a2"], "b": ["b1"]}
    assert coverage_at_k(["a2", "b1"], halves) == 1.0
    assert coverage_at_k(["a1", "a2"], halves) == 0.0


def test_evaluate_run_averages_and_skips_unjudged():
    run = {"q1": ["a"], "q2": ["x"], "q_unjudged": ["a"]}
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    m = evaluate_run(run, qrels)
    assert m["n"] == 2
    assert m["recall@10"] == pytest.approx(0.5)
    assert m["mrr@10"] == pytest.approx(0.5)
