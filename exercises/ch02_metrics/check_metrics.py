"""Checker for ch02. Don't edit. Run: python -m pytest ch02_metrics"""
import random

import pytest

import my_metrics as mine
from src import metrics as ref


# ---- the guide's worked examples ---------------------------------------------------------------
def check_recall_guide_example():
    assert mine.recall_at_k(["X", "A", "Y", "Z", "B"], {"A": 1, "B": 1, "C": 1}) == pytest.approx(2 / 3)


def check_mrr_guide_example():
    assert mine.mrr_at_k(["X", "A", "Y", "Z", "B"], {"A": 1, "B": 1, "C": 1}) == pytest.approx(0.5)


def check_ndcg_guide_example():
    # DCG = 1/log2(3) + 1/log2(6) = 1.018 ; IDCG = 1 + 0.631 + 0.5 = 2.131 ; nDCG = 0.478
    assert mine.ndcg_at_k(["X", "A", "Y", "Z", "B"], {"A": 1, "B": 1, "C": 1}) == pytest.approx(0.4777, abs=1e-3)


def check_ndcg_toy07_k5():
    rel = {"g1": 1, "g2": 1}
    assert mine.ndcg_at_k(["g1", "x", "x2", "x3", "x4"], rel, k=5) == pytest.approx(0.613, abs=1e-3)
    assert mine.ndcg_at_k(["x", "x2", "x3", "g1", "g2"], rel, k=5) == pytest.approx(0.501, abs=1e-3)
    assert mine.ndcg_at_k(["x", "x2", "x3", "x4", "x5"], {"g1": 1}, k=5) == 0.0


def check_ndcg_idcg_uses_all_gold():
    # found only 1 of 3 gold docs, at rank 1: must NOT be a perfect 1.0
    assert mine.ndcg_at_k(["A"], {"A": 1, "B": 1, "C": 1}) < 0.5


def check_paper_exercise_b():
    assert mine.ndcg_at_k(["X", "A", "Y", "B", "Z", "C"], {"A": 1, "B": 1, "C": 1, "D": 1}, k=5) == pytest.approx(0.4144, abs=1e-3)


# ---- edge cases ----------------------------------------------------------------------------------
def check_duplicates_are_counted_once():
    rel = {"A": 1, "B": 1}
    assert mine.recall_at_k(["A", "A", "A"], rel) == pytest.approx(0.5)
    assert mine.ndcg_at_k(["A", "A"], rel) == pytest.approx(ref.ndcg_at_k(["A", "A"], rel))


def check_no_relevant_docs_gives_zero():
    for f in (mine.recall_at_k, mine.mrr_at_k, mine.ndcg_at_k):
        assert f(["A", "B"], {}) == 0.0
        assert f(["A", "B"], {"A": 0}) == 0.0


def check_k_larger_than_list():
    assert mine.recall_at_k(["A"], {"A": 1}, k=100) == 1.0


def check_coverage():
    assert mine.coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1", "b2"]}, k=3) == 1.0
    assert mine.coverage_at_k(["a1", "x", "b1"], {"a": ["a1"], "b": ["b1"]}, k=2) == 0.0
    assert mine.coverage_at_k(["a1", "x", "b2"], {"a": ["a1"], "b": ["b1", "b2"]}, k=3) == 1.0


def check_percentile():
    lat = [20, 21, 19, 22, 20, 23, 21, 20, 95, 24]
    assert mine.percentile(lat, 50) == 21
    assert mine.percentile(lat, 95) == 95
    assert mine.percentile(list(range(1, 21)), 95) == 19


# ---- random cross-check against the reference implementation ---------------------------------------
def check_random_rankings_match_reference():
    rng = random.Random(42)
    docs = [f"d{i}" for i in range(30)]
    for _ in range(300):
        ranked = [rng.choice(docs) for _ in range(rng.randint(0, 15))]
        rel = {d: rng.choice([1, 1, 2]) for d in rng.sample(docs, rng.randint(1, 6))}
        k = rng.choice([1, 3, 5, 10])
        for name in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
            got, want = getattr(mine, name)(ranked, rel, k), getattr(ref, name)(ranked, rel, k)
            assert got == pytest.approx(want), f"{name}(ranked={ranked}, rel={rel}, k={k}): yours {got}, expected {want}"


def check_evaluate():
    run = {"q1": ["g1", "x"], "q2": ["x", "g2"], "q3": ["x"], "q_unjudged": ["x"]}
    qrels = {"q1": {"g1": 1}, "q2": {"g2": 1}, "q3": {"g3": 1}, "q_nogold": {"z": 0}}
    got, want = mine.evaluate(run, qrels), ref.evaluate_run(run, qrels)
    assert got["n"] == want["n"] == 3
    for key in ("ndcg@10", "recall@10", "mrr@10"):
        assert got[key] == pytest.approx(want[key]), key


# ---- FULL checks (python -m pytest exercises\\ch02_metrics --full): grade your metrics on a REAL run ----------
@pytest.mark.full
def check_full_metrics_on_real_bm25_run():
    """500 dev queries x the reference BM25 top-10: your evaluate() must equal the reference to 1e-9,
    and the BM25 number must be the one in the docs (0.264)."""
    import fullkit
    _, qrels = fullkit.split("dev")
    run = {q: [d for d, _ in lst[:10]] for q, lst in fullkit.bm25_run("dev").items()}
    got, want = mine.evaluate(run, qrels), ref.evaluate_run(run, qrels)
    for key in ("ndcg@10", "recall@10", "mrr@10"):
        assert got[key] == pytest.approx(want[key], abs=1e-9), key
    assert got["n"] == 500
    assert got["ndcg@10"] == pytest.approx(0.264, abs=0.002)
