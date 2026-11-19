"""Checker for ch05. Don't edit. Run: python -m pytest exercises/ch05_fusion"""
import random

import pytest

import my_fusion as mine
from src.retrieval import hybrid as ref

BM25 = [("A", 18.2), ("B", 11.0), ("C", 10.7), ("D", 2.1)]
DENSE = [("C", 0.83), ("E", 0.81), ("A", 0.52), ("B", 0.50)]


def check_minmax():
    assert mine.minmax([("a", 10.0), ("b", 5.0), ("c", 0.0)]) == pytest.approx({"a": 1.0, "b": 0.5, "c": 0.0})


def check_minmax_all_equal_does_not_divide_by_zero():
    assert mine.minmax([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}
    assert mine.minmax([]) == {}


def check_linear_toy08_alpha07():
    got = dict(mine.linear(BM25, DENSE, 0.7))
    want = {"C": 0.860, "E": 0.658, "A": 0.342, "B": 0.166, "D": 0.0}
    assert set(got) == set(want), "every doc from EITHER list must appear (missing part = 0)"
    for d, s in want.items():
        assert got[d] == pytest.approx(s, abs=1e-3), d
    assert [d for d, _ in mine.linear(BM25, DENSE, 0.7)] == ["C", "E", "A", "B", "D"]


def check_linear_alpha05_winner():
    top = mine.linear(BM25, DENSE, 0.5)[0]
    assert top[0] == "C" and top[1] == pytest.approx(0.767, abs=1e-3)


def check_rrf_toy08_tie():
    got = mine.rrf([BM25, DENSE])
    assert [d for d, _ in got] == ["A", "C", "B", "E", "D"], "A and C tie; ties are broken by doc id"
    assert got[0][1] == pytest.approx(1 / 61 + 1 / 63)
    assert got[0][1] == pytest.approx(got[1][1])


def check_rrf_weights_and_top():
    got = mine.rrf([BM25, DENSE], weights=[0.3, 0.7], top=2)
    want = ref.rrf([BM25, DENSE], weights=[0.3, 0.7], top=2)
    assert [d for d, _ in got] == [d for d, _ in want]
    assert [s for _, s in got] == pytest.approx([s for _, s in want])
    assert len(got) == 2


def check_random_lists_match_reference():
    rng = random.Random(7)
    for _ in range(200):
        ids = [f"d{i}" for i in range(12)]
        b = sorted(((d, rng.uniform(0, 20)) for d in rng.sample(ids, rng.randint(2, 8))), key=lambda x: -x[1])
        n = sorted(((d, rng.uniform(0, 1)) for d in rng.sample(ids, rng.randint(2, 8))), key=lambda x: -x[1])
        a = rng.choice([0.1, 0.5, 0.7, 0.9])
        g, w = mine.linear(b, n, a), ref.linear(b, n, a)
        assert [d for d, _ in g] == [d for d, _ in w], f"linear order differs (alpha={a})"
        assert [s for _, s in g] == pytest.approx([s for _, s in w])
        g, w = mine.rrf([b, n]), ref.rrf([b, n])
        assert [d for d, _ in g] == [d for d, _ in w], "rrf order differs"


# ---- FULL checks (python -m pytest exercises\\ch05_fusion --full): fuse REAL result lists -----------------------
def _full_lists():
    import fullkit
    bm25 = {q: [tuple(x) for x in lst] for q, lst in fullkit.bm25_run("dev").items()}    # top-100 each
    dense = {q: [tuple(x) for x in lst] for q, lst in fullkit.dense_run("dev").items()}  # top-100 each
    return bm25, dense, fullkit.split("dev")[1]


@pytest.mark.full
def check_full_linear_alpha07_is_our_chosen_system():
    """500 dev queries, BM25 top-100 + dense top-100 → your linear(α=0.7) → nDCG@10 = 0.419 (the shipped number)."""
    from src.metrics import evaluate_run
    bm25, dense, qrels = _full_lists()
    run = {q: [d for d, _ in mine.linear(bm25[q], dense[q], 0.7, top=10)] for q in qrels}
    want = {q: [d for d, _ in ref.linear(bm25[q], dense[q], 0.7, top=10)] for q in qrels}
    assert run == want, "same inputs, same formula: your top-10s should equal the reference exactly"
    assert evaluate_run(run, qrels)["ndcg@10"] == pytest.approx(0.419, abs=0.002)


@pytest.mark.full
def check_full_rrf_is_below_dense_alone():
    """The surprise from guide chapter 5: plain RRF (~0.373) scores BELOW dense alone (0.388) on FiQA."""
    from src.metrics import evaluate_run
    bm25, dense, qrels = _full_lists()
    rrf_run = {q: [d for d, _ in mine.rrf([bm25[q], dense[q]], top=10)] for q in qrels}
    dense_run = {q: [d for d, _ in dense[q][:10]] for q in qrels}
    rrf_ndcg, dense_ndcg = evaluate_run(rrf_run, qrels)["ndcg@10"], evaluate_run(dense_run, qrels)["ndcg@10"]
    assert rrf_ndcg == pytest.approx(0.373, abs=0.003)
    assert rrf_ndcg < dense_ndcg
