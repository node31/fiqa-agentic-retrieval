"""Unit tests for the pure helpers in scripts/finetune.py (no model, no data)."""
from __future__ import annotations

from scripts.finetune import build_triples, in_batch_mask, lr_at, pick_negatives, split_train


def test_split_train_is_disjoint_deterministic_and_sized():
    ids = [str(i) for i in range(100)]
    tr, val = split_train(ids, 10, seed=42)
    assert len(val) == 10 and len(tr) == 90 and not set(tr) & set(val)
    assert split_train(list(reversed(ids)), 10, seed=42) == (tr, val)  # input order doesn't matter


def test_pick_negatives_uses_ranks_and_skips_gold_and_banned():
    ranked = [f"d{i}" for i in range(1, 41)]          # d1 is rank 1
    negs = pick_negatives(ranked, gold={"d12"}, banned={"d15"}, lo=10, hi=30)
    assert negs[0] == "d10" and negs[-1] == "d30"
    assert "d9" not in negs and "d31" not in negs     # outside ranks 10-30
    assert "d12" not in negs and "d15" not in negs    # gold, banned


def test_build_triples_one_per_gold_and_never_gold_or_banned():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
    questions = {"q1": "x", "q2": "y"}
    corpus = ["a", "b", "c", "e1", "e2", "ban"]
    triples = build_triples(questions, qrels, {"q1": ["n1"], "q2": []}, corpus, banned={"ban"}, seed=1)
    assert sorted((q, p) for q, p, _ in triples) == [("q1", "a"), ("q1", "b"), ("q2", "c")]
    for q, _, n in triples:
        assert n not in qrels[q] and n != "ban"
    assert {n for q, _, n in triples if q == "q1"} == {"n1"}  # mined negative preferred


def test_in_batch_mask_hides_other_right_answers_only():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"b": 1}}
    batch = [("q1", "a", "n1"), ("q1", "b", "n2"), ("q2", "b", "n3")]
    m = in_batch_mask(batch, qrels)  # columns: a, b, b, n1, n2, n3
    assert m.shape == (3, 6)
    assert m[0].tolist() == [False, True, True, False, False, False]   # row 0's own column 0 is never masked
    assert m[1].tolist() == [True, False, True, False, False, False]
    assert m[2].tolist() == [False, True, False, False, False, False]


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_at(0, 100, 1.0) == 0.0
    assert lr_at(10, 100, 1.0) == 1.0
    assert 0 < lr_at(55, 100, 1.0) < 1.0
    assert lr_at(100, 100, 1.0) == 0.0
