"""Checker for ch06. Don't edit. Run: python -m pytest exercises/ch06_pipeline"""
import pytest

import mini_pipeline as mine
from common import FIQA_DIR, TINY
from src.beir_data import load_corpus, load_split
from src.metrics import evaluate_run
from src.retrieval.bm25 import BM25Index, tokenize


@pytest.fixture(scope="module")
def out():
    return mine.run(TINY)


def check_structure(out):
    queries, _ = load_split("dev", TINY)
    corpus = load_corpus.__wrapped__(TINY)
    for s in mine.SYSTEMS:
        assert set(out["rankings"][s]) == set(queries), f"{s}: one ranking per dev query"
        for qid, ids in out["rankings"][s].items():
            assert len(ids) <= 10 and len(ids) == len(set(ids)), f"{s}/{qid}: at most 10, no duplicates"
            assert set(ids) <= set(corpus), f"{s}/{qid}: unknown doc ids"


def check_bm25_rankings_match_reference(out):
    corpus = load_corpus.__wrapped__(TINY)
    ref = BM25Index().build(list(corpus), list(corpus.values()))
    queries, _ = load_split("dev", TINY)
    for qid, q in queries.items():
        want = [d for d, _ in ref.search(list(dict.fromkeys(tokenize(q))), 10)]
        assert out["rankings"]["bm25"][qid] == want, f"BM25 ranking for {qid} differs from the reference"


def check_metrics_consistent_with_rankings(out):
    _, qrels = load_split("dev", TINY)
    for s in mine.SYSTEMS:
        want = evaluate_run(out["rankings"][s], qrels)
        for key in ("ndcg@10", "recall@10", "mrr@10"):
            assert out["metrics"][s][key] == pytest.approx(want[key]), f"{s} {key}"


def check_hybrid_uses_both_signals(out):
    # with 24 docs every query hits both lists; the hybrid top-10 should not simply equal one of them everywhere
    h, b, d = (out["rankings"][s] for s in ("hybrid", "bm25", "dense"))
    assert any(h[q] != b[q] for q in h) and any(h[q] != d[q] for q in h), \
        "hybrid is identical to BM25 (or dense) for every query: are you fusing normalised scores?"


# ---- FULL check (python -m pytest exercises\\ch06_pipeline --full): Part A on the real data -------------------
@pytest.mark.full
def check_full_reproduces_part_a_numbers():
    """Your whole pipeline on 57,638 docs and 500 dev queries must reproduce the project's Part A table:
    BM25 0.264 < dense 0.388 < hybrid 0.419 (docs/walkthroughs/part_a_results.md).
    First run ~11 min (encoding the corpus with YOUR encode); later runs ~3 min (cached)."""
    full = mine.run(FIQA_DIR)
    got = {s: full["metrics"][s]["ndcg@10"] for s in mine.SYSTEMS}
    assert got["bm25"] == pytest.approx(0.264, abs=0.003), got
    assert got["dense"] == pytest.approx(0.388, abs=0.003), got
    assert got["hybrid"] == pytest.approx(0.419, abs=0.003), got
