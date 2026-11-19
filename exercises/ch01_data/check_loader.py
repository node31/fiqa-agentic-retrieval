"""Checker for ch01. Don't edit. Run: python -m pytest ch01_data"""
import pytest

import loader as mine
from common import FIQA_DIR, TINY
from src import beir_data as ref


def check_load_jsonl_corpus_matches_reference():
    got = mine.load_jsonl(TINY / "corpus.jsonl")
    want = ref.load_corpus.__wrapped__(TINY)
    assert isinstance(got, dict) and len(got) == 24, f"expected 24 docs, got {len(got)}"
    assert all(isinstance(k, str) for k in got), "doc ids must stay strings"
    assert got == want, "texts differ from the reference: did you join title + ' ' + text and .strip()?"


def check_load_jsonl_queries():
    got = mine.load_jsonl(TINY / "queries.jsonl")
    assert got["q1"] == "how does short selling work"


def check_load_qrels():
    got = mine.load_qrels(TINY / "qrels" / "dev.tsv")
    assert "query-id" not in got, "you included the header line as a query"
    assert got["q1"] == {"1": 1, "2": 1}
    assert got == ref.load_qrels.__wrapped__("dev", TINY)


def check_load_split_keeps_only_split_queries():
    q, r = mine.load_split(TINY, "dev")
    wq, wr = ref.load_split("dev", TINY)
    assert set(q) == set(r), "queries must be exactly the ones that appear in the qrels"
    assert q == wq and r == wr


def check_stats_tiny():
    s = mine.stats(TINY, "dev")
    assert s["n_docs"] == 24 and s["n_queries"] == 4 and s["n_pairs"] == 7
    assert s["avg_gold"] == pytest.approx(1.75)


# ---- FULL checks (python -m pytest exercises\\ch01_data --full): the real 57,638-doc FiQA -----------------
@pytest.mark.full
def check_full_stats():
    s = mine.stats(FIQA_DIR, "dev")
    assert (s["n_docs"], s["n_queries"], s["n_pairs"]) == (57_638, 500, 1_238)
    t = mine.stats(FIQA_DIR, "test")
    assert (t["n_queries"], t["n_pairs"]) == (648, 1_706)
    tr = mine.stats(FIQA_DIR, "train")
    assert (tr["n_queries"], tr["n_pairs"]) == (5_500, 14_166)


@pytest.mark.full
def check_full_matches_reference_loader():
    mine_corpus = mine.load_jsonl(FIQA_DIR / "corpus.jsonl")
    ref_corpus = ref.load_corpus.__wrapped__(FIQA_DIR)
    assert len(mine_corpus) == len(ref_corpus)
    diff = [d for d in ref_corpus if mine_corpus.get(d) != ref_corpus[d]]
    assert not diff, f"{len(diff)} docs differ from the reference, e.g. {diff[:3]}"
    q, r = mine.load_split(FIQA_DIR, "dev")
    rq, rr = ref.load_split("dev", FIQA_DIR)
    assert q == rq and r == rr
