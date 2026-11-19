"""Checker for ch03. Don't edit. Run: python -m pytest exercises/ch03_bm25"""
import functools
import json
import math

import pytest

import my_bm25 as mine
from common import FIQA_DIR, TINY
from src.retrieval.bm25 import BM25Index
from src.retrieval.bm25 import tokenize as ref_tokenize

THREE = {"d1": "short selling means borrowing shares",
         "d2": "a short story about selling a house and selling a car",
         "d3": "mortgage rates rose"}


def _tiny_corpus():
    out = {}
    for line in (TINY / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["_id"]] = (r.get("title", "") + " " + r["text"]).strip()
    return out


@pytest.mark.parametrize("text", [
    "What is short selling?",
    "I'm not paying taxes on my 401(k) withdrawals early",
    "Investing in the stocks, not bonds!",
    "Is investing in Index-Funds better than SHORTING stocks? Not always!",
])
def check_tokenize_matches_reference(text):
    assert mine.tokenize(text) == ref_tokenize(text)


def check_idf():
    assert mine.idf(1000, 1) == pytest.approx(6.5023, abs=1e-3)
    assert mine.idf(1000, 500) == pytest.approx(math.log(2), abs=1e-6)  # (1000-500+.5)/(500+.5) = 1 -> ln(1+1)


def check_build_index_basics():
    idx = mine.build_index(THREE)
    assert idx.n_docs == 3
    assert idx.doc_len == {"d1": 5, "d2": 7, "d3": 3}
    assert idx.avgdl == pytest.approx(5.0)
    assert sorted(idx.postings["sell"]) == [("d1", 1), ("d2", 2)]


def check_three_doc_example():
    res = dict(mine.search(mine.build_index(THREE), "short selling", 3))
    assert set(res) == {"d1", "d2"}, "d3 shares no term with the query, so it must not be returned"
    assert res["d2"] == pytest.approx(0.9848, abs=1e-3)
    assert res["d1"] == pytest.approx(0.9400, abs=1e-3)


def check_repeated_query_term_counts_once():
    idx = mine.build_index(THREE)
    assert mine.search(idx, "short short selling", 3) == mine.search(idx, "short selling", 3)


def check_tiny_fiqa_matches_reference():
    corpus = _tiny_corpus()
    idx = mine.build_index(corpus)
    ref = BM25Index().build(list(corpus), list(corpus.values()))
    for q in ["how does short selling work", "roth ira vs traditional 401k taxes",
              "how big should an emergency fund be", "pay off mortgage early or invest", "dividend etf"]:
        got = mine.search(idx, q, 10)
        want = ref.search(list(dict.fromkeys(ref_tokenize(q))), 10)
        assert [d for d, _ in got] == [d for d, _ in want], f"ranking differs for {q!r}"
        for (_, s1), (_, s2) in zip(got, want):
            assert s1 == pytest.approx(s2, abs=1e-4)


# ---- FULL checks (python -m pytest exercises\\ch03_bm25 --full): your BM25 on all 57,638 docs -----------------
@functools.cache
def _my_full_index():
    import fullkit
    return mine.build_index(fullkit.corpus())          # ~1 min in pure Python; built once for both checks


@pytest.mark.full
def check_full_doc_336018():
    """The worked example in guide chapter 3: 'What is short selling?' on the real corpus."""
    res = dict(mine.search(_my_full_index(), "What is short selling?", 5))
    assert res["336018"] == pytest.approx(10.680, abs=1e-2)
    assert res["127585"] == pytest.approx(10.013, abs=1e-2)


@pytest.mark.full
def check_full_dev_ndcg_matches_reference():
    """All 500 dev queries: same top-10 as the reference BM25, and nDCG@10 = 0.264 (the number in the docs)."""
    import fullkit
    from src.metrics import evaluate_run
    queries, qrels = fullkit.split("dev")
    idx = _my_full_index()
    run = {q: [d for d, _ in mine.search(idx, t, 10)] for q, t in queries.items()}
    ref_run = {q: [d for d, _ in lst[:10]] for q, lst in fullkit.bm25_run("dev").items()}
    # WHY overlap, not equality: the reference uses float32, yours float64, so exact ties can swap order.
    overlap = sum(len(set(run[q]) & set(ref_run[q])) for q in queries) / (10 * len(queries))
    assert overlap > 0.99, f"only {overlap:.1%} of your top-10s match the reference"
    assert evaluate_run(run, qrels)["ndcg@10"] == pytest.approx(0.264, abs=0.002)
