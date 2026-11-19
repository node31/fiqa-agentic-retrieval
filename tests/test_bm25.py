"""Unit tests for BM25 (hand-checkable math + live updates + global stats)."""
import math

import pytest

from src.retrieval.bm25 import BM25Index, CorpusStats, idf, tokenize

DOCS = {
    "d1": "cheap loan interest rate",
    "d2": "stock market investing for beginners",
    "d3": "loan loan loan payoff early",
    "d4": "retirement account roth ira",
}


def build(docs=DOCS):
    return BM25Index(k1=1.2, b=0.75).build(list(docs), list(docs.values()))


def test_tokenize_lowercases_stems_and_drops_stopwords():
    assert tokenize("The Loans are CHEAP") == ["loan", "cheap"]


def test_score_matches_hand_computation():
    idx = build()
    res = dict(idx.search(["cheap"], k=10))
    # 'cheap' only in d1 (4 tokens). N=4, df=1.
    # lengths: d1=4, d2=4 ('for' is a stopword), d3=5, d4=4 -> avgdl = 17/4 = 4.25
    n, avgdl = 4, idx.total_len / 4
    w = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 4 / avgdl)
    assert res["d1"] == pytest.approx(w * 1 * 2.2 / (1 + norm), rel=1e-5)
    assert set(res) == {"d1"}


def test_term_frequency_saturates():
    idx = build()
    res = dict(idx.search(["loan"], k=10))
    # d3 has tf=3 but scores LESS than 3x d1 (tf=1): BM25 saturation
    assert res["d3"] > res["d1"]
    assert res["d3"] < 3 * res["d1"]


def test_add_and_delete_are_visible_immediately():
    idx = build()
    idx.add("d5", "cheap cheap mortgage")
    assert "d5" in dict(idx.search(["mortgag"], k=10))
    assert idx.df["cheap"] == 2
    assert idx.delete("d1")
    assert "d1" not in dict(idx.search(["cheap"], k=10))
    assert idx.df["cheap"] == 1 and len(idx) == 4


def test_upsert_replaces_old_text():
    idx = build()
    idx.add("d1", "completely different words")
    assert "d1" not in dict(idx.search(["cheap"], k=10))
    assert "d1" in dict(idx.search(["complet"], k=10))


def test_global_stats_make_shards_agree_with_single_node():
    """Splitting the corpus in two and scoring with MERGED stats == one big index."""
    full = build()
    a = build({k: DOCS[k] for k in ("d1", "d2")})
    b = build({k: DOCS[k] for k in ("d3", "d4")})
    terms = ["loan", "cheap"]
    g = a.local_stats(terms).merge(b.local_stats(terms))
    merged = dict(a.search(terms, 10, g) + b.search(terms, 10, g))
    single = dict(full.search(terms, 10))
    assert merged.keys() == single.keys()
    for d in single:
        assert merged[d] == pytest.approx(single[d], rel=1e-5)
    # ...whereas LOCAL stats give a different score for d1 (the sharding IDF bug)
    local = dict(a.search(terms, 10))
    assert local["d1"] != pytest.approx(single["d1"], rel=1e-3)


def test_idf_is_positive_even_for_very_common_terms():
    assert idf(10, 10) > 0
    assert CorpusStats(0, 0.0).avgdl == 1.0
