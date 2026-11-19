"""Checker for ch04. Don't edit. Run: python -m pytest exercises/ch04_dense"""
import random

import numpy as np
import pytest

import my_dense as mine

DOCS = {"short": "Short selling means borrowing shares and selling them",
        "mortgage": "Paying off a mortgage early saves interest",
        "betting": "Betting that a share price will fall is called shorting",
        "roth": "A Roth IRA grows tax free"}


# ---- part 1: by hand ---------------------------------------------------------------------------
def check_dot_and_norm():
    assert mine.dot((1, 2), (3, 4)) == 11
    assert mine.norm((3, 4)) == pytest.approx(5.0)


def check_normalize():
    v = mine.normalize((3, 4))
    assert v == pytest.approx([0.6, 0.8])
    assert mine.norm(v) == pytest.approx(1.0)


def check_cosine_hand_examples():
    assert mine.cosine((3, 4), (4, 3)) == pytest.approx(0.96)
    assert mine.cosine((3, 4), (-3, -4)) == pytest.approx(-1.0)
    assert mine.cosine((3, 4), (-4, 3)) == pytest.approx(0.0, abs=1e-12)
    assert mine.cosine((3, 4), (30, 40)) == pytest.approx(1.0), "cosine ignores length: same direction = 1"


def check_cosine_matches_numpy_on_random_384d():
    rng = random.Random(0)
    for _ in range(20):
        a = [rng.gauss(0, 1) for _ in range(384)]
        b = [rng.gauss(0, 1) for _ in range(384)]
        want = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert mine.cosine(a, b) == pytest.approx(want, abs=1e-9)


# ---- part 2: the real model ---------------------------------------------------------------------
def check_encode_shape_and_unit_length():
    E = np.asarray(mine.encode(list(DOCS.values())))
    assert E.shape == (4, 384)
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4), "rows must be normalised (unit length)"


def check_top_k_on_hand_made_vectors():
    docs = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.6, 0.8]])
    res = mine.top_k(np.array([0.6, 0.8]), docs, ["x", "b", "y", "a"], k=3)
    assert [d for d, _ in res] == ["a", "b", "y"], "best first; the a/b tie (both 1.0) is broken by doc id"
    assert [s for _, s in res] == pytest.approx([1.0, 1.0, 0.8])
    assert len(mine.top_k(np.array([0.6, 0.8]), docs, ["x", "b", "y", "a"], k=10)) == 4, "k > n_docs: return all"


def check_dense_finds_synonyms_bm25_cannot():
    res = mine.dense_search("gambling on equities dropping", DOCS, 4)
    assert [d for d, _ in res][0] == "betting", "MiniLM should rank the 'betting ... shorting' passage first"
    assert [s for _, s in res] == sorted([s for _, s in res], reverse=True), "results must be sorted best first"


def check_dense_paraphrase_mortgage():
    assert mine.dense_search("how do I prepay my home loan", DOCS, 1)[0][0] == "mortgage"


def check_negation_is_weak_in_embeddings():
    a, b = np.asarray(mine.encode(["a good investment", "not a good investment"]))
    assert float(a @ b) > 0.7, "embeddings capture topic more than logic: these stay very similar"


# ---- FULL checks (python -m pytest exercises\\ch04_dense --full): the real corpus -----------------------------
@pytest.mark.full
def check_full_encode_matches_reference_vectors():
    """Your encode() on 256 real passages must give the same vectors as the reference index (artifacts/)."""
    import fullkit
    ids, mat = fullkit.dense_matrix()
    corpus = fullkit.corpus()
    mine_vecs = np.asarray(mine.encode([corpus[d] for d in ids[:256]]))
    cos = np.sum(mine_vecs * mat[:256], axis=1)
    assert cos.min() > 0.999, f"worst cosine to the reference vector: {cos.min():.4f} (same model + normalised?)"


@pytest.mark.full
def check_full_dev_ndcg():
    """500 dev queries: your encode (queries) + your top_k over the 57,638 reference doc vectors → nDCG@10 = 0.388."""
    import fullkit
    from src.metrics import evaluate_run
    ids, mat = fullkit.dense_matrix()
    queries, qrels = fullkit.split("dev")
    qids = list(queries)
    qv = np.asarray(mine.encode([queries[q] for q in qids]))
    run = {q: [d for d, _ in mine.top_k(qv[i], mat, ids, 10)] for i, q in enumerate(qids)}
    assert evaluate_run(run, qrels)["ndcg@10"] == pytest.approx(0.388, abs=0.003)
