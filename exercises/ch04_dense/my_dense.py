"""ch04: vectors by hand, then real embeddings. Part 1 is pure Python; part 2 uses sentence-transformers.

Run the checker:   python -m pytest exercises/ch04_dense
Run the demo:      python exercises/ch04_dense/my_dense.py
"""
from __future__ import annotations

import math  # noqa: F401
from typing import Sequence

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---- part 1: by hand (plain Python lists, NO numpy) ----------------------------------------------------
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Σ a_i·b_i.  dot((1, 2), (3, 4)) == 11"""
    raise NotImplementedError("TODO (ch04): implement dot")


def norm(a: Sequence[float]) -> float:
    """Length: sqrt(Σ a_i²).  norm((3, 4)) == 5.0"""
    raise NotImplementedError("TODO (ch04): implement norm")


def normalize(a: Sequence[float]) -> list[float]:
    """Scale to length 1.  normalize((3, 4)) == [0.6, 0.8]"""
    raise NotImplementedError("TODO (ch04): implement normalize")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """dot(a, b) / (norm(a)·norm(b)).  cosine((3, 4), (4, 3)) == 0.96"""
    raise NotImplementedError("TODO (ch04): implement cosine")


# ---- part 2: the real model -------------------------------------------------------------------
_MODEL = None  # cache: loading the model takes seconds, so do it once


def encode(texts: list[str]):
    """Return a numpy array of shape (len(texts), 384) with UNIT-LENGTH rows (normalize_embeddings=True)."""
    raise NotImplementedError("TODO (ch04): load SentenceTransformer(MODEL_NAME, device='cpu') once, then encode")


def top_k(query_vec, doc_vecs, doc_ids: list[str], k: int = 10) -> list[tuple[str, float]]:
    """Score every row of `doc_vecs` (shape (n_docs, 384), unit rows) against one unit `query_vec` (shape (384,))
    and return the top-k [(doc_id, score)] best first, ties by doc_id. Row i belongs to doc_ids[i].

    WHY a separate function: in the real system the doc vectors are computed ONCE (about 10 minutes for 57,638 docs)
    and saved. Each query then only pays for encode(query) + top_k. Hint: `doc_vecs @ query_vec` gives all scores.
    """
    raise NotImplementedError("TODO (ch04): implement top_k")


def dense_search(query: str, docs: dict[str, str], k: int = 10) -> list[tuple[str, float]]:
    """Encode docs + query, then top_k. Fine for a few docs; for the full corpus encode the docs once (see ch06)."""
    raise NotImplementedError("TODO (ch04): implement dense_search with encode() + top_k()")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    docs = {"short": "Short selling means borrowing shares and selling them",
            "mortgage": "Paying off a mortgage early saves interest",
            "betting": "Betting that a share price will fall is called shorting",
            "roth": "A Roth IRA grows tax free"}
    for q in ["gambling on equities dropping", "how do I prepay my home loan"]:
        print(q, "->", [(d, round(s, 3)) for d, s in dense_search(q, docs, 4)])
    if "--full" in sys.argv:   # your encode + top_k over the 57,638 reference doc vectors (artifacts/)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import fullkit
        ids, mat = fullkit.dense_matrix()
        corpus = fullkit.corpus()
        for q in ["What is short selling?", "how do I prepay my home loan"]:
            print(f"\n[full] {q!r}")
            for d, s in top_k(encode([q])[0], mat, ids, 3):
                print(f"  {d:>7s} {s:.3f}  {corpus[d][:90]!r}")
