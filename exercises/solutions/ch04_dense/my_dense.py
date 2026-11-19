"""SOLUTION for ch04: vectors by hand, then real embeddings.

Try the exercise yourself first (exercises/ch04_dense/my_dense.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch04_dense
"""
from __future__ import annotations

import math
from typing import Sequence

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---- part 1: by hand (plain Python lists, NO numpy) ----------------------------------------------------
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):                         # PITFALL: zip() would silently drop the extra dimensions
        raise ValueError(f"length mismatch {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))                  # Pythagoras in n dimensions


def normalize(a: Sequence[float]) -> list[float]:
    n = norm(a)
    if n == 0:                                   # the zero vector has no direction
        raise ValueError("cannot normalize the zero vector")
    return [x / n for x in a]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # cos(angle) = a·b / (|a||b|):  1 = same direction, 0 = unrelated (90°), -1 = opposite
    return dot(a, b) / (norm(a) * norm(b))


# ---- part 2: the real model -------------------------------------------------------------------
_MODEL = None


def encode(texts: list[str]):
    """(len(texts), 384) numpy array, each row length 1."""
    global _MODEL
    if _MODEL is None:
        # WHY lazy import + cache: loading takes ~1.5 s (see results/microbench.json); do it once, and only
        # when needed, so importing this module (part 1) stays instant.
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
    # WHY normalize_embeddings=True: then cosine == dot product, which is what faiss IndexFlatIP computes.
    return _MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def top_k(query_vec, doc_vecs, doc_ids: list[str], k: int = 10) -> list[tuple[str, float]]:
    """All cosines at once, then the k best. This is the per-query cost once doc vectors exist."""
    import numpy as np

    scores = np.asarray(doc_vecs) @ np.asarray(query_vec)   # (n_docs,) = n dot products = n cosines (unit vectors)
    n = len(doc_ids)
    # WHY argpartition: picking the best k of 57,638 doesn't need a full sort (O(n) vs O(n log n)).
    # Take 20 extra candidates so ties at the cut-off are resolved by doc id, not by memory order.
    cand = np.argpartition(-scores, k + 20)[: k + 20] if n > k + 20 else range(n)
    ranked = sorted(((doc_ids[i], float(scores[i])) for i in cand), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def dense_search(query: str, docs: dict[str, str], k: int = 10) -> list[tuple[str, float]]:
    ids = list(docs)                              # fix an order so row i ↔ ids[i]
    doc_vecs = encode([docs[i] for i in ids])     # (n_docs, 384). In the real system this is precomputed ONCE.
    q_vec = encode([query])[0]                    # (384,)
    return top_k(q_vec, doc_vecs, ids, k)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("cos((3,4),(4,3)) =", cosine((3, 4), (4, 3)))      # 0.96
    docs = {"short": "Short selling means borrowing shares and selling them",
            "mortgage": "Paying off a mortgage early saves interest",
            "betting": "Betting that a share price will fall is called shorting",
            "roth": "A Roth IRA grows tax free"}
    for q in ["gambling on equities dropping", "how do I prepay my home loan"]:
        print(q, "->", [(d, round(s, 3)) for d, s in dense_search(q, docs, 4)])
    a, b = encode(["a good investment", "not a good investment"])
    print("negation cosine:", round(float(a @ b), 3))         # ~0.81: embeddings barely notice "not"
    if "--full" in sys.argv:   # your encode + top_k over the 57,638 reference doc vectors (artifacts/)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import fullkit
        ids, mat = fullkit.dense_matrix()
        corpus = fullkit.corpus()
        for q in ["What is short selling?", "how do I prepay my home loan"]:
            print(f"\n[full] {q!r}")
            for d, s in top_k(encode([q])[0], mat, ids, 3):
                print(f"  {d:>7s} {s:.3f}  {corpus[d][:90]!r}")
