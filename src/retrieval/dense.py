"""Dense (embedding) retrieval: encoders + an exact vector index with deletes.

WHAT
    * ``Embedder``: wraps a sentence-transformers model. It turns text into a
      384-d unit vector so that "similar meaning" becomes "small angle".
    * ``HashEmbedder``: a no-download stand-in (feature hashing), used by tests
      and toys so they run offline in milliseconds.
    * ``DenseIndex``: a faiss exact inner-product index keyed by string doc ids,
      with add, remove and top-k search.

WHY EXACT (FLAT) SEARCH
    57K x 384 floats = 88 MB, and a brute-force dot product over 19K vectors
    per shard takes about 1 ms. Approximate indexes (HNSW, IVF) only pay off
    with millions of vectors, and they cost recall plus tuning work. The
    assignment explicitly allows flat search. See docs/03_vector_indexes.md
    and examples/06_faiss_flat_vs_hnsw.py.

WHY INNER PRODUCT
    We L2-normalise every vector, so inner product == cosine similarity, which
    is what these models were trained with.

PITFALLS
    * Normalise BOTH queries and documents, otherwise long docs win.
    * Query-side instruction prefix for BGE (see config.MODELS).
    * faiss wants float32, C-contiguous, shape (n, d). A float64 array gives a
      cryptic error.
    * ``IndexIDMap2`` is what makes remove_ids/reconstruct work with our own ids.

SEE ALSO
    examples/04_embeddings_cosine.py, 05_bge_query_prefix.py, 06_faiss_flat_vs_hnsw.py
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Sequence

import faiss
import numpy as np

from src.config import MODELS, ModelSpec


class Embedder:
    """sentence-transformers encoder that applies the model's query prefix and normalises."""

    def __init__(self, spec: ModelSpec):
        # WHY lazy import: importing torch takes ~3 s and ~300 MB, and tests/toys
        # that use HashEmbedder should not pay that.
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.hf_name, device="cpu")
        self.model.max_seq_length = spec.max_seq_length

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size=32)

    def encode_docs(self, texts: Sequence[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self._encode(list(texts), batch_size=batch_size, progress=progress)

    def _encode(self, texts: list[str], batch_size: int, progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=progress)
        return np.ascontiguousarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic "embedding" by hashing words into buckets (the hashing trick).

    It has no semantics (synonyms do not match), but it is lexical-ish, needs no
    download and is fast. That makes it perfect for testing plumbing such as
    sharding, writes and migration. Two different ``seed`` values behave like two
    different "models", which is handy for migration tests.
    """

    def __init__(self, dim: int = 384, seed: str = "minilm"):
        self.dim, self.seed = dim, seed

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            h = hashlib.md5((self.seed + w).encode()).digest()
            v[int.from_bytes(h[:4], "little") % self.dim] += 1.0 if h[4] & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    encode_docs = lambda self, texts, **_: self.encode_queries(texts)  # noqa: E731


_EMBEDDERS: dict[str, object] = {}
_EMB_LOCK = threading.Lock()


def get_embedder(model_key: str):
    """Process-wide singleton per model (loading a model takes seconds and ~100 MB).

    Set ``EMBED_BACKEND=hash`` to use HashEmbedder everywhere (tests, CI, toys).
    """
    with _EMB_LOCK:  # WHY: two threads must not both load the same model
        if model_key not in _EMBEDDERS:
            if os.environ.get("EMBED_BACKEND", "st") == "hash":
                _EMBEDDERS[model_key] = HashEmbedder(MODELS[model_key].dim, seed=model_key)
            else:
                _EMBEDDERS[model_key] = Embedder(MODELS[model_key])
        return _EMBEDDERS[model_key]


class DenseIndex:
    """Exact cosine search over unit vectors, addressable by string doc id."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._int_of: dict[str, int] = {}  # faiss only accepts int64 ids
        self._str_of: dict[int, str] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self._int_of)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._int_of

    def add(self, doc_ids: Sequence[str], vecs: np.ndarray) -> None:
        """Insert or replace vectors. ``vecs`` must be (n, dim) and normalised."""
        for d in doc_ids:
            if d in self._int_of:
                self.remove(d)  # WHY: upsert = remove old vector, then add the new one
        ints = np.arange(self._next, self._next + len(doc_ids), dtype=np.int64)
        self._next += len(doc_ids)
        for d, i in zip(doc_ids, ints.tolist()):
            self._int_of[d], self._str_of[i] = i, d
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype=np.float32), ints)

    def remove(self, doc_id: str) -> bool:
        i = self._int_of.pop(doc_id, None)
        if i is None:
            return False
        del self._str_of[i]
        # Complexity: O(N) for a flat index (it compacts the array); ~1 ms at 19K.
        self.index.remove_ids(np.array([i], dtype=np.int64))
        return True

    def search(self, qvec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if len(self) == 0:
            return []
        q = np.ascontiguousarray(np.asarray(qvec, dtype=np.float32).reshape(1, -1))
        scores, ids = self.index.search(q, min(k, len(self)))
        out = [(self._str_of[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        out.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break, as in BM25
        return out
