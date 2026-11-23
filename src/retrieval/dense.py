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
    task spec explicitly allows flat search. See docs/topics/03_vector_indexes.md
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

HOW TO READ THIS FILE (top to bottom)
    1. Embedder / HashEmbedder: text -> vectors. Both expose the same two methods,
       encode_queries(texts) and encode_docs(texts), each returning a float32
       array of shape (len(texts), dim) whose rows have length 1.
    2. get_embedder(): picks and caches one encoder per model key.
    3. DenseIndex: stores doc vectors and answers "which stored vectors are closest
       to this query vector?".

    Words used below:
    * embedding / vector : a fixed-length list of floats (here 384) that a model
      produces for a text; texts with similar meaning get vectors pointing in
      similar directions.
    * bi-encoder : a model that embeds the query and each passage SEPARATELY, so all
      passage vectors can be computed once, offline (contrast: cross-encoder in rerank.py).
    * normalise (L2) : divide a vector by its length so the length becomes 1. For unit
      vectors, dot product = cosine of the angle between them, in [-1, 1].
      Example: a=(3,4) -> (0.6,0.8); b=(1,0); dot = 0.6 = cos(angle).
    * inner product (IP) : the dot product, sum(a_i * b_i).
    * exact / flat search : compare the query with EVERY stored vector (no shortcuts),
      so results are always the true top-k.
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Sequence

import faiss
import numpy as np

from src.config import MODELS, ModelSpec, model_path


class Embedder:
    """sentence-transformers encoder that applies the model's query prefix and normalises."""

    def __init__(self, spec: ModelSpec):
        # WHY lazy import: importing torch takes ~3 s and ~300 MB, and tests/toys
        # that use HashEmbedder should not pay that.
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        # Weights are downloaded from the Hugging Face hub on first use and then
        # cached on disk (HF_HUB_OFFLINE=1 forces use of the cache). device="cpu":
        # this project assumes no GPU.
        self.model = SentenceTransformer(model_path(spec), device="cpu")
        # Passages longer than this many model tokens (word pieces, not words) are
        # cut off; only the beginning of a long passage influences its vector.
        self.model.max_seq_length = spec.max_seq_length

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # WHY a prefix only on queries: BGE was trained with an instruction in front
        # of queries ("Represent this sentence for searching relevant passages: ")
        # and nothing in front of passages. MiniLM's prefix is "" so this is a no-op.
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size=32)

    def encode_docs(self, texts: Sequence[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self._encode(list(texts), batch_size=batch_size, progress=progress)

    def _encode(self, texts: list[str], batch_size: int, progress: bool = False) -> np.ndarray:
        # batch_size = how many texts go through the model at once (speed vs memory).
        # normalize_embeddings=True makes every row unit length, so a later dot
        # product equals cosine similarity. Output shape: (len(texts), dim).
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=progress)
        # WHY: faiss needs float32 in one contiguous memory block (see PITFALLS above).
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
        # The hashing trick: every word is hashed to one of `dim` buckets, and the
        # bucket gets +1 or -1. Two texts sharing words share buckets, so their dot
        # product is high, a crude stand-in for "similar meaning".
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            # WHY md5 and not hash(): Python's built-in str hash is randomised per
            # process, so vectors would differ between runs. md5 is stable.
            # Mixing the seed in gives each "model" its own bucket layout.
            h = hashlib.md5((self.seed + w).encode()).digest()
            # Bytes 0-3 choose the bucket; the lowest bit of byte 4 chooses the sign.
            # WHY a random sign: unrelated words colliding in a bucket then tend to
            # cancel out instead of always adding up (less bias from collisions).
            v[int.from_bytes(h[:4], "little") % self.dim] += 1.0 if h[4] & 1 else -1.0
        # Normalise to length 1, like the real models. Empty text stays all zeros.
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # Shape (len(texts), dim); for an empty input, a (0, dim) array so callers
        # can still read .shape[1].
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    # Docs and queries are embedded the same way here (no prefix); **_ swallows
    # keyword arguments such as batch_size/progress that Embedder.encode_docs accepts.
    encode_docs = lambda self, texts, **_: self.encode_queries(texts)  # noqa: E731


# Cache: model key (e.g. "minilm", "bge-small") -> loaded embedder, shared by all
# threads in the process. The lock guards the check-then-load in get_embedder().
_EMBEDDERS: dict[str, object] = {}
_EMB_LOCK = threading.Lock()


def get_embedder(model_key: str):
    """Process-wide singleton per model (loading a model takes seconds and ~100 MB).

    Set ``EMBED_BACKEND=hash`` to use HashEmbedder everywhere (tests, CI, toys).
    """
    with _EMB_LOCK:  # WHY: two threads must not both load the same model
        if model_key not in _EMBEDDERS:
            # EMBED_BACKEND is read only on the FIRST request for a model key;
            # PITFALL: changing it later in the same process has no effect.
            # seed=model_key makes "minilm" and "bge-small" hash differently, so they
            # behave like two different models in migration tests.
            if os.environ.get("EMBED_BACKEND", "st") == "hash":
                _EMBEDDERS[model_key] = HashEmbedder(MODELS[model_key].dim, seed=model_key)
            else:
                _EMBEDDERS[model_key] = Embedder(MODELS[model_key])
        return _EMBEDDERS[model_key]


class DenseIndex:
    """Exact cosine search over unit vectors, addressable by string doc id."""

    def __init__(self, dim: int):
        self.dim = dim
        # Two faiss layers:
        # * IndexFlatIP: a plain array of vectors; search = dot product of the query
        #   with every row, then keep the k largest (exact, "flat").
        # * IndexIDMap2: a wrapper that lets us attach our own int64 id to each vector
        #   (instead of faiss's 0..n-1 position, which shifts after a removal) and
        #   remove/reconstruct by that id.
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        # Two-way mapping between our string doc ids ("12345") and faiss int ids.
        self._int_of: dict[str, int] = {}  # faiss only accepts int64 ids
        self._str_of: dict[int, str] = {}
        # Next unused faiss id. It only grows, so an id is never reused even after
        # a remove; a stale id can therefore never point at a different doc.
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
        # Give the batch consecutive fresh ids, e.g. _next=5 and 3 docs -> [5, 6, 7].
        ints = np.arange(self._next, self._next + len(doc_ids), dtype=np.int64)
        self._next += len(doc_ids)
        for d, i in zip(doc_ids, ints.tolist()):
            self._int_of[d], self._str_of[i] = i, d
        # Row n of vecs is stored under id ints[n], so doc_ids and vecs must be in
        # the same order. PITFALL: vectors are NOT normalised here; pass unit vectors.
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype=np.float32), ints)

    def remove(self, doc_id: str) -> bool:
        # Unlike BM25's tombstones, the vector is physically removed from faiss.
        i = self._int_of.pop(doc_id, None)
        if i is None:
            return False
        del self._str_of[i]
        # Complexity: O(N) for a flat index (it compacts the array); ~1 ms at 19K.
        self.index.remove_ids(np.array([i], dtype=np.int64))
        return True

    def search(self, qvec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        # Returns up to k (doc_id, cosine score) pairs, best first. Scores are in
        # [-1, 1]; for these models scores typically sit around 0.2-0.8 (see hybrid.py).
        if len(self) == 0:
            return []
        # faiss searches a BATCH of queries, so turn a (dim,) vector into a (1, dim)
        # matrix of float32. reshape(1, -1): "1 row, as many columns as needed".
        q = np.ascontiguousarray(np.asarray(qvec, dtype=np.float32).reshape(1, -1))
        # scores and ids both have shape (1, k'): row 0 belongs to our single query.
        # WHY min(k, len): never ask for more results than stored vectors.
        scores, ids = self.index.search(q, min(k, len(self)))
        # faiss pads missing results with id -1, so skip those; map ints back to strings.
        out = [(self._str_of[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        out.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break, as in BM25
        return out
