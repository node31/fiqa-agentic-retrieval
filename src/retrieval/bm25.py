"""BM25 lexical retrieval, written from scratch, with live add/delete.

WHAT
    An inverted index plus the Okapi BM25 scoring function. Given a query, it
    returns the top-k passages ranked by how well their *words* match.

WHY FROM SCRATCH (see docs/adr/0001-own-bm25.md)
    * rank_bm25 has no deletes and scores every document for every query
      (O(N) per query term), which is too slow for a 50 ms budget.
    * Pyserini needs Java.
    * Part B needs two things libraries hide: (1) **global corpus statistics**
      injected at query time (sharding), and (2) **live upserts/deletes**.

THE FORMULA (docs/01_lexical_retrieval_bm25.md works an example by hand)
    score(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene variant, always > 0)

    tf   = how often t occurs in d          N     = number of live docs
    df   = how many docs contain t          |d|   = length of d in tokens
    k1   = tf saturation (1.2)              avgdl = average doc length
    b    = length normalisation (0.75)

DATA STRUCTURES
    base segment : a scipy CSC matrix (docs × terms) built in bulk at startup.
                   Column j holds the postings list of term j: which docs
                   contain it and how often. Reading a column is O(df).
    delta segment: a Python dict term -> [(doc_idx, tf)] for docs added later.
    alive mask   : deletes just flip a bit; dead docs are masked at query time.
    This is a two-segment version of Lucene's LSM-style segments. Merging the
    delta into the base ("compaction") is left as a drill in docs/DRILLS.md.

HOW IT FITS
    Each shard owns one BM25Index over its slice of the corpus. The router asks
    every shard for ``local_stats(terms)``, sums them into global statistics, and
    passes them back into ``search`` so every shard scores with the SAME IDF.
    That makes sharded scores comparable and equal to single-node scores.

PITFALLS
    * Local IDF: if each shard used its own N/df, the same doc would get a
      different score depending on which shard it lives on. See
      examples/12_sharded_idf_bug.py.
    * Query and documents MUST go through the identical tokenize() function.

SEE ALSO
    examples/00_tokenization.py, 01_tf_idf_by_hand.py, 02_bm25_by_hand.py,
    03_inverted_index_sparse.py; tests/test_bm25.py
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from src.config import BM25_B, BM25_K1

# --- Tokenisation -------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# WHY this list: Lucene's classic English stop set (33 words), so our BM25 is
# comparable to the published Pyserini/Anserini BEIR baselines. Note it DOES drop
# "no"/"not". Negation is lost for lexical matching, a known BM25 weakness that
# dense models partly fix (exercise in docs/01_lexical_retrieval_bm25.md).
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split()
)

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("english")
except ImportError:  # pragma: no cover - stemming is optional
    _STEMMER = None


@lru_cache(maxsize=1_000_000)
def _stem(word: str) -> str:
    # WHY cache: FiQA has ~7.5M tokens but only ~100K distinct words, so caching
    # makes stemming effectively free after the first pass.
    return _STEMMER.stemWord(word) if _STEMMER else word


def tokenize(text: str) -> list[str]:
    """Lowercase -> split on non-alphanumerics -> drop stopwords -> stem.

    >>> tokenize("Investing in the stocks, not bonds!")
    ['invest', 'stock', 'bond']
    """
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# --- Corpus statistics (the sharding-relevant part) -----------------------------


@dataclass
class CorpusStats:
    """The numbers IDF and length-normalisation need. Summable across shards.

    >>> a = CorpusStats(10, 100.0, {"loan": 2}); b = CorpusStats(5, 50.0, {"loan": 1})
    >>> a.merge(b).df["loan"], a.merge(b).avgdl
    (3, 10.0)
    """

    n_docs: int
    total_len: float
    df: dict[str, int] = field(default_factory=dict)

    @property
    def avgdl(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 1.0

    def merge(self, other: "CorpusStats") -> "CorpusStats":
        df = dict(self.df)
        for t, c in other.df.items():
            df[t] = df.get(t, 0) + c
        return CorpusStats(self.n_docs + other.n_docs, self.total_len + other.total_len, df)

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "total_len": self.total_len, "df": self.df}

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusStats":
        return cls(int(d["n_docs"]), float(d["total_len"]), dict(d["df"]))


def idf(n_docs: int, df: int) -> float:
    """Lucene BM25 IDF. Rare terms (small df) get large weights.

    >>> round(idf(1000, 1), 2), round(idf(1000, 500), 2)
    (6.5, 0.69)
    """
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


# --- The index -------------------------------------------------------------------


class BM25Index:
    """Inverted index + BM25 scorer supporting bulk build, add, delete and global stats."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}  # term -> column in the base matrix
        self.terms: list[str] = []  # column -> term (inverse of vocab)
        self.doc_ids: list[str] = []  # internal row -> external doc id
        self.row_of: dict[str, int] = {}  # external doc id -> LIVE internal row
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.df: Counter[str] = Counter()  # document frequency over LIVE docs only
        self.total_len = 0.0
        self._base_csc: sparse.csc_matrix | None = None  # fast column (term) access
        self._base_csr: sparse.csr_matrix | None = None  # fast row (doc) access, for deletes
        self._n_base = 0
        self._delta: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._delta_terms: dict[int, Counter] = {}

    # ---- building ----------------------------------------------------------
    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        """Bulk-index documents into the base segment. Call once, on an empty index."""
        assert not self.doc_ids, "build() is for an empty index; use add() afterwards"
        rows, cols, vals = [], [], []
        lens = np.zeros(len(doc_ids), dtype=np.float32)
        for r, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lens[r] = sum(counts.values())
            for term, tf in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(r)
                cols.append(j)
                vals.append(tf)
                self.df[term] += 1
        self.terms = [None] * len(self.vocab)  # type: ignore[list-item]
        for t, j in self.vocab.items():
            self.terms[j] = t
        coo = sparse.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(doc_ids), len(self.vocab))
        )
        self._base_csc, self._base_csr = coo.tocsc(), coo.tocsr()
        self._n_base = len(doc_ids)
        self.doc_ids = list(doc_ids)
        self.row_of = {d: i for i, d in enumerate(doc_ids)}
        self.doc_len = lens
        self.alive = np.ones(len(doc_ids), dtype=bool)
        self.total_len = float(lens.sum())
        return self

    # ---- live updates ----------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        """Insert or replace one document (replace = delete old row + append new row)."""
        if doc_id in self.row_of:
            self.delete(doc_id)
        counts = Counter(tokenize(text))
        row = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.row_of[doc_id] = row
        # WHY np.append: O(N) copy per write, ~20 µs at this size; fine for a demo,
        # a production index would use chunked arrays.
        self.doc_len = np.append(self.doc_len, np.float32(sum(counts.values())))
        self.alive = np.append(self.alive, True)
        self.total_len += float(self.doc_len[row])
        for term, tf in counts.items():
            self._delta[term].append((row, tf))
            self.df[term] += 1
        self._delta_terms[row] = counts

    def delete(self, doc_id: str) -> bool:
        """Remove a document. Returns False if it was not present."""
        row = self.row_of.pop(doc_id, None)
        if row is None:
            return False
        self.alive[row] = False  # WHY: tombstone; postings stay but are masked out
        self.total_len -= float(self.doc_len[row])
        for term in self._terms_of_row(row):
            self.df[term] -= 1
            if self.df[term] <= 0:
                del self.df[term]
        return True

    def _terms_of_row(self, row: int) -> Iterable[str]:
        if row < self._n_base:
            csr = self._base_csr
            return (self.terms[j] for j in csr.indices[csr.indptr[row]: csr.indptr[row + 1]])
        return self._delta_terms.pop(row, Counter()).keys()

    # ---- statistics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.row_of)

    def local_stats(self, terms: Iterable[str]) -> CorpusStats:
        """This index's N, total length and df for just the requested terms."""
        return CorpusStats(len(self.row_of), self.total_len, {t: self.df.get(t, 0) for t in terms})

    # ---- search ----------------------------------------------------------------
    def search(self, terms: Sequence[str], k: int = 10,
               stats: CorpusStats | None = None) -> list[tuple[str, float]]:
        """Score docs for pre-tokenised, de-duplicated ``terms`` and return the top-k.

        Args:
            terms: output of ``tokenize(query)``, ideally de-duplicated.
            k: number of results.
            stats: GLOBAL statistics from the router. ``None`` = use local stats
                (correct only for a single-node index).

        Complexity: O(Σ_t df(t)) to accumulate + O(N) to pick the top-k.
        """
        if not self.doc_ids:
            return []
        stats = stats or self.local_stats(terms)
        k1, b = self.k1, self.b
        # Length normalisation for EVERY doc, computed once per query (vectorised).
        norm = k1 * (1.0 - b + b * self.doc_len / max(stats.avgdl, 1e-9))
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        csc = self._base_csc
        for t in terms:
            df = stats.df.get(t, 0)
            if df <= 0:
                continue  # WHY: term absent everywhere -> contributes nothing
            w = idf(stats.n_docs, df)
            j = self.vocab.get(t)
            if csc is not None and j is not None:
                lo, hi = csc.indptr[j], csc.indptr[j + 1]
                rows, tf = csc.indices[lo:hi], csc.data[lo:hi]
                scores[rows] += w * tf * (k1 + 1) / (tf + norm[rows])
            for row, tf in self._delta.get(t, ()):
                scores[row] += w * tf * (k1 + 1) / (tf + norm[row])
        scores[~self.alive] = 0.0
        return _top_k(scores, self.doc_ids, k)


def _top_k(scores: np.ndarray, ids: Sequence[str], k: int) -> list[tuple[str, float]]:
    """Top-k positive scores, ties broken by doc id so results are deterministic."""
    nz = np.flatnonzero(scores > 0)
    if nz.size == 0:
        return []
    if nz.size > k:
        # WHY argpartition: O(N) instead of O(N log N) full sort. Take a little
        # extra (k + 50) so ties at the boundary are resolved deterministically.
        part = np.argpartition(-scores[nz], min(k + 50, nz.size - 1))[: k + 50]
        nz = nz[part]
    ranked = sorted(((ids[i], float(scores[i])) for i in nz), key=lambda x: (-x[1], x[0]))
    return ranked[:k]
