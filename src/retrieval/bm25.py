"""BM25 lexical retrieval, written from scratch, with live add/delete.

WHAT
    An inverted index plus the Okapi BM25 scoring function. Given a query, it
    returns the top-k passages ranked by how well their *words* match.

WHY FROM SCRATCH (see docs/topics/01_lexical_retrieval_bm25.md)
    * rank_bm25 has no deletes and scores every document for every query
      (O(N) per query term), which is too slow for a 50 ms budget.
    * Pyserini needs Java.
    * Part B needs two things libraries hide: (1) **global corpus statistics**
      injected at query time (sharding), and (2) **live upserts/deletes**.

THE FORMULA (docs/topics/01_lexical_retrieval_bm25.md works an example by hand)
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
    delta into the base ("compaction") is not implemented; the delta stays small at this scale.

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

HOW TO READ THIS FILE (top to bottom)
    1. tokenize()     text -> list of normalised words ("terms"). Used for docs AND queries.
    2. CorpusStats    the corpus-wide numbers (N, total length, df per term) that the
                      formula needs; small, JSON-able and summable across shards.
    3. idf()          the "how rare is this term" weight.
    4. BM25Index      build() bulk-loads docs, add()/delete() change them live,
                      local_stats() reports stats to the router, search() scores.
    5. _top_k()       picks the best k scores out of one score per document.

    Words used below:
    * term          : one token after tokenize(), e.g. "invest".
    * posting       : one (doc, tf) pair saying "this doc contains this term tf times".
    * posting list  : all postings of one term. An *inverted index* is the map
                      term -> posting list (inverted because a normal document maps
                      doc -> words). It lets a query touch only docs that share a term.
    * row / column  : every doc gets an internal integer row; every term (in the base
                      segment) gets an integer column.
    * tombstone     : a "this row is deleted" marker; the data stays until compaction.
    * CSC / CSR     : two scipy sparse-matrix layouts (Compressed Sparse Column / Row).
                      Both store only non-zero cells. CSC keeps each column's cells
                      together (cheap "which docs contain term j?"), CSR keeps each
                      row's cells together (cheap "which terms are in doc i?").

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
# Tokenisation decides what counts as "the same word". BM25 can only match terms
# that come out of this function identically for the query and the document.

# A token is a maximal run of lowercase letters/digits; everything else (spaces,
# punctuation, "$", "-") is a separator. WHY lowercase-only: tokenize() lowercases
# the text first, so "Stock" and "stock" become one term. PITFALL: "401(k)" becomes
# ["401", "k"] and "e-mail" becomes ["e", "mail"]; that is accepted, not a bug.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# WHY this list: Lucene's classic English stop set (33 words), so our BM25 is
# comparable to the published Pyserini/Anserini BEIR baselines. Note it DOES drop
# "no"/"not". Negation is lost for lexical matching, a known BM25 weakness that
# dense models partly fix (exercise in docs/topics/01_lexical_retrieval_bm25.md).
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split()
)

# Stemming chops words down to a shared root ("investing", "invested", "invests"
# -> "invest") so different forms of a word match each other. Snowball "english"
# is the Porter2 algorithm. WHY optional: if the package is missing we fall back to
# no stemming; search still works, just with fewer matches (lower recall).
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
    # Step by step for the doctest above:
    #   lower + regex -> ['investing', 'in', 'the', 'stocks', 'not', 'bonds']
    #   drop stopwords -> ['investing', 'stocks', 'bonds']   ("in", "the", "not" removed)
    #   stem           -> ['invest', 'stock', 'bond']
    # Duplicates are kept on purpose: build()/add() count them to get tf.
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# --- Corpus statistics (the sharding-relevant part) -----------------------------


@dataclass
class CorpusStats:
    """The numbers IDF and length-normalisation need. Summable across shards.

    >>> a = CorpusStats(10, 100.0, {"loan": 2}); b = CorpusStats(5, 50.0, {"loan": 1})
    >>> a.merge(b).df["loan"], a.merge(b).avgdl
    (3, 10.0)
    """

    # n_docs    = N, the number of LIVE (not deleted) documents.
    # total_len = sum of all live doc lengths in tokens; avgdl = total_len / n_docs.
    # df        = document frequency: term -> how many live docs contain it at least
    #             once (a doc with "loan" 5 times still adds only 1 to df["loan"]).
    #             Usually holds only the query's terms, not the whole vocabulary.
    # WHY store total_len instead of avgdl: sums add up across shards, averages do
    # not. In the doctest: (100 + 50) / (10 + 5) = 10.0, which is the true average.
    n_docs: int
    total_len: float
    df: dict[str, int] = field(default_factory=dict)

    @property
    def avgdl(self) -> float:
        # avgdl = average document length. WHY 1.0 when empty: avoids dividing by
        # zero; with no docs there is nothing to score anyway.
        return self.total_len / self.n_docs if self.n_docs else 1.0

    def merge(self, other: "CorpusStats") -> "CorpusStats":
        # Shards hold disjoint sets of docs, so every statistic is a plain sum.
        # Returns a NEW object; neither input is modified.
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
    # Reading the doctest: a term in 1 of 1000 docs gets ln(1 + 999.5/1.5) ≈ 6.5,
    # a term in half the docs gets ln(1 + 500.5/500.5) = ln 2 ≈ 0.69.
    # WHY the "+0.5": smoothing so df = 0 or df = N never divides by zero.
    # WHY the "1 +" inside ln (Lucene's change to classic BM25): the classic
    # ln((N-df+0.5)/(df+0.5)) goes NEGATIVE for terms in more than half the docs,
    # which would make matching a common word lower a doc's score. Here IDF > 0 always.
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


# --- The index -------------------------------------------------------------------


class BM25Index:
    """Inverted index + BM25 scorer supporting bulk build, add, delete and global stats."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        # k1 controls tf saturation (how fast extra repeats stop helping); b controls
        # how strongly long docs are penalised (0 = not at all, 1 = fully).
        # Internal rows only grow: a deleted or replaced doc keeps its row (as a
        # tombstone) and a re-added doc gets a NEW row at the end. So the arrays
        # doc_ids / doc_len / alive all have one entry per row ever created, while
        # row_of / df / total_len describe live docs only.
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}  # term -> column in the base matrix
        self.terms: list[str] = []  # column -> term (inverse of vocab)
        self.doc_ids: list[str] = []  # internal row -> external doc id
        self.row_of: dict[str, int] = {}  # external doc id -> LIVE internal row
        self.doc_len = np.zeros(0, dtype=np.float32)  # shape (n_rows,): tokens per row
        self.alive = np.zeros(0, dtype=bool)  # shape (n_rows,): False = tombstoned row
        self.df: Counter[str] = Counter()  # document frequency over LIVE docs only
        self.total_len = 0.0
        self._base_csc: sparse.csc_matrix | None = None  # fast column (term) access
        self._base_csr: sparse.csr_matrix | None = None  # fast row (doc) access, for deletes
        self._n_base = 0  # rows [0, _n_base) live in the base matrices; later rows are delta
        # Delta segment = postings of docs added after build(). WHY a dict of lists:
        # appending to a scipy sparse matrix means rebuilding it; a list append is O(1).
        self._delta: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(row, tf)]
        # row -> Counter(term -> tf) for delta docs, so delete() knows which df
        # entries to decrement (the base segment answers that from its CSR copy).
        self._delta_terms: dict[int, Counter] = {}

    # ---- building ----------------------------------------------------------
    def build(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        """Bulk-index documents into the base segment. Call once, on an empty index."""
        assert not self.doc_ids, "build() is for an empty index; use add() afterwards"
        # Collect the matrix as "COO" triplets (row, col, value) = (doc row, term
        # column, tf): one triplet per distinct term per doc. WHY triplets: it is
        # the cheapest way to assemble a sparse matrix; we convert it once at the end.
        rows, cols, vals = [], [], []
        lens = np.zeros(len(doc_ids), dtype=np.float32)
        for r, text in enumerate(texts):
            # counts = term frequency (tf) of every term in this doc, e.g.
            # Counter({'loan': 2, 'rate': 1}). Its total is the doc length |d|.
            counts = Counter(tokenize(text))
            lens[r] = sum(counts.values())
            for term, tf in counts.items():
                # setdefault hands a new term the next free column (0, 1, 2, ...)
                # and returns the existing column for a term seen before.
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(r)
                cols.append(j)
                vals.append(tf)
                self.df[term] += 1  # once per doc, because counts has distinct terms
        # Invert vocab (term -> column) into terms (column -> term).
        self.terms = [None] * len(self.vocab)  # type: ignore[list-item]
        for t, j in self.vocab.items():
            self.terms[j] = t
        # postings matrix: rows = docs, cols = terms, cell = tf (0 cells not stored).
        # For FiQA that is 57,638 rows x ~100K columns, yet almost every cell is 0.
        coo = sparse.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(doc_ids), len(self.vocab))
        )
        # WHY keep two copies: search() needs columns (CSC), delete() needs rows (CSR).
        # Worked example, 3 docs, columns 0="loan", 1="rate":
        #   doc0 = loan x2, doc1 = rate x1, doc2 = loan x1 + rate x3
        #   CSC: data=[2,1,1,3]  indices(rows)=[0,2,1,2]  indptr=[0,2,4]
        #   -> column j's posting list is indices[indptr[j]:indptr[j+1]] with tfs at the
        #      same positions in data: "loan" = rows [0,2] with tf [2,1].
        #   CSR is the same idea per row: indptr=[0,1,2,4], indices(cols)=[0,1,0,1].
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
        # WHY delete-then-append instead of editing in place: the old postings may sit
        # in the immutable base matrix. Tombstoning the old row and appending a fresh
        # row keeps both segments append-only (the LSM idea).
        if doc_id in self.row_of:
            self.delete(doc_id)
        counts = Counter(tokenize(text))
        row = len(self.doc_ids)  # the new doc takes the next row number
        self.doc_ids.append(doc_id)
        self.row_of[doc_id] = row
        # WHY np.append: O(N) copy per write, ~20 µs at this size; fine for a demo,
        # a production index would use chunked arrays.
        self.doc_len = np.append(self.doc_len, np.float32(sum(counts.values())))
        self.alive = np.append(self.alive, True)
        self.total_len += float(self.doc_len[row])
        # New docs go only to the delta segment, even if the term already has a base
        # column (and brand-new terms never get a column). search() reads both.
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
        # The postings stay, but the STATISTICS must forget the doc right away, so
        # N (len(row_of)), total_len and df describe live docs only. Otherwise IDF
        # and avgdl would drift as deletes pile up.
        self.total_len -= float(self.doc_len[row])
        for term in self._terms_of_row(row):
            self.df[term] -= 1
            if self.df[term] <= 0:
                del self.df[term]
        return True

    def _terms_of_row(self, row: int) -> Iterable[str]:
        # Distinct terms of one row. Base rows: slice the CSR row (indices holds the
        # term columns of that row, see the example in build()). Delta rows: pop the
        # saved Counter. WHY pop: this is only called by delete(), and after the
        # delete that Counter is never needed again, so we free its memory.
        if row < self._n_base:
            csr = self._base_csr
            return (self.terms[j] for j in csr.indices[csr.indptr[row]: csr.indptr[row + 1]])
        return self._delta_terms.pop(row, Counter()).keys()

    # ---- statistics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.row_of)

    def local_stats(self, terms: Iterable[str], row_mask: np.ndarray | None = None) -> CorpusStats:
        """This index's N, total length and df for just the requested terms.

        ``row_mask`` (one bool per internal row) restricts the statistics to a subset of docs, e.g. one
        tenant's: N, total length and df then count only those docs. WHY a tenant needs its own
        statistics: df summed over everyone would let one tenant's documents change another tenant's
        scores, and a careful attacker could probe how common a term is in someone else's documents.
        """
        # WHY only the requested terms: the router calls this once per query on
        # every shard; sending df for the whole ~100K-term vocabulary would be waste.
        if row_mask is None:
            return CorpusStats(len(self.row_of), self.total_len, {t: self.df.get(t, 0) for t in terms})
        live = self.alive & row_mask
        df = {}
        for t in terms:
            n = 0
            j = self.vocab.get(t)
            if self._base_csc is not None and j is not None:
                csc = self._base_csc
                n += int(live[csc.indices[csc.indptr[j]:csc.indptr[j + 1]]].sum())
            n += sum(1 for row, _ in self._delta.get(t, ()) if live[row])
            df[t] = n
        return CorpusStats(int(live.sum()), float(self.doc_len[live].sum()), df)

    # ---- search ----------------------------------------------------------------
    def search(self, terms: Sequence[str], k: int = 10,
               stats: CorpusStats | None = None, row_mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Score docs for pre-tokenised, de-duplicated ``terms`` and return the top-k.

        Args:
            terms: output of ``tokenize(query)``, ideally de-duplicated.
            k: number of results.
            stats: GLOBAL statistics from the router. ``None`` = use local stats
                (correct only for a single-node index).
            row_mask: one bool per internal row; rows that are False never score (a tenant filter).
                Applied BEFORE the top-k, so the caller still gets k results from its own docs.

        Complexity: O(Σ_t df(t)) to accumulate + O(N) to pick the top-k.
        """
        if not self.doc_ids:
            return []
        # WHY global stats matter: IDF and avgdl must come from the WHOLE corpus
        # (all shards). Only tf and |d| are truly per-doc, and those are local.
        # stats=None means "single node": fall back to this index's own numbers.
        stats = stats or self.local_stats(terms, row_mask)
        k1, b = self.k1, self.b
        # Length normalisation for EVERY doc, computed once per query (vectorised).
        # norm[i] = k1·(1 - b + b·|d_i|/avgdl): the denominator part that does not
        # depend on the term. Shape (n_rows,). A doc of exactly average length gets
        # norm = k1; a doc twice as long gets 1.2·(0.25 + 1.5) = 2.1 (bigger = lower score).
        # max(..., 1e-9) only guards against dividing by a zero avgdl.
        norm = k1 * (1.0 - b + b * self.doc_len / max(stats.avgdl, 1e-9))
        # scores: one float per internal row, shape (n_rows,). We ADD each query
        # term's contribution into it ("score accumulation"), touching only the
        # rows in that term's posting list instead of every document.
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        csc = self._base_csc
        # PITFALL: a term listed twice in `terms` is scored twice (its weight doubles).
        # Callers should pass de-duplicated terms, e.g. list(dict.fromkeys(tokenize(q))).
        for t in terms:
            df = stats.df.get(t, 0)
            if df <= 0:
                continue  # WHY: term absent everywhere -> contributes nothing
            w = idf(stats.n_docs, df)
            j = self.vocab.get(t)
            # Base segment: slice column j of the CSC matrix (its posting list).
            # rows and tf are equal-length arrays, so the update below is one
            # vectorised numpy operation over all docs containing t.
            # tf saturation, for an average-length doc (norm = 1.2):
            #   tf=1 -> 1·2.2/2.2 = 1.0,  tf=2 -> 4.4/3.2 = 1.375,  tf=10 -> 22/11.2 ≈ 1.96
            # i.e. repeats help less and less, and the factor never exceeds k1+1 = 2.2.
            if csc is not None and j is not None:
                lo, hi = csc.indptr[j], csc.indptr[j + 1]
                rows, tf = csc.indices[lo:hi], csc.data[lo:hi]
                # Note: `scores[rows] += ...` would silently apply only ONE update per row if
                # rows had duplicates, but a CSC column lists each row at most once.
                scores[rows] += w * tf * (k1 + 1) / (tf + norm[rows])
            # Delta segment: same formula, one posting at a time (it is small).
            for row, tf in self._delta.get(t, ()):
                scores[row] += w * tf * (k1 + 1) / (tf + norm[row])
        # Apply tombstones: deleted rows may have collected a score from their
        # leftover postings; zeroing them makes _top_k skip them (it keeps only > 0).
        scores[~self.alive] = 0.0
        if row_mask is not None:
            scores[~row_mask] = 0.0  # other tenants' docs: filtered BEFORE the top-k
        return _top_k(scores, self.doc_ids, k)


def _top_k(scores: np.ndarray, ids: Sequence[str], k: int) -> list[tuple[str, float]]:
    """Top-k positive scores, ties broken by doc id so results are deterministic."""
    # nz = row numbers of docs that matched at least one query term (score > 0).
    # Typically a small fraction of the corpus, so everything below is cheaper.
    nz = np.flatnonzero(scores > 0)
    if nz.size == 0:
        return []
    if nz.size > k:
        # WHY argpartition: O(N) instead of O(N log N) full sort. Take a little
        # extra (k + 50) so ties at the boundary are resolved deterministically.
        # argpartition(x, m) reorders positions so the m-th one holds the value a full
        # sort would put there, smaller-or-equal values before it, larger-or-equal after,
        # each side in no particular order. We pass -scores so "smaller" means "better": the
        # first k + 50 positions are the k + 50 best docs (unsorted). The result
        # holds positions into scores[nz], so nz[part] maps them back to rows.
        part = np.argpartition(-scores[nz], min(k + 50, nz.size - 1))[: k + 50]
        nz = nz[part]
    # Fully sort the small survivor set: highest score first, then doc id A-Z for ties.
    ranked = sorted(((ids[i], float(scores[i])) for i in nz), key=lambda x: (-x[1], x[0]))
    return ranked[:k]
