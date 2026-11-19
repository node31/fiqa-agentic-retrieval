"""ch03: BM25 from scratch, written by you. Standard library + `snowballstemmer` only.

Run the checker:   python -m pytest exercises/ch03_bm25
Run the demo:      python exercises/ch03_bm25/my_bm25.py
"""
from __future__ import annotations

import math  # noqa: F401
import re  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401

import snowballstemmer

K1, B = 1.2, 0.75
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split())
_STEMMER = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """lowercase → regex [a-z0-9]+ → drop STOPWORDS → stem each word with _STEMMER.stemWord(word).

    Example: tokenize("What is short selling?") == ["what", "short", "sell"]
    """
    raise NotImplementedError("TODO (ch03): implement tokenize")


class Index:
    """Your inverted index. Store whatever you need, at least:
       postings: {term: [(doc_id, tf), ...]}    doc_len: {doc_id: number of tokens}
       n_docs, avgdl
    """

    def __init__(self):
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_len: dict[str, int] = {}
        self.n_docs = 0
        self.avgdl = 0.0


def build_index(docs: dict[str, str]) -> Index:
    """Tokenise every doc and fill an Index (postings with term frequencies, lengths, N, avgdl)."""
    raise NotImplementedError("TODO (ch03): implement build_index")


def idf(n_docs: int, df: int) -> float:
    """Lucene BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5))."""
    raise NotImplementedError("TODO (ch03): implement idf")


def search(index: Index, query: str, k: int = 10) -> list[tuple[str, float]]:
    """BM25 top-k as [(doc_id, score)], highest first, ties by doc_id; only score > 0.

    For each DISTINCT query term t with df > 0, for each (doc, tf) in postings[t]:
        score[doc] += idf(t) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len[doc] / avgdl))
    """
    raise NotImplementedError("TODO (ch03): implement search")


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import dataset_from_argv, use_chapter

    use_chapter("ch01_data")
    from loader import load_jsonl, load_split  # your ch01 code

    name, data_dir = dataset_from_argv()            # add --full for all 57,638 docs (~1 min to index)
    t0 = time.perf_counter()
    idx = build_index(load_jsonl(data_dir / "corpus.jsonl"))
    print(f"[{name}] indexed {idx.n_docs:,} docs, {len(idx.postings):,} terms, avgdl {idx.avgdl:.1f} "
          f"in {time.perf_counter() - t0:.1f}s")
    queries, qrels = load_split(data_dir, "dev")
    for qid in list(queries)[:4]:
        print(f"{queries[qid]!r} (gold {sorted(qrels[qid])}) ->", [(d, round(s, 3)) for d, s in search(idx, queries[qid], 5)])
    if name == "full":
        print("'What is short selling?' ->", [(d, round(s, 3)) for d, s in search(idx, "What is short selling?", 2)],
              " (guide: 336018 10.68, 127585 10.013)")
