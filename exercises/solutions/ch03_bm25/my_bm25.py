"""SOLUTION for ch03: BM25 from scratch. Standard library + snowballstemmer.

Try the exercise yourself first (exercises/ch03_bm25/my_bm25.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch03_bm25
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import snowballstemmer

K1, B = 1.2, 0.75     # WHY these values: the Lucene/Elasticsearch defaults; BEIR's BM25 baseline uses them
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split())
_STEMMER = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[a-z0-9]+")   # WHY compile once: tokenize runs ~57K times while indexing


def tokenize(text: str) -> list[str]:
    """lowercase → words → drop stopwords → stem.  "What is short selling?" → ["what", "short", "sell"]"""
    words = _WORD.findall(text.lower())                 # "what's" → ["what", "s"]: punctuation splits words
    # PITFALL: drop stopwords BEFORE stemming. The list holds surface forms ("this"), and the stemmer
    # would turn "this" into "thi", which is no longer in the list.
    return [_STEMMER.stemWord(w) for w in words if w not in STOPWORDS]


class Index:
    """postings: {term: [(doc_id, tf), ...]}, doc_len: {doc_id: n tokens}, n_docs, avgdl."""

    def __init__(self):
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_len: dict[str, int] = {}
        self.n_docs = 0
        self.avgdl = 0.0


def build_index(docs: dict[str, str]) -> Index:
    """One pass over the corpus: count terms per doc and append to each term's posting list."""
    idx = Index()
    for doc_id, text in docs.items():
        tf = Counter(tokenize(text))                    # {"sell": 2, "short": 1, ...}
        idx.doc_len[doc_id] = sum(tf.values())          # length = number of tokens AFTER stopword removal
        for term, count in tf.items():
            idx.postings[term].append((doc_id, count))  # WHY inverted: at query time we only touch docs with the term
    idx.n_docs = len(docs)
    idx.avgdl = sum(idx.doc_len.values()) / max(1, idx.n_docs)
    return idx


def idf(n_docs: int, df: int) -> float:
    """Lucene's BM25 idf. The "1 +" keeps it positive even for a term in more than half the docs."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def search(index: Index, query: str, k: int = 10) -> list[tuple[str, float]]:
    """BM25 top-k. Cost: O(sum of posting-list lengths of the query terms), not O(corpus)."""
    scores: dict[str, float] = defaultdict(float)
    # WHY dict.fromkeys: dedupe query terms but keep their order. "sell sell" must not double-count.
    for term in dict.fromkeys(tokenize(query)):
        postings = index.postings.get(term)
        if not postings:                                # unknown term: contributes nothing
            continue
        w = idf(index.n_docs, len(postings))            # df = length of the posting list
        for doc_id, tf in postings:
            # length normalisation: 1.0 for an average-length doc, >1 for longer docs (penalised)
            norm = 1 - B + B * index.doc_len[doc_id] / index.avgdl
            # tf saturation: grows with tf but levels off at (K1 + 1); 10 mentions != 10x the score
            scores[doc_id] += w * tf * (K1 + 1) / (tf + K1 * norm)
    ranked = sorted(((d, s) for d, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))
    return ranked[:k]                                   # WHY (-score, id): deterministic ties → reproducible evals


if __name__ == "__main__":
    import os
    import sys
    import time
    from pathlib import Path

    os.environ.setdefault("EXERCISES_SOLUTIONS", "1")   # use the ch01 SOLUTION, not your stub
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common import dataset_from_argv, use_chapter

    use_chapter("ch01_data")
    from loader import load_jsonl, load_split  # ch01 solution

    if "--full" not in sys.argv:   # the worksheet's 3-doc example first
        three = {"d1": "short selling means borrowing shares",
                 "d2": "a short story about selling a house and selling a car",
                 "d3": "mortgage rates rose"}
        idx = build_index(three)
        print("tokens d2:", tokenize(three["d2"]))          # ['short', 'stori', 'about', 'sell', 'hous', 'sell', 'car']
        print("lengths  :", idx.doc_len, "avgdl", idx.avgdl)  # {'d1': 5, 'd2': 7, 'd3': 3} avgdl 5.0
        print("search   :", [(d, round(s, 4)) for d, s in search(idx, "short selling")])  # d2 0.9848, d1 0.94

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
