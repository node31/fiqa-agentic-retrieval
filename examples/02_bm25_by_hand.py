"""Toy 02: BM25 by hand. Term-frequency saturation (k1) and length normalisation (b).

Concept: the BM25 formula and its two knobs.
Real code: src/retrieval/bm25.py::BM25Index.search     Doc: docs/01_lexical_retrieval_bm25.md
"""
from _common import banner, note, step, try_this

from src.retrieval.bm25 import BM25Index, idf

banner("Toy 02: BM25 = IDF x saturated, length-normalised TF")
k1, b = 1.2, 0.75
step(1, "The per-term formula:  idf * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl))")
step(2, f"Saturation (k1={k1}): how much does the 2nd, 5th, 20th occurrence add? (|d| = avgdl)")
for tf in [1, 2, 5, 20, 100]:
    part = tf * (k1 + 1) / (tf + k1)
    print(f"     tf={tf:3d} -> tf-part = {part:.3f}   (upper bound k1+1 = {k1 + 1})")
note("Plain TF would give 100 for tf=100. BM25 caps it near 2.2, so keyword stuffing stops paying.")
step(3, f"Length normalisation (b={b}): same tf=1, different document lengths (avgdl = 50)")
for dl in [10, 50, 200, 1000]:
    part = 1 * (k1 + 1) / (1 + k1 * (1 - b + b * dl / 50))
    print(f"     |d|={dl:5d} -> tf-part = {part:.3f}")
note("A match in a 10-word doc is worth ~4x a match in a 1000-word doc: the short doc is more "
     "likely to be ABOUT the word. b=0 turns this off and b=1 applies it fully.")
step(4, "Hand computation vs our real index on a 3-doc corpus")
docs = {"a": "loan loan loan rate", "b": "loan rate stock market investing ideas today", "c": "stock market"}
ix = BM25Index(k1, b).build(list(docs), list(docs.values()))
N, avgdl = 3, ix.total_len / 3
w = idf(N, 2)  # 'loan' appears in a and b
for d, L, tf in [("a", 4, 3), ("b", 7, 1)]:
    hand = w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avgdl))
    real = dict(ix.search(["loan"], 3))[d]
    print(f"     doc {d}: |d|={L} tf={tf} hand={hand:.4f} index={real:.4f}")
note("Same numbers: the index is literally this formula, vectorised over the postings list.")
try_this("Set k1=0. What does BM25 become? (hint: binary 'contains the word or not')",
         "FiQA passages vary from 1 to ~3000 words. Would you raise or lower b? Test on dev!",
         "Write a doc that beats doc 'a' for query 'loan' using only ONE occurrence of 'loan'.")
