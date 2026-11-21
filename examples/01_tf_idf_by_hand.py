"""Toy 01: TF and IDF by hand. Why rare words carry more evidence than common ones.

Concept: term frequency (tf), document frequency (df), inverse document frequency (idf).
Real code: src/retrieval/bm25.py::idf      Doc: docs/topics/01_lexical_retrieval_bm25.md section 2
"""
import math
from collections import Counter

from _common import TOY_DOCS, banner, note, step, try_this

from src.retrieval.bm25 import idf, tokenize

banner("Toy 01: TF-IDF by hand on a 10-document corpus")
docs = {d: tokenize(t) for d, t in TOY_DOCS.items()}
N = len(docs)
df = Counter(t for toks in docs.values() for t in set(toks))
step(1, f"N = {N} documents. Document frequency df(t) = number of docs containing t.")
for t in ["tax", "stock", "mortgag", "unlimit"]:
    print(f"     df({t!r:10s}) = {df[t]}")
step(2, "IDF turns df into a weight. Lucene BM25 IDF = ln(1 + (N - df + 0.5) / (df + 0.5))")
for t in ["tax", "stock", "mortgag", "unlimit"]:
    w = idf(N, df[t])
    print(f"     idf({t!r:10s}) = ln(1 + ({N} - {df[t]} + 0.5)/({df[t]} + 0.5)) = {w:.3f}")
note(f"'unlimit' (in {df['unlimit']} doc) weighs {idf(N, df['unlimit']) / idf(N, df['tax']):.1f}x more than "
     f"'tax' (in {df['tax']} docs). Matching a rare word is stronger evidence that the document is about your query.")
step(3, "TF-IDF score of the query 'short stock borrow' for each doc = sum over terms of tf * idf")
query = tokenize("short stock borrow")
rows = []
for d, toks in docs.items():
    tf = Counter(toks)
    s = sum(tf[t] * idf(N, df[t]) for t in query if df[t])
    rows.append((s, d))
for s, d in sorted(rows, reverse=True)[:5]:
    print(f"     {d:4s} score={s:.3f}  {TOY_DOCS[d][:60]}...")
note("Plain TF-IDF has two flaws BM25 fixes: tf grows without limit (10 mentions = 10x score), and "
     "long documents win just by containing more words. See toy 02.")
print(f"\n  sanity: ln(2) = {math.log(2):.3f}. A term in EVERY doc gets idf = ln(1 + 0.5/(N+0.5)) ~ 0, "
      f"here {idf(N, N):.3f}")
try_this("Add a document that repeats 'short' 20 times. Where does it rank with TF-IDF?",
         "Compute the classic idf = log(N/df) for a term in all docs. Why does Lucene add 1 inside the log?",
         "Which query words in FiQA would you expect to have the highest idf? Check with BM25Index.df.")
