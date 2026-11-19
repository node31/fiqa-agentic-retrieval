"""Toy 04: sentence embeddings + cosine similarity. Matching MEANING instead of words.

Concept: a bi-encoder maps text to a unit vector, and similarity = dot product = cosine.
Real code: src/retrieval/dense.py (Embedder, DenseIndex)    Doc: docs/02_dense_retrieval.md
"""
import numpy as np
from _common import TOY_DOCS, banner, load_st_model, note, step, try_this

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.dense import HashEmbedder

banner("Toy 04: embeddings find synonyms that BM25 cannot")
model = load_st_model("sentence-transformers/all-MiniLM-L6-v2")
enc = (lambda xs: model.encode(xs, normalize_embeddings=True)) if model else HashEmbedder().encode_queries
ids, texts = list(TOY_DOCS), list(TOY_DOCS.values())
D = np.asarray(enc(texts))
step(1, f"Each passage -> a vector of {D.shape[1]} numbers. Norms: {np.linalg.norm(D, axis=1)[:3].round(3)}...")
note("normalize_embeddings=True makes every vector length 1, so dot product = cosine of the angle.")
query = "gambling on equities dropping"  # paraphrases the shorting passages d1/d2/d10 with zero shared words
shorting = {"d1", "d2", "d10"}
shared = sorted(set(tokenize(query)) & {t for x in texts for t in tokenize(x)})
step(2, f"Query {query!r} -> stems {tokenize(query)}. Stems shared with ANY passage: {shared or 'none'}")
bm = BM25Index().build(ids, texts).search(tokenize(query), 3)
print(f"     BM25 top-3: {[(d, round(s, 2)) for d, s in bm] or 'nothing matched'}")
q = np.asarray(enc([query]))[0]
sims = D @ q
step(3, "Dense: cosine(query, each doc) = dot product of unit vectors")
top3 = [ids[i] for i in np.argsort(-sims)[:3]]
for i in np.argsort(-sims)[:4]:
    print(f"     {ids[i]:4s} cos={sims[i]:.3f}  {texts[i][:62]}")
if model:
    note(f"{len(shorting & set(top3))} of the 3 shorting passages are in the dense top-3 with no shared word: "
         "'gambling' ~ 'betting', 'dropping' ~ 'fall', 'equities' ~ 'stock'. The model learned these from data.")
else:
    note("Fallback HashEmbedder has no semantics, so install/cache MiniLM to see the real effect.")
step(4, "Geometry: cosine near 1 = same direction (similar meaning), near 0 = unrelated")
a, b = np.asarray(enc(["pay off my mortgage", "repay the home loan early"]))
c = np.asarray(enc(["roth ira withdrawal rules"]))[0]
print(f"     cos(mortgage, home loan) = {a @ b:.3f}    cos(mortgage, roth ira) = {a @ c:.3f}")
try_this("Find a query where BM25 beats dense (hint: exact codes like '1099-B' or rare names).",
         "Remove normalize_embeddings=True: why does NOTHING change? (hint: print(model); MiniLM ends in a "
         "Normalize layer.) Drop that layer and rank by raw dot product: which docs gain?",
         "Embed 'not a good investment' and 'a good investment'. How close are they? Why?")
