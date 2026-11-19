# ch04: Meaning search (embeddings) yourself

Guide: `docs/from_zero/04_meaning_search_embeddings.md`.

> 🎯 **Why:** dense retrieval gave the biggest quality jump in the project (+0.12 nDCG over BM25). Writing cosine
> by hand demystifies "vector search", and running the real model on 4 sentences shows *why* it finds answers that
> share no words with the question.

## Steps
1. `PAPER.md`: cosine by hand.
2. Implement in `my_dense.py`, in order:
   - pure Python, no numpy: `dot`, `norm`, `normalize`, `cosine`
   - with the model: `encode`, `top_k`, `dense_search`
3. `python -m pytest exercises\ch04_dense` (the model checks take ~10 s the first time: loading MiniLM).
4. `python exercises\ch04_dense\my_dense.py`: try your own sentences. Find a pair with **no shared words** but a high cosine,
   and a pair with shared words but a low cosine.

## Hints
- `from sentence_transformers import SentenceTransformer`; load
  `"sentence-transformers/all-MiniLM-L6-v2"` once (a module-level cache), `device="cpu"`.
- `model.encode(texts, normalize_embeddings=True)` returns a numpy array (n, 384) of unit vectors, so cosine = dot product.
- `top_k`: `scores = doc_vecs @ query_vec` (all cosines at once), then the k best, ties by doc id.
- `dense_search`: encode the docs, encode the query, `top_k`. On the full corpus you never re-encode the docs per query:
  encoding 57,638 passages takes ~10 minutes, so it's done once and saved (ch06 does that for you).

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch04_dense` | cosine by hand, `top_k` on hand-made vectors, the real model on 4 sentences | ~10 s |
| full | `python -m pytest exercises\ch04_dense --full` | your `encode` gives the same vectors as the project's index (cos > 0.999), and your `encode` + `top_k` over the 57,638 stored doc vectors give dense nDCG@10 **0.388** on 500 dev queries | ~30 s |
| demo | `python exercises\ch04_dense\my_dense.py [--full]` | the 4-sentence demo; with `--full` real FiQA passages for 2 queries | |
