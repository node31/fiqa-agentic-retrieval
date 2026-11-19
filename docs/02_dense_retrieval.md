# 02 · Dense retrieval: sentence embeddings

**Code:** `src/retrieval/dense.py`, `src/config.py::MODELS`, `src/index/build_index.py` · **Toys:** 04, 05

## 1. Intuition

A neural *bi-encoder* reads a piece of text and outputs a point in 384-dimensional space,
placed so that texts with similar **meaning** land close together. "Betting against equities" and
"shorting a stock" share no words but end up neighbours. Retrieval = embed the query, then find the
nearest passage-points.

Analogy: a map where cities are placed by culture, not geography. Similar-meaning texts are
"nearby" even when they're spelled completely differently.

## 2. The math

- Encoder f: text → ℝ³⁸⁴. We L2-normalise: v = f(x)/‖f(x)‖.
- Similarity = cosine = **dot product of unit vectors**: sim(q,d) = v_q · v_d ∈ [−1, 1].
- Retrieval = top-k of `D @ v_q`, where D is the (N × 384) matrix of passage vectors.

Worked example (toy 04, MiniLM): cos("pay off my mortgage", "repay the home loan early") is high
(≈ 0.7), cos("pay off my mortgage", "roth ira withdrawal rules") is low (≈ 0.1–0.2). Run it to see the exact values.

**How the model learned this:** contrastive training on hundreds of millions of (question, answer)
and paraphrase pairs. It pulls matching pairs together and pushes random pairs apart. Nobody wrote
rules about synonyms.

## 3. Diagram

```
offline (once):  57,638 passages --encoder--> D (57,638 x 384 float32 = 88 MB) --> artifacts/emb_minilm.npy
online (per query): "what is short selling?" --encoder (~15 ms CPU)--> q (384,) --> D @ q --> top-k
```

## 4. How this repo implements it

| model | params | dim | query prefix | dev nDCG@10 | dev R@10 | query p50 (single node) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (**primary**) | 22M, 6 layers | 384 | none | **0.388** | 0.467 | 18.8 ms |
| bge-small-en-v1.5 (migration target) | 33M, 12 layers | 384 | "Represent this sentence for searching relevant passages: " | **0.398** | 0.468 | 30.6 ms |

- `ModelSpec` pins everything that changes a vector: HF name, `max_seq_length = 256` (passages are
  truncated beyond ~190 words; p90 passage = 270 words) and the query prefix.
- `Embedder` = SentenceTransformer + prefix + normalisation. `HashEmbedder` = a dependency-free
  stand-in (feature hashing) so tests and toys run offline in milliseconds.
- `build_index.py` caches corpus vectors with their id order **and a text hash per row**, so a
  shard can reuse a cached vector only if its passage text hasn't changed (used by migration).
- Why MiniLM is primary although bge-small is +0.010 nDCG: bge is 12 layers, so query encoding
  costs 30.6 vs 18.8 ms, which eats most of the 50 ms budget once the router's HTTP fan-out is added.
  Hybrid erases the gap anyway (0.419 MiniLM vs 0.416 bge). See [ADR 0002](adr/0002-dense-models.md).

## 5. Run the toys

```bash
python examples/04_embeddings_cosine.py   # synonyms BM25 can't match; cosine geometry
python examples/05_bge_query_prefix.py    # what the BGE instruction changes
```

## 6. Observe it in the real system

`results/bench.json`: dense-minilm beats BM25 by **+0.124 nDCG@10** (0.388 vs 0.264). The cost is
16× the latency (18.8 vs 1.2 ms), almost all of it in the **query encoder**. The faiss search over
57K vectors is ~1 ms. Published MTEB FiQA test numbers: MiniLM 0.369, bge-small 0.403. Our dev
numbers are in the same range.

## 7. Pitfalls and interview questions

- **Forgetting normalisation**: inner product then favours long vectors, not similar directions.
- **Prefix on the wrong side**: BGE's instruction goes on queries only.
- **Mixing models**: a query vector from model A against doc vectors from model B is *silent garbage*,
  not an error. The migration design ([10](10_model_migration.md)) exists to prevent it.
- **Truncation**: anything beyond `max_seq_length` tokens is invisible to the model. Long FiQA answers
  with the key sentence at the end are a known failure (see `results/failures.md`).
- *Q: Why not fine-tune on FiQA train?* Allowed and probably +5–10 points, but out of scope here. A
  good "what next" answer.

## 8. Exercises

1. (Concept) Why is a bi-encoder fast at query time while a cross-encoder isn't? ([06](06_reranking.md))
2. (Predict) Will dense retrieval rank "1099-B" form questions well? Why or why not?
3. (Do) Measure query encode time with `torch.set_num_threads(1)` vs 4 vs 16. Explain the curve.
4. (Do) Rebuild the MiniLM cache with `max_seq_length=128`. What happens to nDCG and to index build time?

Solutions: [solutions/02_dense.md](solutions/02_dense.md)

## 9. Further reading

- Reimers & Gurevych, *Sentence-BERT* (2019)
- Karpukhin et al., *Dense Passage Retrieval* (2020)
- Xiao et al., *C-Pack: BGE embeddings* (2023)
- MTEB leaderboard (FiQA2018 task)
