# 03 · Vector indexes: flat vs approximate

**Code:** `src/retrieval/dense.py::DenseIndex` · **Toy:** 06 · **Related:** [15 scaling](15_scaling_to_100M.md)

## 1. Intuition

Given a query vector, find the k stored vectors with the highest dot product.
- **Flat (exact):** compare against every vector. Always correct, cost grows linearly with N.
- **Approximate (ANN):** build a structure (graph, clusters, compressed codes) that skips most
  vectors. Much faster at large N, but it sometimes misses a true neighbour (recall < 1).

Analogy: finding the tallest person in a stadium. Flat: measure everyone. HNSW: ask
section captains, who point you to their tallest friends, and so on. Usually right, much faster.

## 2. The mechanics

| index | query cost | memory / vector | recall | updates |
|---|---|---|---|---|
| Flat (IndexFlatIP) | O(N·d) | 4d bytes (1,536 B) | 1.0 | add O(1), remove O(N) |
| HNSW (M = 32) | ~O(log N) | 4d + ~2M·4 B ≈ 1.8 KB | 0.95–0.99 tunable (efSearch) | add OK, delete is hard |
| IVF-Flat (nlist) | O(N/nlist · nprobe) | 4d | depends on nprobe | add OK, centroids drift |
| IVF-PQ (m bytes) | as IVF, on compressed codes | m bytes (e.g. 48) | 0.8–0.95 | add OK |

## 3. Diagram

```
Flat:  q · [v1 v2 v3 ... v19,206]  -> one BLAS matrix-vector product per query (~1 ms per shard)
HNSW:  top layer (few nodes) -> greedy hop -> ... -> bottom layer (all nodes) -> best-first search
IVF:   q -> nearest nprobe of nlist centroids -> scan only those lists
```

## 4. How this repo implements it

`faiss.IndexIDMap2(faiss.IndexFlatIP(384))`:
- **Flat**, because 19K vectors per shard × 384 dims takes ~1 ms per query. That is negligible next to the
  ~15 ms query encoder, and the assignment allows exact search at this size.
- **IDMap2** lets us use our own int64 ids (mapped from string doc ids) and supports `remove_ids`,
  which live deletes and upserts need.
- Deterministic tie-break `(-score, doc_id)` so merged shard results equal single-node results.

## 5. Run the toy

```bash
python examples/06_faiss_flat_vs_hnsw.py   # recall vs latency for HNSW settings on 20K random vectors
```

## 6. Observe it in the real system

Dense single-node p50 is 18.8 ms (`results/bench.json`), of which the faiss search is about 1 ms. Switching
to HNSW would save under 1 ms (~5%) and cost ~1–5% recall. **Not worth it at this scale.** At 100M
vectors the picture flips: see [15](15_scaling_to_100M.md).

## 7. Pitfalls and interview questions

- HNSW **deletes** are usually "mark deleted". The graph degrades and needs periodic rebuilds.
- IVF needs **training** (k-means) on representative data. After a model change, retrain.
- `IndexFlatIP` on **unnormalised** vectors is not cosine.
- *Q: When would you switch from flat to ANN?* When the flat search time is no longer negligible next to
  the encoder, or when memory forces compression. Roughly beyond ~1–5M vectors per node on CPU.

## 8. Exercises

1. (Concept) Why does flat search parallelise perfectly across shards?
2. (Predict) Doubling N doubles flat latency. What does it do to HNSW latency?
3. (Do) In toy 06, find the smallest `efSearch` that gives recall ≥ 0.99.
4. (Do) Compute memory for 100M × 384 float32 vectors. Then for PQ with 48 bytes/vector.

Solutions: [solutions/03_vector_indexes.md](solutions/03_vector_indexes.md)

## 9. Further reading

- Malkov & Yashunin, *HNSW* (2016) · Jégou et al., *Product Quantization* (2011)
- Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs* (faiss, 2017)
- faiss wiki: "Guidelines to choose an index"
