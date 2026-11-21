# 03 · Vector indexes: flat vs approximate

Code: `src/retrieval/dense.py::DenseIndex`. Related: [15 scaling](15_scaling_to_100M.md).

## Flat vs approximate

Given a query vector, the index returns the k stored vectors with the highest dot product. A flat (exact) index
compares against every vector: always correct, with cost growing linearly in N. An approximate (ANN) index builds a
structure (graph, clusters, compressed codes) that skips most vectors. It is much faster at large N but sometimes
misses a true neighbour, so recall drops below 1.

| index | query cost | memory / vector | recall | updates |
|---|---|---|---|---|
| Flat (IndexFlatIP) | O(N·d) | 4d bytes (1,536 B) | 1.0 | add O(1), remove O(N) |
| HNSW (M = 32) | ~O(log N) | 4d + ~2M·4 B ≈ 1.8 KB | 0.95 to 0.99, tunable (efSearch) | add OK, deletes are "mark deleted" and the graph degrades |
| IVF-Flat (nlist) | O(N/nlist · nprobe) | 4d | depends on nprobe | add OK, centroids drift, retrain after a model change |
| IVF-PQ (m bytes) | as IVF, on compressed codes | m bytes (e.g. 48) | 0.8 to 0.95 | add OK |

```
Flat:  q · [v1 v2 v3 ... v19,167]  -> one BLAS matrix-vector product per query (1.4 ms per 19K-vector shard, measured)
HNSW:  top layer (few nodes) -> greedy hop -> ... -> bottom layer (all nodes) -> best-first search
IVF:   q -> nearest nprobe of nlist centroids -> scan only those lists
```

## What I chose

I use `faiss.IndexIDMap2(faiss.IndexFlatIP(384))`:

- Flat, because 19K vectors per shard × 384 dims takes 1.4 ms per query (`results/microbench.json`). That is small
  next to the 8 ms query encoder, and the spec allows exact search at this size. The vectors are normalised, so
  inner product is cosine.
- `IDMap2` lets me use my own int64 ids (mapped from string doc ids) and supports `remove_ids`, which live deletes
  and upserts need.
- A deterministic tie-break `(-score, doc_id)` makes merged shard results equal single-node results.

## Where the time goes

Dense single-node p50 is 18.8 ms (`results/bench.json`). `scripts/microbench.py` splits it: query encode 8.2 ms,
faiss flat over all 57.6K vectors 5.3 ms (1.4 ms over one 19.2K shard), and about 5 ms of Python glue (id mapping,
result lists, the harness). 57K costs 3.8× a 19K shard instead of 3× because 88 MB of vectors no longer fits in
cache, so the scan becomes memory-bandwidth bound (about 17 GB/s).

In the 3-shard service, I estimate HNSW would save about 1 ms of about 29 ms and cost 1 to 5% recall, so I kept flat.
I would switch to ANN once flat search is no longer small next to the encoder, or when memory forces compression,
roughly beyond 1 to 5M vectors per node on CPU. At 100M vectors the picture flips: see [15](15_scaling_to_100M.md).

```bash
python scripts/microbench.py
```

References: Malkov & Yashunin, *HNSW* (2016); Jégou et al., *Product Quantization* (2011).
