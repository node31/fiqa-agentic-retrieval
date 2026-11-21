"""Toy 06: exact (flat) vs approximate (HNSW) vector search. Recall vs speed vs memory.

Concept: brute force is exact and O(N); HNSW is a navigable graph, ~O(log N) but approximate.
Real code: src/retrieval/dense.py::DenseIndex (flat)    Doc: docs/topics/03_vector_indexes.md
"""
import json
import time

import faiss
import numpy as np
from _common import ROOT, banner, note, step, try_this

banner("Toy 06: IndexFlatIP vs IndexHNSWFlat on 20,000 random 384-d unit vectors")
rng = np.random.default_rng(0)
N, d, nq, k = 20_000, 384, 200, 10
X = rng.standard_normal((N, d)).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)
src_ids = rng.choice(N, nq)
# Noise of total length ~0.3 (0.3/sqrt(d) per dimension), so each query really is NEAR its source point.
Q = X[src_ids] + (0.3 / np.sqrt(d)) * rng.standard_normal((nq, d)).astype("float32")
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
faiss.omp_set_num_threads(1)

flat = faiss.IndexFlatIP(d)
flat.add(X)
t0 = time.perf_counter()
S, truth = flat.search(Q, k)
t_flat = (time.perf_counter() - t0) / nq * 1e3
step(1, f"Flat (exact): {t_flat:.2f} ms/query, recall@10 = 1.000 by definition, memory = N*d*4 = "
        f"{N * d * 4 / 2**20:.1f} MB")
print(f"     mean cosine of the exact neighbours: #1 {S[:, 0].mean():.3f} (the query's source point), "
      f"#2 {S[:, 1].mean():.3f}, #10 {S[:, 9].mean():.3f}")

for M, ef in [(16, 16), (16, 64), (32, 128)]:
    h = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    h.add(X)
    build = time.perf_counter() - t0
    h.hnsw.efSearch = ef
    t0 = time.perf_counter()
    _, got = h.search(Q, k)
    t_h = (time.perf_counter() - t0) / nq * 1e3
    rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(truth, got)])
    rec1 = np.mean(got[:, 0] == truth[:, 0])
    step(2, f"HNSW M={M} efSearch={ef}: {t_h:.3f} ms/query ({t_flat / t_h:.1f}x faster than flat), "
            f"recall@1 = {rec1:.3f}, recall@10 vs exact = {rec:.3f}, build {build:.1f}s")
note(f"recall@1 is easy (the source point stands out at cosine {S[:, 0].mean():.2f}), recall@10 is hard: the true "
     f"#2..#10 sit at ~{S[:, 1:].mean():.2f}, barely above thousands of other random points. Random vectors are the "
     "WORST case for HNSW (no cluster structure, so no useful neighbourhoods). Real embeddings do better.")
mb = ROOT / "results" / "microbench.json"
enc_ms = json.loads(mb.read_text())["warm_encode"]["p50_ms"] if mb.exists() else None
note(f"At FiQA scale (19K vectors per shard) flat search costs ~{t_flat:.1f} ms here, "
     + (f"well below the {enc_ms:.0f} ms warm query encode in results/microbench.json, " if enc_ms else
        "well below the cost of encoding the query, ")
     + "so an ANN index would save little and cost recall. At 100M vectors flat is impossible: "
     "see docs/topics/15_scaling_to_100M.md for IVF-PQ arithmetic.")
try_this("Increase N to 200,000. How do flat and HNSW query times scale? (linear vs ~log)",
         "Raise efSearch until recall >= 0.99. What is the speedup over flat at that point?",
         "Add faiss.IndexIVFFlat with nlist=256, nprobe in {1, 8, 32}: plot recall vs time.")
