"""Micro-benchmark the pieces inside one query, to check the "where does the time go" claims in the docs.

    python scripts/microbench.py            # writes results/microbench.json

Measures (4 torch/faiss threads, like one shard or eval process):
  * process-cold costs: model load, FIRST query encode, second encode (what the startup warmup hides)
  * warm MiniLM query encode
  * faiss IndexFlatIP search over 19.2K vectors (one shard) and 57.6K (whole corpus)
  * BM25 search on the full corpus
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import ARTIFACTS_DIR as ARTIFACTS, RESULTS_DIR as RESULTS  # noqa: E402

THREADS = 4
torch.set_num_threads(THREADS)
faiss.omp_set_num_threads(THREADS)


def timed(fn, n=200):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return {"p50_ms": round(statistics.median(xs), 3), "p95_ms": round(xs[int(0.95 * len(xs)) - 1], 3)}


def main():
    out = {"threads": THREADS}
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out["model_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    q = "Is it better to pay off my mortgage early or invest in index funds?"
    for label in ("first_encode_ms", "second_encode_ms"):
        t0 = time.perf_counter()
        qv = m.encode([q], normalize_embeddings=True)
        out[label] = round((time.perf_counter() - t0) * 1000, 1)
    out["warm_encode"] = timed(lambda: m.encode([q], normalize_embeddings=True))

    emb = np.load(ARTIFACTS / "emb_minilm.npy").astype("float32")
    qv = qv.astype("float32")
    for name, n in (("faiss_flat_19k", 19_206), ("faiss_flat_57k", len(emb))):
        idx = faiss.IndexIDMap2(faiss.IndexFlatIP(emb.shape[1]))
        idx.add_with_ids(emb[:n], np.arange(n, dtype="int64"))
        out[name] = timed(lambda: idx.search(qv, 100))
    out["vector_bytes_per_doc"] = emb.shape[1] * 4

    from src.beir_data import load_corpus
    from src.retrieval.bm25 import BM25Index, tokenize

    corpus = load_corpus()
    bm = BM25Index().build(list(corpus), list(corpus.values()))
    terms = tokenize(q)
    out["bm25_57k"] = timed(lambda: bm.search(terms, 100))
    out["text_bytes_per_passage_mean"] = round(sum(len(t.encode()) for t in corpus.values()) / len(corpus))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "microbench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
