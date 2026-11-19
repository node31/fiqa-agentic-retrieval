"""Toy 11: measuring memory. RSS vs tracemalloc vs back-of-envelope arithmetic.

Concept: where a shard's RAM goes (model weights vs vectors vs postings vs Python overhead).
Real code: src/util.py::rss_mb       Doc: docs/07_benchmarking_latency_memory.md
"""
import gc
import tracemalloc

import numpy as np
from _common import banner, note, step, try_this

from src.util import rss_mb

banner("Toy 11: three ways to measure memory, and why they disagree")
step(1, f"Process RSS now: {rss_mb()}")
note("RSS = pages of THIS process resident in RAM, incl. the Python interpreter, imported C libraries "
     "(numpy, faiss) and freed-but-not-returned memory. 'peak' = high-water mark.")
tracemalloc.start()
before = rss_mb()["rss_mb"]
vecs = np.ones((19_000, 384), dtype=np.float32)  # one shard's dense vectors
cur, peak = tracemalloc.get_traced_memory()
step(2, f"Allocate one shard's vectors: 19,000 x 384 x 4 bytes = {19_000 * 384 * 4 / 2**20:.1f} MB (arithmetic)")
print(f"     tracemalloc sees {cur / 2**20:.1f} MB; RSS grew {rss_mb()['rss_mb'] - before:.1f} MB")
docs = {str(i): "word " * 90 for i in range(19_000)}
cur2, _ = tracemalloc.get_traced_memory()
step(3, f"19,000 Python strings of ~450 chars: tracemalloc +{(cur2 - cur) / 2**20:.1f} MB "
        f"(raw text is {19_000 * 450 / 2**20:.1f} MB, plus per-object overhead)")
note("Python objects carry ~50 bytes of overhead each, and dicts hash tables on top. Millions of small "
     "objects (e.g. postings as Python lists) cost far more than the same data in numpy arrays. That is "
     "why our BM25 base segment is a scipy CSC matrix.")
del vecs, docs
gc.collect()
step(4, f"After del + gc: RSS {rss_mb()}  (often does NOT drop fully: the allocator keeps freed pages)")
note("So: report PEAK RSS for capacity planning (the 2 GB/shard limit), and use arithmetic + tracemalloc "
     "to explain WHERE it goes. A real shard is ~ torch+model (~400-500 MB) + vectors (28 MB) + "
     "postings (~20 MB) + text (~20 MB): the model dominates.")
try_this("Check results/bench.json max_shard_peak_rss_mb. Which component would you shrink first?",
         "Store vectors as float16. How much RAM does a shard save? What happens to faiss?",
         "Measure RSS before and after `import torch`. Surprised?")
