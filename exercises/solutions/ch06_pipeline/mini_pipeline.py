"""SOLUTION for ch06: Part A end to end from the ch01-ch05 solutions. Tiny FiQA (seconds) or full (--full).

Try the exercise yourself first (exercises/ch06_pipeline/mini_pipeline.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch06_pipeline [--full]
"""
from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EXERCISES_SOLUTIONS", "1")   # WHY: make use_chapter() load the other SOLUTION files
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import TINY, cached_npy, dataset_from_argv, use_chapter  # noqa: E402

use_chapter("ch01_data", "ch02_metrics", "ch03_bm25", "ch04_dense", "ch05_fusion")
import loader  # noqa: E402
import my_bm25  # noqa: E402
import my_dense  # noqa: E402
import my_fusion  # noqa: E402
import my_metrics  # noqa: E402

SYSTEMS = ("bm25", "dense", "hybrid")


def doc_vectors(corpus: dict[str, str], dataset: str):
    def compute():
        print(f"[ch06] encoding {len(corpus):,} docs with my_dense.encode (cached afterwards)...", flush=True)
        return my_dense.encode(list(corpus.values()))
    key = inspect.getsource(my_dense.encode) + my_dense.MODEL_NAME + str(len(corpus))
    return cached_npy(f"ch06-{dataset}-docvecs", compute, key)


def run(data_dir: Path = TINY, k: int = 10, alpha: float = 0.7, depth: int = 100) -> dict:
    data_dir = Path(data_dir)
    # 1. data
    corpus = loader.load_jsonl(data_dir / "corpus.jsonl")
    queries, qrels = loader.load_split(data_dir, "dev")
    qids = list(queries)

    # 2. the two "offline" steps, done ONCE (this is what an index build is)
    index = my_bm25.build_index(corpus)                    # ~1 min on 57K docs in pure Python
    doc_ids = list(corpus)                                 # row i of vecs ↔ doc_ids[i]
    vecs = doc_vectors(corpus, data_dir.name)              # ~10 min on 57K docs, then cached
    q_vecs = my_dense.encode([queries[q] for q in qids])   # WHY one batch: 500 queries in ~2 s instead of ~8 s

    # 3. the "online" steps, per query
    rankings: dict[str, dict[str, list[str]]] = {s: {} for s in SYSTEMS}
    for i, qid in enumerate(qids):
        bm25 = my_bm25.search(index, queries[qid], depth)            # top-100 candidates
        dense = my_dense.top_k(q_vecs[i], vecs, doc_ids, depth)      # top-100 candidates
        rankings["bm25"][qid] = [d for d, _ in bm25[:k]]
        rankings["dense"][qid] = [d for d, _ in dense[:k]]
        # PITFALL: fuse the DEEP lists, then cut. Cutting each list to 10 first would lose the docs that only
        # become top-10 after fusion (that's the whole point of hybrid).
        rankings["hybrid"][qid] = [d for d, _ in my_fusion.linear(bm25, dense, alpha, top=k)]

    # 4. score each system with the same metric code
    return {"rankings": rankings, "metrics": {s: my_metrics.evaluate(rankings[s], qrels) for s in SYSTEMS}}


if __name__ == "__main__":
    name, data_dir = dataset_from_argv()
    t0 = time.perf_counter()
    out = run(data_dir)
    print(f"[{name}] {time.perf_counter() - t0:.1f}s\n{'system':8s} {'nDCG@10':>8s} {'R@10':>6s} {'MRR':>6s}")
    for s in SYSTEMS:
        m = out["metrics"][s]
        print(f"{s:8s} {m['ndcg@10']:8.3f} {m['recall@10']:6.3f} {m['mrr@10']:6.3f}")
    queries, qrels = loader.load_split(data_dir, "dev")
    for qid in list(queries)[:4]:
        print(f"\n{qid} {queries[qid]!r}  gold={sorted(qrels[qid])}")
        for s in SYSTEMS:
            print(f"  {s:7s}", out["rankings"][s][qid][:5])
