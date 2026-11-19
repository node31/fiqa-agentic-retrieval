"""ch06: Part A end to end, built from YOUR ch01-ch05 code. Works on the tiny FiQA (seconds) and the full one.

Run the checker:   python -m pytest exercises/ch06_pipeline            (tiny, seconds)
                   python -m pytest exercises/ch06_pipeline --full     (57,638 docs; ~11 min the first time, ~3 min after)
Run the demo:      python exercises/ch06_pipeline/mini_pipeline.py [--full]
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TINY, cached_npy, dataset_from_argv, use_chapter  # noqa: E402

use_chapter("ch01_data", "ch02_metrics", "ch03_bm25", "ch04_dense", "ch05_fusion")
import loader  # noqa: E402,F401  (your ch01)
import my_bm25  # noqa: E402,F401  (your ch03)
import my_dense  # noqa: E402,F401  (your ch04)
import my_fusion  # noqa: E402,F401  (your ch05)
import my_metrics  # noqa: E402,F401  (your ch02)

SYSTEMS = ("bm25", "dense", "hybrid")


def doc_vectors(corpus: dict[str, str], dataset: str):
    """GIVEN (you don't write this): your my_dense.encode() over every doc, run ONCE and cached in exercises/.cache/.
    Row i belongs to list(corpus)[i]. Tiny: about 1 s. Full: about 10 minutes the first time, then instant.
    The cache key includes your encode()'s source code, so editing encode() recomputes automatically.
    """
    def compute():
        print(f"[ch06] encoding {len(corpus):,} docs with YOUR my_dense.encode (cached afterwards)...", flush=True)
        return my_dense.encode(list(corpus.values()))
    key = inspect.getsource(my_dense.encode) + my_dense.MODEL_NAME + str(len(corpus))
    return cached_npy(f"ch06-{dataset}-docvecs", compute, key)


def run(data_dir: Path = TINY, k: int = 10, alpha: float = 0.7, depth: int = 100) -> dict:
    """Return {"rankings": {system: {qid: [doc ids, best first, length <= k]}},
               "metrics":  {system: my_metrics.evaluate(rankings[system], qrels)}}

    1. corpus = loader.load_jsonl(data_dir / "corpus.jsonl");  queries, qrels = loader.load_split(data_dir, "dev")
    2. bm25:   index = my_bm25.build_index(corpus) ONCE; per query my_bm25.search(index, query, depth)
    3. dense:  vecs = doc_vectors(corpus, data_dir.name) ONCE; per query my_dense.top_k(encode([query])[0], vecs,
               list(corpus), depth).  Tip: encode all queries in one call, it's much faster than one by one.
    4. hybrid: my_fusion.linear(bm25 top-`depth`, dense top-`depth`, alpha, top=k)
    5. cut bm25/dense to k; metrics with my_metrics.evaluate
    WHY depth=100 for fusion: a doc ranked 30th by BM25 but 2nd by dense must reach the fusion step to be rescued.
    """
    raise NotImplementedError("TODO (ch06): wire your ch01-ch05 functions together")


if __name__ == "__main__":
    name, data_dir = dataset_from_argv()
    t0 = time.perf_counter()
    out = run(data_dir)
    print(f"[{name}] {time.perf_counter() - t0:.1f}s\n{'system':8s} {'nDCG@10':>8s} {'R@10':>6s} {'MRR':>6s}")
    for s in SYSTEMS:
        m = out["metrics"][s]
        print(f"{s:8s} {m['ndcg@10']:8.3f} {m['recall@10']:6.3f} {m['mrr@10']:6.3f}")
    if name == "full":
        print("ours:    bm25 0.264 · dense 0.388 · hybrid 0.419  (results/, docs/walkthroughs/part_a_results.md)")
    queries, qrels = loader.load_split(data_dir, "dev")
    for qid in list(queries)[:4]:                        # the tiny set has 4 queries; show 4 on full too
        print(f"\n{qid} {queries[qid]!r}  gold={sorted(qrels[qid])}")
        for s in SYSTEMS:
            print(f"  {s:7s}", out["rankings"][s][qid][:5])
