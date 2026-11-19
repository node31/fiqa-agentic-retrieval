"""Ablations (Part A3), dev split only.

1. Fusion sweep (quality only, cheap): retrieve BM25 top-100 and dense top-100 ONCE
   per query, then recompute every fusion variant offline. Fusion is pure
   arithmetic on these two lists, so this is exactly what the router would return
   (the router fuses the same merged lists) at a fraction of the cost.
       RRF with dense weight α ∈ {0.5 ... 0.9}, linear with α ∈ {0.1 ... 0.9}
2. Rerank sweep (quality + latency): cross-encoder over the top-k of the best
   first-stage config, k ∈ {10, 20, 50, 100}.

    python scripts/ablations.py --model minilm [--rerank]
Writes results/ablation_fusion_<model>.json and results/ablation_rerank_<model>.json.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beir_data import load_split  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.metrics import evaluate_run  # noqa: E402
from src.retrieval.hybrid import fuse  # noqa: E402
from src.service.router import single_node_router  # noqa: E402
from src.util import percentile  # noqa: E402

ALPHAS_RRF = [0.5, 0.6, 0.7, 0.8, 0.9]
ALPHAS_LIN = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def first_stage(router, queries):
    """Per query: global BM25 top-100 and dense top-100 (what the router fuses)."""
    out = {}
    for qid, q in queries.items():
        b = asyncio.run(router.search(q, k=100, strategy="bm25"))["results"]
        d = asyncio.run(router.search(q, k=100, strategy="dense"))["results"]
        out[qid] = ([(r["doc_id"], r["score"]) for r in b], [(r["doc_id"], r["score"]) for r in d])
    return out


def fusion_sweep(lists, qrels):
    rows = []
    for method, alphas in (("rrf", ALPHAS_RRF), ("linear", ALPHAS_LIN)):
        for a in alphas:
            run = {q: [d for d, _ in fuse(b, dn, method, a, top=10)] for q, (b, dn) in lists.items()}
            m = evaluate_run(run, qrels)
            rows.append({"method": method, "alpha_dense": a, **{k: round(v, 4) for k, v in m.items() if k != "n"}})
            print(f"  {method:6s} alpha={a:.1f}  nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f} MRR={m['mrr@10']:.4f}")
    return rows


def rerank_sweep(router, queries, qrels, method, alpha, ks=(10, 20, 50, 100)):
    rows = []
    for k in ks:
        run, lat = {}, []
        for qid, q in queries.items():
            t0 = time.perf_counter()
            res = asyncio.run(router.search(q, k=10, strategy="hybrid", fusion=method, alpha=alpha, rerank_k=k))
            lat.append((time.perf_counter() - t0) * 1e3)
            run[qid] = [r["doc_id"] for r in res["results"]]
        m = evaluate_run(run, qrels)
        rows.append({"rerank_k": k, **{x: round(v, 4) for x, v in m.items() if x != "n"},
                     "p50_ms": round(percentile(lat, 50), 1), "p95_ms": round(percentile(lat, 95), 1)})
        print(f"  rerank@{k:<3d} nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f} "
              f"p50/p95={rows[-1]['p50_ms']}/{rows[-1]['p95_ms']}ms")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--method", default="linear")
    ap.add_argument("--alpha", type=float, default=0.7)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    queries, qrels = load_split("dev")
    router = single_node_router(models=(a.model,))
    if a.rerank:
        rows = rerank_sweep(router, queries, qrels, a.method, a.alpha)
        (RESULTS_DIR / f"ablation_rerank_{a.model}.json").write_text(json.dumps(
            {"split": "dev", "model": a.model, "first_stage": {"fusion": a.method, "alpha": a.alpha}, "rows": rows},
            indent=1))
        return
    print(f"fusion sweep ({a.model}, dev):")
    rows = fusion_sweep(first_stage(router, queries), qrels)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"ablation_fusion_{a.model}.json").write_text(
        json.dumps({"split": "dev", "model": a.model, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
