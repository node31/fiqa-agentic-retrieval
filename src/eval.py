"""Evaluation harness: quality, stratified quality, cold/warm latency and peak RAM.

WHAT
    Runs one retrieval CONFIGURATION over a query split and reports:
      overall     nDCG@10, Recall@10, MRR@10
      stratified  Recall@10 by query length (short <5 / medium 5-15 / long >15 words)
                  Recall@10 for queries whose gold passage is among the 10% longest docs vs the rest
      latency     cold p50/p95 (first 20 queries of a FRESH process) and
                  warm p50/p95 (all dev queries, after a 100-query warmup on TRAIN queries)
      memory      peak RSS of the process doing the search (local) or max over shards (service)

    python -m src.eval --mode local   --strategy hybrid --model minilm --name hybrid-rrf
    python -m src.eval --mode service --strategy hybrid --url http://127.0.0.1:8000

PROTOCOL DECISIONS (docs/07_benchmarking_latency_memory.md)
    * Warmup uses TRAIN queries, so no dev query is ever "seen" before it is timed.
    * Each configuration runs in its own fresh process (scripts/bench.py does this),
      so "cold" really is cold: first inference, cold CPU caches, new connections.
    * Latency is measured by the CLIENT, end to end: for service mode that includes
      HTTP to the router, so it is the number the 50 ms constraint refers to.
    * The test split is locked behind --final-test-run. Tune on dev only.

PITFALLS
    * Percentiles from 20 samples are noisy: p95 of 20 is the 19th value.
    * Don't average p95s across runs. Recompute from raw samples.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from src.beir_data import load_corpus, load_split
from src.config import PRIMARY_MODEL, RESULTS_DIR, SEED
from src.metrics import evaluate_run, recall_at_k
from src.util import percentile, rss_mb


def length_bucket(query: str) -> str:
    n = len(query.split())
    return "short(<5)" if n < 5 else "medium(5-15)" if n <= 15 else "long(>15)"


def long_doc_ids(corpus: dict[str, str], top_frac: float = 0.10) -> set[str]:
    lens = sorted(((len(t.split()), d) for d, t in corpus.items()), reverse=True)
    return {d for _, d in lens[: int(len(lens) * top_frac)]}


def stratified(run: dict, qrels: dict, queries: dict, long_docs: set[str]) -> dict:
    """Recall@10 per stratum, with the number of queries in each (always report n!)."""
    groups: dict[str, list[float]] = {}
    for qid, ranked in run.items():
        r = recall_at_k(ranked, qrels[qid], 10)
        groups.setdefault("qlen:" + length_bucket(queries[qid]), []).append(r)
        # A query is in the "long gold" group if ANY of its gold docs is in the top-10% longest.
        key = "gold_in_top10pct_longest" if set(qrels[qid]) & long_docs else "gold_not_longest"
        groups.setdefault("doclen:" + key, []).append(r)
    return {g: {"recall@10": round(sum(v) / len(v), 4), "n": len(v)} for g, v in sorted(groups.items())}


def latency_summary(ms: list[float]) -> dict:
    return {"p50_ms": round(percentile(ms, 50), 2), "p95_ms": round(percentile(ms, 95), 2), "n": len(ms)}


class Searcher:
    """Uniform ``search(query) -> ranked doc ids`` over the in-process router or the HTTP service."""

    def __init__(self, mode: str, url: str | None = None, **params):
        self.mode, self.params = mode, params
        if mode == "local":
            from src.service.router import single_node_router

            self.loop = asyncio.new_event_loop()
            model = params.pop("model", PRIMARY_MODEL)
            models = () if params.get("strategy") == "bm25" else (model,)
            self.router = single_node_router(models=models, active_model=model,
                                             bm25_stats=params.pop("bm25_stats", "global"))
        else:
            import httpx

            params.pop("model", None)
            params.pop("bm25_stats", None)
            self.http = httpx.Client(base_url=url, timeout=30)  # persistent keep-alive client

    def search(self, query: str, k: int = 10) -> list[str]:
        if self.mode == "local":
            out = self.loop.run_until_complete(self.router.search(query, k=k, **self.params))
        else:
            r = self.http.post("/search", json={"query": query, "k": k, **self.params})
            r.raise_for_status()
            out = r.json()
            if out.get("partial"):
                raise RuntimeError(f"partial result during eval: {out['missing_shards']}")
        return [x["doc_id"] for x in out["results"]]

    def memory(self) -> dict:
        if self.mode == "local":
            return {"process": rss_mb()}
        st = self.http.get("/admin/status").json()
        shards = {i: {"rss_mb": s["rss_mb"], "peak_rss_mb": s["peak_rss_mb"]} for i, s in st["shards"].items()}
        return {"shards": shards, "max_shard_peak_rss_mb": max(s["peak_rss_mb"] for s in shards.values())}


def evaluate(searcher: Searcher, split: str = "dev", cold_n: int = 20, warmup_n: int = 100,
             limit: int | None = None) -> dict:
    queries, qrels = load_split(split)
    items = list(queries.items())[:limit]

    def timed(q: str):
        t0 = time.perf_counter()
        ids = searcher.search(q)
        return ids, (time.perf_counter() - t0) * 1e3

    cold = [timed(q)[1] for _, q in items[:cold_n]]  # 1) fresh process: first N queries
    train_q, _ = load_split("train")
    rng = random.Random(SEED)
    for q in rng.sample(list(train_q.values()), warmup_n):  # 2) warmup on TRAIN queries
        searcher.search(q)
    run, warm = {}, []
    for qid, q in items:  # 3) the measured run
        run[qid], ms = timed(q)
        warm.append(ms)
    overall = {k: round(v, 4) if isinstance(v, float) else v for k, v in evaluate_run(run, qrels).items()}
    return {
        "split": split, "overall": overall,
        "stratified": stratified(run, qrels, queries, long_doc_ids(load_corpus())),
        "latency": {"cold_first20": latency_summary(cold), "warm": latency_summary(warm)},
        "memory": searcher.memory(), "run": run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate one retrieval configuration")
    ap.add_argument("--mode", choices=["local", "service"], default="local")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategy", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--fusion", choices=["rrf", "linear"], default="rrf")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rerank-k", type=int, default=0)
    ap.add_argument("--bm25-stats", choices=["global", "local"], default="global")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final-test-run", action="store_true", help="required to touch the test split")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=4, help="torch intra-op threads (local mode)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.split == "test" and not a.final_test_run:
        raise SystemExit("Refusing to evaluate on TEST without --final-test-run (tune on dev only!)")
    import torch

    torch.set_num_threads(a.threads)
    t0 = time.time()
    searcher = Searcher(a.mode, a.url, strategy=a.strategy, model=a.model, fusion=a.fusion, alpha=a.alpha,
                        rerank_k=a.rerank_k, bm25_stats=a.bm25_stats)
    load_s = time.time() - t0
    res = evaluate(searcher, a.split, limit=a.limit)
    res.update(name=a.name, config={k: v for k, v in vars(a).items() if k not in ("out", "url", "final_test_run")},
               startup_s=round(load_s, 1))
    out = a.out or RESULTS_DIR / "runs" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    o, lat = res["overall"], res["latency"]
    print(f"{a.name:28s} nDCG@10={o['ndcg@10']:.4f} R@10={o['recall@10']:.4f} MRR@10={o['mrr@10']:.4f} "
          f"warm p50/p95={lat['warm']['p50_ms']}/{lat['warm']['p95_ms']}ms "
          f"cold p50/p95={lat['cold_first20']['p50_ms']}/{lat['cold_first20']['p95_ms']}ms")


if __name__ == "__main__":
    main()
