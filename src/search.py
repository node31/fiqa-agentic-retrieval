"""CLI: query the RUNNING service (router) and print ranked passages with scores.

    python src/search.py --query "what is short selling?" --top-k 10 --strategy hybrid

Start the service first: ``python -m src.service.cluster`` (local processes) or
``docker compose up``. The router URL defaults to $ROUTER_URL or http://127.0.0.1:8000.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python src/search.py`

import httpx  # noqa: E402

from src.config import DEFAULT_ALPHA, DEFAULT_FUSION, ROUTER_URL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Search the FiQA retrieval service")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--strategy", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--fusion", choices=["rrf", "linear"], default=DEFAULT_FUSION)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--rerank-k", type=int, default=0, help="cross-encoder rerank depth (0 = off)")
    ap.add_argument("--url", default=ROUTER_URL)
    a = ap.parse_args()
    try:
        r = httpx.post(f"{a.url}/search", timeout=30, json={
            "query": a.query, "k": a.top_k, "strategy": a.strategy, "fusion": a.fusion,
            "alpha": a.alpha, "rerank_k": a.rerank_k, "include_text": True})
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"error talking to router at {a.url}: {e}", file=sys.stderr)
        return 1
    out = r.json()
    print(f"query: {a.query!r}  strategy={out['strategy']} model={out['model']} "
          f"server_latency={out['latency_ms']}ms")
    if out["partial"]:
        print(f"WARNING: partial results, shards missing: {out['missing_shards']}")
    for i, res in enumerate(out["results"], 1):
        snippet = textwrap.shorten(res.get("text", ""), 220)
        print(f"{i:2d}. [{res['score']:.4f}] doc {res['doc_id']}: {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
