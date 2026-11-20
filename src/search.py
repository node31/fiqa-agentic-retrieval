"""CLI: query the RUNNING service (router) and print ranked passages with scores.

    python src/search.py --query "what is short selling?" --top-k 10 --strategy hybrid

Start the service first: ``python -m src.service.cluster`` (local processes) or
``docker compose up``. The router URL defaults to $ROUTER_URL or http://127.0.0.1:8000.

HOW IT WORKS
    1. parse the command-line flags (defaults for --fusion/--alpha come from
       src/config.py, i.e. the shipped linear / 0.7 system)
    2. send ONE HTTP POST to <router>/search with those settings as JSON
    3. print each result: rank, score, doc id and a ~220-character snippet
    This file never touches an index itself: all the searching happens in the
    router and its shards. It is a thin client, handy for eyeballing results.
    Exit code: 0 = ok, 1 = could not talk to the router.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python src/search.py`
# WHY: run as a plain script, Python only puts src/ itself on the import path, so
# `import src.config` would fail. Adding the repo root first fixes that. The
# imports below must come AFTER this line, hence the `noqa: E402`
# ("module-level import not at top of file") markers that silence the linter.

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
    # Send the query to the router. The JSON keys mirror the router's /search API;
    # include_text=True asks it to return passage text too (for the snippets).
    try:
        r = httpx.post(f"{a.url}/search", timeout=30, json={
            "query": a.query, "k": a.top_k, "strategy": a.strategy, "fusion": a.fusion,
            "alpha": a.alpha, "rerank_k": a.rerank_k, "include_text": True})
        # Turn an HTTP error status (4xx/5xx) into an exception, handled just below.
        r.raise_for_status()
    # HTTPError covers both "router not running / timed out" and bad status codes:
    # print a friendly one-liner instead of a long traceback.
    except httpx.HTTPError as e:
        print(f"error talking to router at {a.url}: {e}", file=sys.stderr)
        return 1
    out = r.json()
    # server_latency = time measured INSIDE the router (excludes the HTTP round trip
    # and this script's start-up), so it is lower than what a client experiences.
    print(f"query: {a.query!r}  strategy={out['strategy']} model={out['model']} "
          f"server_latency={out['latency_ms']}ms")
    # "partial" = some shard didn't answer in time, so results come only from the
    # other shards' part of the corpus (good documents may be missing).
    if out["partial"]:
        print(f"WARNING: partial results, shards missing: {out['missing_shards']}")
    for i, res in enumerate(out["results"], 1):
        # textwrap.shorten collapses whitespace and cuts at a word boundary, adding "[...]".
        snippet = textwrap.shorten(res.get("text", ""), 220)
        print(f"{i:2d}. [{res['score']:.4f}] doc {res['doc_id']}: {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
