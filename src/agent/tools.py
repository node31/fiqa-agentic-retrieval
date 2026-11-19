"""The agents' single tool: ``search(query, strategy)`` backed by our retrieval service.

WHAT
    ``SearchTool`` turns a search backend into what an LLM can consume: the top-k
    hits as (doc_id, short snippet). Snippets are truncated on purpose. Every
    character we show the model costs tokens, and 200 chars per passage is enough
    to judge relevance (docs/14_agent_evaluation_cost.md does the arithmetic).

BACKENDS
    http_search_fn(url)       the running router (what production agents would use)
    local_search_fn(router)   in-process single-node router (fast evals, no servers)

PITFALL
    The model may hallucinate a strategy name. We validate and fall back to "hybrid"
    instead of crashing. Never trust tool arguments produced by an LLM.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable

STRATEGIES = ("bm25", "dense", "hybrid")
SearchFn = Callable[[str, str, int], list[dict]]


def http_search_fn(url: str) -> SearchFn:
    import httpx

    client = httpx.Client(base_url=url, timeout=10)  # one keep-alive client, shared by threads

    def fn(query: str, strategy: str, k: int) -> list[dict]:
        r = client.post("/search", json={"query": query, "k": k, "strategy": strategy, "include_text": True})
        r.raise_for_status()
        return r.json()["results"]

    return fn


def local_search_fn(router) -> SearchFn:
    local = threading.local()

    def fn(query: str, strategy: str, k: int) -> list[dict]:
        # WHY one persistent loop PER THREAD: agent workers run in plain threads. asyncio.run() per
        # call would create (and tear down) a new loop + executor every time, which leaked threads
        # and torch per-thread state in our first ablation runs (process grew to 9 GB).
        loop = getattr(local, "loop", None)
        if loop is None:
            loop = local.loop = asyncio.new_event_loop()
        return loop.run_until_complete(router.search(query, k=k, strategy=strategy, include_text=True))["results"]

    return fn


class SearchTool:
    def __init__(self, search_fn: SearchFn, k: int = 10, snippet_chars: int = 200):
        self.search_fn, self.k, self.snippet_chars = search_fn, k, snippet_chars
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, query: str, strategy: str = "hybrid") -> list[dict]:
        if strategy not in STRATEGIES:
            strategy = "hybrid"
        with self._lock:
            self.calls += 1
        hits = self.search_fn(query, strategy, self.k)
        return [{"doc_id": h["doc_id"], "score": h["score"], "text": h.get("text", "")[: self.snippet_chars]}
                for h in hits]


def render_hits(hits: list[dict]) -> str:
    """Compact, token-cheap rendering: one line per hit."""
    return "\n".join(f"[{h['doc_id']}] {' '.join(h['text'].split())}" for h in hits) or "(no results)"
