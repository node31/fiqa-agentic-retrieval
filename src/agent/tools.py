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

HOW IT FITS
    A "tool" here is just a Python callable our agent code runs when the model's JSON
    action asks for it. The model never calls it directly. The chain is:
    model JSON {"action": "search", ...} -> agent code -> ``SearchTool(query, strategy)``
    -> search_fn (HTTP or in-process) -> hits -> ``render_hits`` -> text in the next prompt.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable

# The only strategy names the search service accepts (keyword, embedding, or both fused).
STRATEGIES = ("bm25", "dense", "hybrid")
# Any function (query, strategy, k) -> list of hit dicts. Hits need "doc_id" and "score"; "text"
# is optional. Both factories below return one, so SearchTool works with either backend.
SearchFn = Callable[[str, str, int], list[dict]]


def http_search_fn(url: str) -> SearchFn:
    """SearchFn that POSTs to a running search service (router) at ``url``."""
    import httpx

    client = httpx.Client(base_url=url, timeout=10)  # one keep-alive client, shared by threads

    def fn(query: str, strategy: str, k: int) -> list[dict]:
        # include_text=True: we need passage text to build the snippets the model reads.
        r = client.post("/search", json={"query": query, "k": k, "strategy": strategy, "include_text": True})
        # Turn HTTP errors (4xx/5xx) into exceptions; the caller decides (the orchestrator
        # records them per worker).
        r.raise_for_status()
        return r.json()["results"]

    return fn


def local_search_fn(router) -> SearchFn:
    """SearchFn that calls an in-process router directly (no HTTP server needed).

    ``router.search`` is async (a coroutine), but agent workers are plain threads, so each
    thread drives the coroutine on its own event loop.
    """
    # threading.local(): each thread sees its OWN ``local.loop`` attribute.
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
    """The search tool the agents use: ``tool(query, strategy) -> [{"doc_id", "score", "text"}]``.

    Wraps any SearchFn, validates the strategy, counts calls, and truncates passage text to
    ``snippet_chars`` so prompts stay small. The call counter is lock-protected; whether the
    search itself is thread-safe depends on ``search_fn`` (both factories above are).
    """

    def __init__(self, search_fn: SearchFn, k: int = 10, snippet_chars: int = 200):
        self.search_fn, self.k, self.snippet_chars = search_fn, k, snippet_chars
        # The lock only protects the call counter (workers call this tool in parallel).
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, query: str, strategy: str = "hybrid") -> list[dict]:
        # Validate LLM-produced arguments (see PITFALL in the module docstring).
        if strategy not in STRATEGIES:
            strategy = "hybrid"
        with self._lock:
            self.calls += 1
        hits = self.search_fn(query, strategy, self.k)
        # Keep only what the agents need, and cut each passage to a short snippet (token cost).
        return [{"doc_id": h["doc_id"], "score": h["score"], "text": h.get("text", "")[: self.snippet_chars]}
                for h in hits]


def render_hits(hits: list[dict]) -> str:
    """Compact, token-cheap rendering: one line per hit.

    Format: ``[doc_id] snippet``. The id in brackets is what the model must copy back in a
    finish action. ``" ".join(text.split())`` collapses newlines/runs of spaces so each hit
    stays on one line.
    """
    return "\n".join(f"[{h['doc_id']}] {' '.join(h['text'].split())}" for h in hits) or "(no results)"
