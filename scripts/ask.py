"""Ask ONE question and see the answer: plain search (bm25 / dense / hybrid) or an agent (single / multi).

    python scripts/ask.py "what is a roth ira?"                                   # hybrid, in-process
    python scripts/ask.py "what is a roth ira?" --strategy bm25
    python scripts/ask.py "what is a roth ira?" --url http://localhost:8000       # through the router
    python scripts/ask.py "roth ira and mortgage offset?" --strategy single --backend azure --confirm-spend
    python scripts/ask.py "roth ira and mortgage offset?" --strategy multi  --backend azure --confirm-spend
    python scripts/ask.py "what is a roth ira?" --url http://localhost:8000 --token $(python -m scripts.mint_token alice)

WHERE THE SEARCH RUNS
    default   in-process, NOT sharded: ``single_node_router()`` loads the whole corpus (BM25 + the
              MiniLM embedding cache) into this Python process and loads the encoder. No server
              needed; startup took ~30 s on my laptop.
    --url     a running router (the Docker stack or ``python -m src.service.cluster``), over HTTP.
              The agents' searches go there too.

WHAT IT PRINTS
    bm25 / dense / hybrid   the top-k passages (rank, score, doc id, snippet) and the time taken.
    single                  every step the agent took (searches it asked for, then finish), the final
                            top-k, LLM calls, tokens and the estimated cost.
    multi                   the planner's split (or "simple"), the verifier's rewrites, the final top-k,
                            LLM calls, tokens and the estimated cost.

SECURITY (docs/design/SECURITY.md)
    With --url, ``--token`` (or env FIQA_TOKEN) makes every search run as that user: once an admin turns on
    identity and tenant_filter, the agent only sees that tenant's passages. Before an agent's LLM sees the
    question, the router's guardrails check it (POST /guardrails/check): a refused question is never sent
    to the model and costs no tokens. A refusal prints the rule that fired and exits with code 3.

COST
    The agents need an LLM. ``--backend azure`` (gpt-6-sol, key in .env) costs money and needs
    ``--confirm-spend``; one question is roughly 0.1K to 3.4K tokens (well under a cent at the
    placeholder prices in src/config.py). Answers are cached in .llm_cache/, so asking the same
    question again is free. ``--backend copilot`` (free, slow) and ``--backend fake`` (scripted, for
    plumbing only) also work.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.tools import SearchTool, http_guardrail_check, http_search_fn, local_search_fn  # noqa: E402


def make_search_fn(url: str | None, token: str | None = None):
    """(query, strategy, k) -> hits with text: the router at ``url``, or an in-process single node."""
    if url:
        return http_search_fn(url, token)
    from src.service.router import single_node_router

    print("loading the corpus, indexes and encoder in-process (single node, ~30 s) ...", flush=True)
    t0 = time.perf_counter()
    router = single_node_router()
    fn = local_search_fn(router)
    fn("warm up", "hybrid", 1)   # the MiniLM encoder loads lazily on the first dense query; keep that out of the timing
    print(f"ready in {time.perf_counter() - t0:.1f} s", flush=True)
    return fn


def show_hits(ids: list[str], texts: dict[str, str], scores: dict[str, float] | None = None, width: int = 200) -> None:
    for i, d in enumerate(ids, 1):
        score = f"[{scores[d]:.4f}] " if scores and d in scores else ""
        print(f"{i:2d}. {score}doc {d}: {textwrap.shorten(texts.get(d, ''), width)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=["bm25", "dense", "hybrid", "single", "multi"], default="hybrid")
    ap.add_argument("--url", help="router URL, e.g. http://localhost:8000 (default: in-process single node)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--backend", choices=["azure", "copilot", "fake"], default="azure", help="LLM for single / multi")
    ap.add_argument("--confirm-spend", action="store_true", help="required for --backend azure (it costs money)")
    ap.add_argument("--token", default=os.environ.get("FIQA_TOKEN"),
                    help="bearer token for --url (scripts/mint_token.py); default env FIQA_TOKEN")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # a cp1252 console would choke on some passage text

    llm = None
    if a.strategy in ("single", "multi"):
        if a.backend == "azure" and not a.confirm_spend:
            ap.error("--backend azure costs money: add --confirm-spend (or use --backend copilot / fake)")
        from src.agent.llm import get_llm

        llm = get_llm(a.backend, confirm_spend=a.confirm_spend)

    if a.url and llm is not None:
        verdict = http_guardrail_check(a.url, a.token)(a.question)
        if not verdict["allowed"]:
            rules = ", ".join(f.get("category") or f["rule"] for f in verdict["findings"])
            print(f"refused by the router's guardrails ({rules}); the question was not sent to the LLM")
            return 3
        for f in verdict["findings"]:
            print(f"guardrail flagged (log mode, not blocked): {f.get('category') or f['rule']}")
    raw = make_search_fn(a.url, a.token)
    texts: dict[str, str] = {}

    def recording(query: str, strategy: str, k: int) -> list[dict]:
        hits = raw(query, strategy, k)
        for h in hits:   # remember full texts so the final answer can be printed with snippets
            texts.setdefault(h["doc_id"], h.get("text", ""))
        return hits

    print(f"\nquestion: {a.question!r}   strategy: {a.strategy}   search: {a.url or 'in-process single node'}")
    t0 = time.perf_counter()
    if llm is None:
        try:
            hits = recording(a.question, a.strategy, a.k)
        except Exception as e:  # httpx.HTTPStatusError: show the router's reason, not a traceback
            resp = getattr(e, "response", None)
            if resp is None or resp.status_code not in (401, 403, 429):
                raise
            print(f"HTTP {resp.status_code}: {resp.json().get('detail')}")
            return 3
        ms = (time.perf_counter() - t0) * 1e3
        print(f"{len(hits)} hits in {ms:.1f} ms\n")
        show_hits([h["doc_id"] for h in hits], texts, {h["doc_id"]: h["score"] for h in hits})
        return 0

    tool = SearchTool(recording, k=a.k)
    if a.strategy == "single":
        from src.agent.single_agent import run_single_agent

        res = run_single_agent(a.question, llm, tool, k=a.k)
        print("\nstep 0: search the question as asked (hybrid, no LLM call)")
        for t in res.trace:
            act = t["action"]
            if act.get("action") == "search":
                print(f"step {t['step']}: the model asked for another search ({act.get('strategy', 'hybrid')}): "
                      f"{act.get('query')!r}")
            else:
                print(f"step {t['step']}: the model finished and picked {len(act.get('doc_ids') or [])} ids "
                      f"(the rest of the top {a.k} is filled from all searches, fused with RRF)")
    else:
        from src.agent.orchestrator import Orchestrator

        res = Orchestrator(llm, tool, k=a.k).run(a.question)
        for t in res.trace:
            if "plan" in t:
                how = {"gate:simple": "the free gate (short, single-clause question): no planner call",
                       "planner:simple": "the planner said: one topic, no split",
                       "planner:decomposed": "the planner split it",
                       "planner:failed": "the planner failed: searched the question as asked"}.get(t["decided_by"], t["decided_by"])
                print(f"\nplan: {how}")
                for i, s in enumerate(t["plan"]):
                    print(f"  sub-query {i}: {s!r}")
            if "verifier_rewrites" in t:
                rw = t["verifier_rewrites"]
                print("verifier: every sub-query's results looked on-topic" if not rw else
                      "verifier rewrote: " + "; ".join(f"sub-query {i} -> {q!r}" for i, q in rw.items()))
    print(f"\nfinal top {len(res.doc_ids)} ({res.searches} searches, {res.latency_s:.1f} s):")
    show_hits(res.doc_ids, texts)
    print(f"\nLLM calls {res.llm_calls}, tokens {res.input_tokens} in + {res.output_tokens} out, "
          f"estimated ${res.cost_usd:.5f} (placeholder prices in src/config.py)")
    for e in res.errors:
        print(f"recovered from: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
