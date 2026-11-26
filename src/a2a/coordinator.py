"""A coordinator agent that is itself an A2A server, and delegates to the FiQA agent over A2A.

    python -m src.a2a.server --router http://localhost:8000                 # the FiQA agent on :8300
    python -m src.a2a.coordinator --agent http://localhost:8300             # the coordinator on :8310
    python -m src.a2a.client "what is a roth ira, and how do I buy my first home?" --agent http://localhost:8310

TWO AGENTS, TWO ROLES
    The FiQA agent (src/a2a/server.py) knows the corpus: it searches it, keeps the user's memory, and answers
    one question at a time. The coordinator knows nothing about finance. Its one skill is splitting a compound
    question ("what is X, and how do I Y?") into parts, asking a specialist each part, and merging the answers.
    It finds the specialist the way any A2A client does: it reads the FiQA agent's card, checks the protocol
    version, and sends each part as a SendMessage. To its own callers it is an ordinary A2A agent with a card,
    tasks, GetTask, ListTasks and CancelTask: this module builds it from the same create_app as the FiQA agent,
    with a different card and a different search function.

    caller --SendMessage--> coordinator --SendMessage (part 1)--> FiQA agent --/search--> router
                                        --SendMessage (part 2)--> FiQA agent --/search--> router
           <---- Task (fused top k) ----  <---- Task, Task ------

IDENTITY IS PASSED ALONG
    The coordinator calls the FiQA agent with the CALLER's bearer token, so the FiQA agent and the router see
    alice as alice: her tenant filter, her memory, her rate limit. Without it every delegated call would run as
    the coordinator, and two users' searches and memories would mix. (In production the coordinator would trade
    the user's token for a narrower one made for this hop, an OAuth "on-behalf-of" token exchange, instead of
    forwarding the original.)

    >>> split_question("what is a roth ira, and how do I buy my first home?")
    ['what is a roth ira', 'how do I buy my first home']
    >>> rrf([["a", "b"], ["c", "a"]], k=3)       # "a" is in both lists, so it wins
    ['a', 'c', 'b']
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import httpx

from src.a2a import protocol as P
from src.a2a.client import A2AClient
from src.agent.fake_policies import split_question

__all__ = ["split_question", "rrf", "coordinator_card", "delegating_search", "create_coordinator_app"]


def rrf(rankings: list[list[str]], k: int = 10, c: int = 60) -> list[str]:
    """Reciprocal rank fusion: each list votes 1 / (c + rank) for its docs; ties keep first-seen order."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0.0) + 1.0 / (c + r)
    return sorted(score, key=lambda d: -score[d])[:k]


def coordinator_card(base_url: str, agent_url: str) -> dict:
    return {
        "name": "FiQA coordinator",
        "description": "Splits a compound personal-finance question into its parts, asks the FiQA retrieval agent "
                       f"({agent_url}) each part over A2A, and merges the passages. Uses no LLM.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": f"{base_url.rstrip('/')}/a2a", "protocolBinding": "JSONRPC",
                                 "protocolVersion": P.PROTOCOL_VERSION}],
        "provider": {"organization": "fiqa-agentic-retrieval (learning project)",
                     "url": "https://github.com/node31/fiqa-agentic-retrieval"},
        "documentationUrl": "https://github.com/node31/fiqa-agentic-retrieval/blob/main/docs/topics/21_a2a.md",
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {
            "scheme": "Bearer", "bearerFormat": "JWT",
            "description": "Forwarded to the FiQA agent, so the caller's tenant and memory apply there."}}},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [{"id": "delegate", "name": "Split and delegate",
                    "description": "Splits a question with two topics, sends each part to the FiQA agent, and fuses "
                                   "the answers with reciprocal rank fusion.",
                    "tags": ["delegation", "compound", "finance"],
                    "examples": ["what is a roth ira, and how do I buy my first home?"]}],
    }


def delegating_search(agent_url: str, http_factory: Callable[[str | None], httpx.Client] | None = None,
                      part_skill: str = "retrieve"):
    """search_factory for create_app: (token) -> (query, strategy, k) -> fused hits from the FiQA agent.

    ``http_factory(token)`` gives the HTTP client for one caller (tests pass a TestClient; the two-agent demo
    passes a client that records every message). The FiQA agent's card is read once per caller. ``part_skill`` is
    the FiQA agent's skill each part goes to: "retrieve" (no LLM) or "answer" (the FiQA agent's LLM also writes a
    short answer per part; the coordinator still fuses only the passages, and uses no LLM itself).
    """
    def factory(token: str | None):
        agent = A2AClient(agent_url, token, http=http_factory(token) if http_factory else None)

        def search(query: str, strategy: str, k: int) -> list[dict]:
            parts = split_question(query)
            first = agent.send(parts[0], k=k, skill=part_skill)
            tasks = [first] + [agent.send(p, context_id=first["contextId"], k=k, skill=part_skill) for p in parts[1:]]
            hits: dict[str, dict] = {}
            for t in tasks:
                if t["status"]["state"] != P.COMPLETED:   # e.g. REJECTED by the router's guardrails
                    msg = (t["status"].get("message") or {"parts": [{"text": t["status"]["state"]}]})
                    raise RuntimeError(f"the FiQA agent answered {t['status']['state']}: {P.text_of(msg)}")
                for art in t["artifacts"]:
                    for part in art["parts"]:
                        for h in (part.get("data") or {}).get("hits", []):
                            hits.setdefault(h["doc_id"], h)
            fused = rrf([[h["doc_id"] for h in t["artifacts"][0]["parts"][0]["data"]["hits"]] for t in tasks], k=k)
            return [{"doc_id": d, "score": 0.0, "text": hits[d].get("snippet", "")} for d in fused]

        return search

    return factory


def create_coordinator_app(agent_url: str, base_url: str = "http://localhost:8310",
                           http_factory: Callable[[str | None], httpx.Client] | None = None, part_skill: str = "retrieve"):
    from src.a2a.server import create_app

    return create_app(memory=None, search_factory=delegating_search(agent_url, http_factory, part_skill),
                      base_url=base_url, card=coordinator_card(base_url, agent_url), plain_skills=("delegate",))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", default=os.environ.get("A2A_AGENT_URL", "http://localhost:8300"),
                    help="the FiQA agent to delegate to")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8310)
    a = ap.parse_args(argv)
    import uvicorn

    uvicorn.run(create_coordinator_app(a.agent, f"http://{a.host}:{a.port}"), host=a.host, port=a.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
