"""Deterministic "LLM" policies for the fake backend (tests, toys, --dry-run cost estimates).

They mimic a sensible model with simple string heuristics, so the whole agent
pipeline can run end to end for free and reproducibly. They are NOT meant to be
smart. Real numbers come from the copilot/azure backends.

HOW IT WORKS
    ``heuristic_responder(system, user)`` is plugged into ``FakeLLM``. It reads the
    "ROLE: ..." first line of the system prompt to know which agent is asking, then
    answers with the same JSON a real model would send for that role:
        planner       split the question with a regex; >= 2 parts -> decomposed.
        verifier      always "relevant" (so no re-searches happen).
        single_agent  search each remaining part once, then finish with [] (keep the
                      fused ranking). It never invents ids.
    Because the output depends only on the input text, tests are reproducible.
"""
from __future__ import annotations

import json
import re

# Split points for a naive decomposition: "? " / "; " / ". " before a capital letter (a new
# sentence) / "and", "also", "and also" (with an optional comma before). All groups are
# non-capturing (?:...) so re.split does not insert the separators into the result.
_SPLIT = re.compile(r"\?\s+|;\s+|\.\s+(?=[A-Z])|,?\s+\b(?:and also|also|and)\b\s+", re.I)


def split_question(q: str) -> list[str]:
    """Naive decomposition: split on '?', ';', sentence breaks and 'and/also'; keep parts with >= 3 words."""
    # WHY >= 3 words: drops fragments like "stocks and bonds" -> "bonds" that are not real needs.
    parts = [p.strip(" ?.,") for p in _SPLIT.split(q) if p and len(p.split()) >= 3]
    return parts if len(parts) >= 2 else [q]


def _field(user: str, name: str) -> str:
    """Value of the first line ``<name>: value`` in the user prompt (e.g. "Question: ...")."""
    m = re.search(rf"^{name}:\s*(.*)$", user, re.M)
    return m.group(1).strip() if m else ""


def heuristic_responder(system: str, user: str) -> str:
    """Fake model: dispatch on the prompt's ROLE line and reply with that role's JSON."""
    role = system.split("\n", 1)[0].replace("ROLE:", "").strip()
    question = _field(user, "Question")
    if role == "planner":
        parts = split_question(question)
        return json.dumps({"simple": len(parts) == 1, "subqueries": parts[:3] if len(parts) > 1 else []})
    if role == "verifier":
        # One "relevant" check per "Sub-question N:" block the orchestrator sent.
        n = len(re.findall(r"^Sub-question \d+:", user, re.M))
        return json.dumps({"checks": [{"sub": i, "relevant": True, "rewrite": ""} for i in range(n)]})
    if role == "single_agent":
        parts = split_question(question)
        # Count searches done so far ("Search 1", "Search 2", ... headers in the user prompt).
        # Search 1 is the full question (step 0), so the next part to search is parts[n_searches];
        # parts[0] is never searched alone because the full question already covers it.
        n_searches = len(re.findall(r"^Search \d+", user, re.M))
        if len(parts) > 1 and n_searches < len(parts) and "LAST action" not in system:
            return json.dumps({"action": "search", "query": parts[n_searches], "strategy": "hybrid"})
        if '"confidence"' in system:
            # The cascade's cheap tier: pick the first 5 ids shown; confident only for one-topic questions,
            # so tests see both outcomes (kept, escalated).
            ids = re.findall(r"^\[([^\]]+)\]", user, re.M)[:5]
            return json.dumps({"action": "finish", "doc_ids": ids, "confidence": "high" if len(parts) == 1 else "low"})
        return json.dumps({"action": "finish", "doc_ids": []})  # [] = keep the fused ranking
    # Fail loudly: a new prompt role needs a matching fake policy.
    raise ValueError(f"fake LLM has no policy for role {role!r}")
