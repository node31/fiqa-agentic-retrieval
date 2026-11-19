"""Deterministic "LLM" policies for the fake backend (tests, toys, --dry-run cost estimates).

They mimic a sensible model with simple string heuristics, so the whole agent
pipeline can run end to end for free and reproducibly. They are NOT meant to be
smart. Real numbers come from the copilot/azure backends.
"""
from __future__ import annotations

import json
import re

_SPLIT = re.compile(r"\?\s+|;\s+|\.\s+(?=[A-Z])|,?\s+\b(?:and also|also|and)\b\s+", re.I)


def split_question(q: str) -> list[str]:
    """Naive decomposition: split on '?', ';', sentence breaks and 'and/also'; keep parts with >= 3 words."""
    parts = [p.strip(" ?.,") for p in _SPLIT.split(q) if p and len(p.split()) >= 3]
    return parts if len(parts) >= 2 else [q]


def _field(user: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.*)$", user, re.M)
    return m.group(1).strip() if m else ""


def heuristic_responder(system: str, user: str) -> str:
    role = system.split("\n", 1)[0].replace("ROLE:", "").strip()
    question = _field(user, "Question")
    if role == "planner":
        parts = split_question(question)
        return json.dumps({"simple": len(parts) == 1, "subqueries": parts[:3] if len(parts) > 1 else []})
    if role == "verifier":
        n = len(re.findall(r"^Sub-question \d+:", user, re.M))
        return json.dumps({"checks": [{"sub": i, "relevant": True, "rewrite": ""} for i in range(n)]})
    if role == "single_agent":
        parts = split_question(question)
        n_searches = len(re.findall(r"^Search \d+", user, re.M))
        if len(parts) > 1 and n_searches < len(parts) and "LAST action" not in system:
            return json.dumps({"action": "search", "query": parts[n_searches], "strategy": "hybrid"})
        return json.dumps({"action": "finish", "doc_ids": []})  # [] = keep the fused ranking
    raise ValueError(f"fake LLM has no policy for role {role!r}")
