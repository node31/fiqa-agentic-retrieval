"""SOLUTION for ch08: an agent loop with guardrails, driven by a scripted fake LLM. Standard library only.

Try the exercise yourself first (exercises/ch08_agents/agent_loop.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch08_agents [--full]
"""
from __future__ import annotations

import json
from typing import Callable

SearchFn = Callable[[str], list[str]]
LLMFn = Callable[[str], str]


class ScriptedLLM:
    """A fake LLM: returns the given replies in order (then repeats the last one). Counts calls."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls, self.prompts = list(replies), 0, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def parse_action(text: str) -> dict:
    """The first JSON object in `text`. Real LLMs wrap JSON in chatter and ```json fences, so we scan for it."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:                       # inside "...": braces don't count
                if escaped:
                    escaped = False             # this char was escaped (e.g. \" ), skip it
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:                  # found the matching close brace
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break                   # e.g. {'single': 'quotes'}: try the next "{"
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in reply: {text[:80]!r}")


def interleave(lists: list[list[str]], k: int = 10) -> list[str]:
    """Round-robin across lists, skipping duplicates. WHY: guarantees every sub-question gets slots
    (a sort-by-score merge lets the sub-question with higher raw scores take all 10)."""
    out: list[str] = []
    seen: set[str] = set()
    longest = max((len(lst) for lst in lists), default=0)
    for rank in range(longest):
        for lst in lists:
            if rank < len(lst) and lst[rank] not in seen:
                seen.add(lst[rank])
                out.append(lst[rank])
                if len(out) == k:
                    return out
    return out


def _prompt(question: str, results: list[tuple[str, list[str]]]) -> str:
    lines = [f"Question: {question}", "Results so far:"]
    for q, ids in results:
        lines.append(f"- search {q!r} -> {ids}")
    lines.append('Reply with ONE JSON object: {"action": "search", "query": "..."} '
                 'or {"action": "finish", "doc_ids": [...]}')
    return "\n".join(lines)


def run_agent(question: str, llm: LLMFn, search_fn: SearchFn, k: int = 10, max_steps: int = 3) -> dict:
    """The 7-point contract from README.md, step by step."""
    # (1) step 0 is free: search the raw question first. WHY: the agent ALWAYS has a sensible fallback answer,
    #     and the LLM's first look already contains evidence (saves one call).
    results: list[tuple[str, list[str]]] = [(question, search_fn(question))]
    errors: list[str] = []
    chosen: list[str] = []
    llm_calls = 0

    for step in range(1, max_steps + 1):          # (2) hard cap on LLM calls: cost and latency are bounded
        reply = llm(_prompt(question, results))
        llm_calls += 1
        try:
            action = parse_action(reply)
        except ValueError as e:                   # (5) bad output is recorded, never raised
            errors.append(f"step {step}: {e}")
            break
        kind = action.get("action")
        if kind == "search" and step < max_steps and action.get("query"):
            results.append((action["query"], search_fn(action["query"])))   # (3) search and loop again
            continue
        if kind not in ("search", "finish"):
            errors.append(f"step {step}: unknown action {kind!r}")
            break
        # (3) last step: even a "search" is treated as finish.  (4) keep only ids we actually showed it
        seen = {d for _, ids in results for d in ids}
        chosen = [d for d in dict.fromkeys(action.get("doc_ids") or []) if d in seen]   # drops invented ids
        break

    # (6) fill the remaining slots from all searches, round-robin
    filler = interleave([ids for _, ids in results], k=k + len(chosen))
    doc_ids = list(dict.fromkeys(chosen + filler))[:k]
    return {"doc_ids": doc_ids, "llm_calls": llm_calls, "searches": len(results), "errors": errors}   # (7)


if __name__ == "__main__":
    INDEX = {"short": ["d1", "d10", "d2"], "mortgage": ["d3", "d9", "d8"], "borrow": ["d10", "d1"]}

    def search(q: str) -> list[str]:
        out = []
        for word, ids in INDEX.items():
            if word in q.lower():
                out += [i for i in ids if i not in out]
        return out

    llm = ScriptedLLM(['I will search.\n```json\n{"action": "search", "query": "mortgage early payoff"}\n```',
                       '{"action": "finish", "doc_ids": ["d3", "d1", "INVENTED"]}'])
    res = run_agent("How does shorting work, and should I pay off my mortgage early?", llm, search)
    print(res)   # doc_ids starts d3, d1 (INVENTED dropped), then fills from both searches
    for i, p in enumerate(llm.prompts, 1):
        print(f"--- prompt {i} ({len(p)} chars) ---\n{p}\n")
