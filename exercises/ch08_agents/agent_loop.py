"""ch08: an agent loop with guardrails, driven by a scripted fake LLM. Standard library only.

Run the checker:   python -m pytest exercises/ch08_agents
Run the demo:      python exercises/ch08_agents/agent_loop.py
"""
from __future__ import annotations

import json  # noqa: F401
from typing import Callable

SearchFn = Callable[[str], list[str]]   # query -> ranked doc ids
LLMFn = Callable[[str], str]            # prompt -> reply text


class ScriptedLLM:
    """A fake LLM: returns the given replies in order (then repeats the last one). Counts calls. (Already done.)"""

    def __init__(self, replies: list[str]):
        self.replies, self.calls, self.prompts = list(replies), 0, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def parse_action(text: str) -> dict:
    """Return the FIRST JSON object in `text` as a dict, tolerating chatter and ```json fences.
    Raise ValueError if there is no valid JSON object.

    parse_action('Sure!\\n```json\\n{"action": "finish", "doc_ids": []}\\n```') == {"action": "finish", "doc_ids": []}
    Hint: find the first "{", then scan forward counting { and } (ignore braces inside strings) to find its end.
    """
    raise NotImplementedError("TODO (ch08): implement parse_action")


def interleave(lists: list[list[str]], k: int = 10) -> list[str]:
    """Round-robin: rank 1 of every list, then rank 2 of every list, ... skipping duplicates, until k items.

    interleave([["a1", "a2", "a3"], ["b1", "a1", "b2"]], k=4) == ["a1", "b1", "a2", "a3"]
    """
    raise NotImplementedError("TODO (ch08): implement interleave")


def run_agent(question: str, llm: LLMFn, search_fn: SearchFn, k: int = 10, max_steps: int = 3) -> dict:
    """The agent loop. Follow the 7-point contract in README.md exactly.

    Returns {"doc_ids": [...<= k], "llm_calls": int, "searches": int, "errors": [str, ...]}
    """
    raise NotImplementedError("TODO (ch08): implement run_agent (see README contract)")


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
    print(res)
    for i, p in enumerate(llm.prompts, 1):
        print(f"--- prompt {i} ({len(p)} chars) ---\n{p}\n")
