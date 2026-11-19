"""Single agent (Part C2): one LLM, one tool, at most 3 steps, then a final top-10.

THE LOOP (docs/12_single_agent.md has a worked trace)

    results_1 = search(question, "hybrid")          # step 0: free, no LLM, grounds the agent
    for step in 1..3:                                # each step = exactly one LLM call
        action = LLM(system, question + all results so far)
        if action is search and steps remain:  results_n = search(action.query, action.strategy)
        else: break                                  # finish (or forced finish at step 3)
    final = LLM's chosen ids (validated)  +  RRF fill over every result list, top 10

WHY THESE CHOICES
    * Search first, think second: the first LLM call sees real results, so a simple
      question can finish after ONE call instead of spending a call on "search(question)".
    * Validated output: ids the model invents, or ids it never saw, are dropped. A model
      that answers with garbage degrades to the fused ranking, never to an empty list.
    * RRF fill: guarantees 10 results and blends multiple searches fairly. With two
      searches (one per topic) RRF interleaves them, which is what coverage needs.
    * Errors (bad JSON, budget, timeouts) end the loop and fall back to what we have.
      An agent must degrade, never crash the request.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.agent import prompts
from src.agent.llm import LLM, Budget, BudgetExceeded, cost_usd, parse_json
from src.agent.tools import SearchTool, render_hits
from src.retrieval.hybrid import rrf


@dataclass
class AgentResult:
    doc_ids: list[str]
    path: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens)


def fuse_lists(lists: list[list[dict]], k: int = 10) -> list[str]:
    return [d for d, _ in rrf([[(h["doc_id"], h["score"]) for h in lst] for lst in lists], top=k)]


def run_single_agent(question: str, llm: LLM, tool: SearchTool, max_steps: int = 3,
                     budget: Budget | None = None, k: int = 10) -> AgentResult:
    t0 = time.perf_counter()
    budget = budget or Budget(max_calls=max_steps, max_tokens=12_000)
    res = AgentResult([], path="single")
    searches = [("hybrid", question, tool(question, "hybrid"))]  # step 0: ground in real results
    chosen: list[str] = []
    for step in range(1, max_steps + 1):
        remaining = max_steps - step  # actions left AFTER this one; 0 means "you must finish"
        system = prompts.SINGLE_AGENT.replace("{remaining}", str(remaining + 1))
        if remaining == 0:
            system += "\nThis is your LAST action: you must reply with finish."
        user = f"Question: {question}\n\n" + "\n\n".join(
            f"Search {i} ({s}): {q!r}\n{render_hits(h)}" for i, (s, q, h) in enumerate(searches, 1))
        try:
            out = llm.complete(system, user, max_output_tokens=250, budget=budget)
            action = parse_json(out.text)
        except (BudgetExceeded, ValueError, RuntimeError, TimeoutError) as e:
            res.errors.append(f"step {step}: {e!r}"[:300])
            break
        res.trace.append({"step": step, "action": action})
        if action.get("action") == "search" and remaining > 0 and action.get("query"):
            strat = action.get("strategy", "hybrid")
            searches.append((strat, action["query"], tool(action["query"], strat)))
            continue
        seen = {h["doc_id"] for _, _, hits in searches for h in hits}
        chosen = [d for d in dict.fromkeys(map(str, action.get("doc_ids") or [])) if d in seen]
        break
    fill = fuse_lists([h for _, _, h in searches], k=k * 3)
    res.doc_ids = list(dict.fromkeys(chosen + fill))[:k]
    res.llm_calls, res.input_tokens, res.output_tokens = budget.calls, budget.input_tokens, budget.output_tokens
    res.searches, res.latency_s = len(searches), time.perf_counter() - t0
    return res
