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

JARGON
    agent       an LLM in a loop: look at results -> pick an action -> we run it -> repeat.
    step        one iteration of that loop = one LLM call. The "step cap" (max_steps) is
                the hard limit on iterations, so the agent cannot loop forever.
    hallucinated id   a doc id the model made up (or misremembered). It looks valid but is
                not in any result we showed, so it is dropped.
    RRF         reciprocal rank fusion: merge ranked lists by summing 1/(60 + rank) per doc
                (see src/retrieval/hybrid.py).

HOW TO READ ``run_single_agent``
    1. Step 0: search the raw question (free; also the fallback if every LLM call fails).
    2. Loop: build the prompt from ALL searches so far, call the LLM, parse its JSON action.
       "search" -> run it and loop again; anything else -> treat as finish and stop.
    3. Keep only chosen ids we actually showed, then fill up to k from the fused ranking.
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
    """Outcome of one agent run (single or multi): the final ids plus everything we measure.

    ``path`` says which pipeline produced it ("single" / "multi"); ``llm_calls`` and the token
    counts come from the shared Budget; ``trace`` is a step-by-step log of the agent's
    decisions (for debugging and the docs); ``errors`` lists failures we recovered from.
    """

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
    """RRF-merge several hit lists (one per search) into one ranked list of doc ids."""
    return [d for d, _ in rrf([[(h["doc_id"], h["score"]) for h in lst] for lst in lists], top=k)]


def run_single_agent(question: str, llm: LLM, tool: SearchTool, max_steps: int = 3,
                     budget: Budget | None = None, k: int = 10) -> AgentResult:
    """Run the search-then-decide loop described in the module docstring.

    The step-0 search results are always part of the fill, so a failing LLM never empties the result.
    PITFALL: exceptions from the search tool itself are NOT caught here; they reach the caller.
    """
    t0 = time.perf_counter()
    # WHY max_calls=max_steps: one LLM call per step, so the budget enforces the step cap too.
    budget = budget or Budget(max_calls=max_steps, max_tokens=12_000)
    res = AgentResult([], path="single")
    # Every search is kept as (strategy, query, hits); all of them are shown to the model.
    searches = [("hybrid", question, tool(question, "hybrid"))]  # step 0: ground in real results
    chosen: list[str] = []
    for step in range(1, max_steps + 1):
        remaining = max_steps - step  # actions left AFTER this one; 0 means "you must finish"
        # The prompt counts THIS action too, hence remaining + 1.
        system = prompts.SINGLE_AGENT.replace("{remaining}", str(remaining + 1))
        if remaining == 0:
            system += "\nThis is your LAST action: you must reply with finish."
        # The user prompt is rebuilt from scratch each step: the model has no memory between
        # calls, so everything it should know (question + every search so far) goes in again.
        user = f"Question: {question}\n\n" + "\n\n".join(
            f"Search {i} ({s}): {q!r}\n{render_hits(h)}" for i, (s, q, h) in enumerate(searches, 1))
        try:
            out = llm.complete(system, user, max_output_tokens=250, budget=budget)
            action = parse_json(out.text)
        except (BudgetExceeded, ValueError, RuntimeError, TimeoutError) as e:
            # Degrade: stop asking the model and use the searches we already have.
            res.errors.append(f"step {step}: {e!r}"[:300])
            break
        res.trace.append({"step": step, "action": action})
        # The "tool call": the model only DESCRIBED a search; we run it. A search request with no
        # query, or on the last step, falls through and is treated as finish.
        # (SearchTool itself replaces an invalid strategy name with "hybrid".)
        if action.get("action") == "search" and remaining > 0 and action.get("query"):
            strat = action.get("strategy", "hybrid")
            searches.append((strat, action["query"], tool(action["query"], strat)))
            continue
        # Finish: validate the model's picks. dict.fromkeys dedupes while keeping order, str()
        # normalises numeric ids, and "d in seen" drops hallucinated ids.
        seen = {h["doc_id"] for _, _, hits in searches for h in hits}
        chosen = [d for d in dict.fromkeys(map(str, action.get("doc_ids") or [])) if d in seen]
        break
    # Fill: the model's picks first, then the RRF-fused ranking of every search (scores are
    # ignored by RRF; only ranks matter). k * 3 is generous headroom: k fused ids alone would
    # already guarantee k results after dedupe.
    fill = fuse_lists([h for _, _, h in searches], k=k * 3)
    res.doc_ids = list(dict.fromkeys(chosen + fill))[:k]
    # Usage numbers come from the budget, which saw every LLM call of this run.
    res.llm_calls, res.input_tokens, res.output_tokens = budget.calls, budget.input_tokens, budget.output_tokens
    res.searches, res.latency_s = len(searches), time.perf_counter() - t0
    return res
