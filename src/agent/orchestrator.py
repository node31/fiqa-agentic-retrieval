"""Multi-agent orchestrator (Part C3): planner -> parallel retrieval workers -> verifier.

ROLES (docs/13_multi_agent_orchestration.md, ADR 0009)
    Orchestrator  plain code, NOT an LLM. Owns the budget, the timeouts and the
                  final 10 slots. Control flow you can unit-test is cheaper and
                  more reliable than asking a model to orchestrate.
    Planner       LLM. Decides "simple vs compound" and writes standalone sub-queries.
                  Skipped entirely by a free heuristic gate for short, single-clause
                  questions ("don't over-orchestrate").
    Workers       one per sub-query, run IN PARALLEL threads, each with a timeout.
                  They call the search tool (no LLM needed, because the planner already
                  wrote a good query). Parallelism makes latency ~max, not ~sum.
    Verifier      LLM, one batched call for ALL sub-queries. It looks at the top-5
                  snippets per sub-query and may request ONE re-search with a better
                  query. That fixes the main failure mode: a sub-query that drifted.

SHARING THE 10 SLOTS (ADR 0010, examples/25_slot_allocation.py)
    Round-robin interleave across sub-query lists (A1, B1, A2, B2, ...), dedupe,
    then fill from the original-question search. With 2 sub-queries each half gets
    ~5 slots. That guarantees Coverage@10 whenever each worker found anything,
    at the price of dropping each half's ranks 6-10. A score-sorted merge would let
    one "easy" half take all 10 slots.

GUARDRAILS
    * One Budget per user query, shared by planner + verifier: call cap and token cap.
    * Budget exhausted -> skip the remaining LLM steps. We still return results.
    * Worker failure or timeout -> drop that list and record it in ``errors``. If ALL
      fail, fall back to searching the original question. Planner or verifier error ->
      act as if it said "simple" / "all relevant".

JARGON
    compound question   asks for 2-3 different things at once ("how does X work, and
                        should I do Y?"). One search tends to cover only one part.
    decomposition       splitting a compound question into standalone sub-queries.
    coverage            did the top 10 include something relevant for EVERY part of the
                        question (Coverage@10), not just for the easiest part.
    failure-isolated    one worker crashing or timing out loses only its own list.

THE LIFE OF ONE ``Orchestrator.run`` (numbered like the comments in ``run``)
    1. plan      gate or planner LLM -> ["sub-query A", "sub-query B"] (or [question]).
    2. workers   search every sub-query, plus the full question as a slot filler, in parallel.
    3. verify    one LLM call over all sub-queries' top-5; re-search the ones it rewrote.
    4. allocate  interleave(sub-query lists) + full-question list, dedupe, cut to k.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, wait

from src.agent import prompts
from src.agent.llm import LLM, Budget, BudgetExceeded, parse_json
from src.agent.single_agent import AgentResult
from src.agent.tools import SearchTool, render_hits

# Cheap signals that a question MIGHT be compound: a conjunction word, a ";" or ",", or two
# question marks ("\?.+\?"). WHY so trigger-happy: a false "compound" only costs one planner
# call (which can still answer "simple"); a false "simple" skips decomposition entirely.
_COMPOUND_HINT = re.compile(r"\b(and|also|plus|as well as|additionally|then|while)\b|[;,]|\?.+\?", re.I)


def looks_simple(question: str, max_words: int = 12) -> bool:
    """Free gate: short and no conjunction/second question -> skip the planner LLM call.

    >>> looks_simple("What is short selling?")
    True
    >>> looks_simple("How does short selling work, and should I pay off my mortgage early?")
    False
    """
    return len(question.split()) <= max_words and not _COMPOUND_HINT.search(question)


def plan(question: str, llm: LLM, budget: Budget) -> tuple[list[str], str]:
    """Return (sub-queries, how-decided). Always returns at least [question].

    The planner is the "decomposition" role. ``how-decided`` goes into the trace so an eval
    can count how often the free gate vs the LLM made the call.
    """
    # Free gate first: most plain questions never pay for a planner call.
    if looks_simple(question):
        return [question], "gate:simple"
    out = llm.complete(prompts.PLANNER, f"Question: {question}", max_output_tokens=200, budget=budget)
    p = parse_json(out.text)
    # Validate the model's output: keep only non-empty strings, and at most 3 sub-queries
    # (each one costs a search and a share of only 10 slots).
    subs = [s.strip() for s in p.get("subqueries") or [] if isinstance(s, str) and s.strip()][:3]
    # Fewer than 2 usable sub-queries = nothing to split; searching the original wording is safer.
    if p.get("simple") or len(subs) < 2:
        return [question], "planner:simple"
    return subs, "planner:decomposed"


def verify(question: str, subs: list[str], lists: dict[int, list[dict]], llm: LLM, budget: Budget) -> dict[int, str]:
    """Return {sub_index: better_query} for sub-queries whose results look off-topic.

    ``lists`` maps sub-query index -> its hits (only workers that succeeded are present).
    """
    # One batched prompt: every sub-question with its top-5 snippets. WHY top-5 and one call:
    # enough to judge "on-topic or not" while keeping the verifier to a single, short LLM call.
    blocks = [f"Sub-question {i}: {subs[i]}\n{render_hits(lists[i][:5])}" for i in sorted(lists)]
    out = llm.complete(prompts.VERIFIER, f"Question: {question}\n\n" + "\n\n".join(blocks),
                       max_output_tokens=250, budget=budget)
    checks = parse_json(out.text).get("checks", [])
    # Keep a rewrite only if: the model said relevant=false (missing "relevant" counts as
    # relevant, i.e. "do nothing"), it gave a non-empty rewrite, and the index refers to a
    # sub-query we actually showed it (never trust an index produced by an LLM).
    return {int(c["sub"]): c["rewrite"] for c in checks
            if not c.get("relevant", True) and c.get("rewrite") and int(c.get("sub", -1)) in lists}


def interleave(lists: list[list[str]], k: int = 10) -> list[str]:
    """Round-robin slot allocation with de-duplication.

    >>> interleave([["a1", "a2", "a3"], ["b1", "a1", "b2"]], k=4)
    ['a1', 'b1', 'a2', 'a3']

    Take rank 1 of every list, then rank 2 of every list, and so on, skipping ids already
    taken. In the example "a1" appears in both lists, so B's rank-2 turn is skipped.
    """
    out: list[str] = []
    seen: set[str] = set()
    # Outer loop = rank (row), inner loop = list (column): that order IS the round robin.
    for rank in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank] not in seen:
                seen.add(lst[rank])
                out.append(lst[rank])
                if len(out) == k:
                    return out
    return out


class Orchestrator:
    """Plain-code coordinator of planner -> workers -> verifier. Create once, call ``run`` per query.

    ``max_llm_calls=3`` is enough for planner + verifier (2 calls) with one spare; the thread
    pool is created once and reused across queries.
    """

    def __init__(self, llm: LLM, tool: SearchTool, max_llm_calls: int = 3, max_tokens: int = 8_000,
                 worker_timeout_s: float = 10.0, verify: bool = True, k: int = 10):
        self.llm, self.tool, self.k = llm, tool, k
        self.max_llm_calls, self.max_tokens = max_llm_calls, max_tokens
        # Stored as use_verifier: the name ``verify`` already belongs to the module-level
        # verify() function that run() calls.
        self.worker_timeout_s, self.use_verifier = worker_timeout_s, verify
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="worker")

    def _run_workers(self, queries: dict[int, str], res: AgentResult) -> dict[int, list[dict]]:
        """Search every {index: query} in parallel; return {index: hits} for the ones that succeeded.

        Failures and timeouts are written to ``res.errors`` instead of being raised.
        """
        # WHY threads: a search is mostly waiting (HTTP or I/O), so threads overlap well and
        # total latency is ~the slowest search, not the sum of all of them.
        futs = {i: self.pool.submit(self.tool, q, "hybrid") for i, q in queries.items()}
        # One shared deadline for the whole batch; after it we use whatever has finished.
        wait(futs.values(), timeout=self.worker_timeout_s)
        out = {}
        for i, f in futs.items():
            if not f.done():
                # PITFALL: cancel() cannot stop a search that is already running in a thread;
                # it only prevents one that has not started. We simply stop waiting for it.
                f.cancel()
                res.errors.append(f"worker {i} timed out after {self.worker_timeout_s}s")
            elif f.exception() is not None:
                res.errors.append(f"worker {i} failed: {f.exception()!r}"[:300])
            else:
                out[i] = f.result()
        # Counts every search we STARTED, including ones that failed or timed out.
        res.searches += len(queries)
        return out

    def run(self, question: str) -> AgentResult:
        """Answer one user question with up to k doc ids, degrading (see GUARDRAILS) instead of raising."""
        t0 = time.perf_counter()
        # A fresh budget per user query, shared by the planner and the verifier.
        budget = Budget(self.max_llm_calls, self.max_tokens)
        res = AgentResult([], path="multi")
        try:  # 1. plan
            subs, how = plan(question, self.llm, budget)
        except (BudgetExceeded, ValueError, RuntimeError, TimeoutError) as e:
            # Degrade to "simple": search the question as-is.
            subs, how = [question], "planner:failed"
            res.errors.append(f"planner: {e!r}"[:300])
        res.trace.append({"plan": subs, "decided_by": how})
        # 2. parallel workers. For decomposed questions, also search the full question (slot filler).
        # Indexes 0..len(subs)-1 are the sub-queries; index len(subs) is the full question.
        queries = dict(enumerate(subs))
        if len(subs) > 1:
            queries[len(subs)] = question
        lists = self._run_workers(queries, res)
        # 3. verify (only when decomposed: for a simple question the planner/gate already decided)
        # ... and only if at least one SUB-query worker succeeded (nothing to check otherwise).
        if self.use_verifier and len(subs) > 1 and any(i in lists for i in range(len(subs))):
            try:
                # Show the verifier only the sub-query lists (not the full-question filler).
                fixes = verify(question, subs, {i: lists[i] for i in range(len(subs)) if i in lists},
                               self.llm, budget)
                res.trace.append({"verifier_rewrites": fixes})
                # Re-search the rewritten sub-queries; a successful re-search REPLACES that
                # sub-query's list, a failed one leaves the original list in place.
                if fixes:
                    lists.update(self._run_workers(fixes, res))
            except (BudgetExceeded, ValueError, RuntimeError, TimeoutError, KeyError, TypeError) as e:
                # Degrade to "all relevant": keep the lists we have.
                res.errors.append(f"verifier: {e!r}"[:300])
        # 4. allocate the 10 slots
        # Sub-query lists in sub-query order (failed workers are simply absent).
        sub_lists = [[h["doc_id"] for h in lists[i]] for i in range(len(subs)) if i in lists]
        # The full-question list only fills slots left over after the interleave.
        tail = [h["doc_id"] for h in lists.get(len(subs), [])] if len(subs) > 1 else []
        if not sub_lists and not tail:  # every worker failed -> last-resort direct search
            try:
                tail = [h["doc_id"] for h in self.tool(question, "hybrid")]
            except Exception as e:  # noqa: BLE001 - report, never raise to the caller
                res.errors.append(f"fallback search failed: {e!r}"[:300])
        # WHY interleave, not sort by score: scores from different queries are not comparable,
        # and a score sort lets one "easy" sub-query take all 10 slots (coverage drops to one
        # part). dict.fromkeys dedupes ids that are in both the interleave and the tail.
        res.doc_ids = list(dict.fromkeys(interleave(sub_lists, self.k) + tail))[: self.k]
        res.llm_calls, res.input_tokens, res.output_tokens = budget.calls, budget.input_tokens, budget.output_tokens
        res.latency_s = time.perf_counter() - t0
        return res
