r"""Multi-agent orchestrator (Part C3): planner -> parallel retrieval workers -> verifier.

ROLES (docs/topics/13_multi_agent_orchestration.md)
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

SHARING THE 10 SLOTS (docs/topics/13_multi_agent_orchestration.md, examples/25_slot_allocation.py)
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

WORKED EXAMPLE (real runs from the Azure gpt-6-sol eval, results/agent_eval.json)

    Both LLM prompts below are EXACTLY what the model received: I rebuilt each one with this
    code and found its sha256 in .llm_cache/azure/ (the cache key is a hash of the full prompt,
    see src/agent/llm.py). The searches were re-run against the router, and the final 10 ids
    match the doc_ids recorded for the question. A "*" after a doc id marks a gold passage.

    --- Three ways step 1 can go -------------------------------------------------------------

    a. Free gate, 0 LLM calls (plain question 7391):
           "How to trade large number of shares?"
       7 words (limit 12) and no "and" / "also" / "," / ";" / second "?", so looks_simple() is
       True. subs = [question]; one worker searches it; no verifier. 56 of the 100 plain
       questions took this path.

    b. Planner says "simple", 1 LLM call (plain question 6617):
           "(How) can I print my own checks on my printer on regular paper?"
       13 words, so the gate can't skip it (it only skips 12 words or fewer). The planner got
       "Question: <text>" and replied {"simple":true,"subqueries":[]}, so subs = [question]
       and the rest is the same as (a). 37 of the 100 plain questions took this path.

    c. Planner decomposes, 2 LLM calls. Every one of the 100 compound questions went this
       way (and 7 plain ones). The rest of this example follows one of them.

    --- The compound question: c012 ----------------------------------------------------------

    "How does owning a home and paying a mortgage fit into family savings and investment, and how is Return on Capital computed in Joel Greenblatt's Magic Formula?"
        half a: "How does owning a home and paying on a mortgage fit into family savings and investment?"
        half b: "Trying to understand Return on Capital (Joel Greenblatt's Magic Formula version)"
        gold passages: 5657, 386128, 429023, 214798, 206794, 64456 (half a)
                       and 559363, 557607 (half b)

    --- 1. plan ------------------------------------------------------------------------------

    looks_simple(question) is False: 27 words, and it contains "and" and ",".
    So plan() calls the planner. System prompt: prompts.PLANNER. User prompt, exactly:

        Question: How does owning a home and paying a mortgage fit into family savings and investment, and how is Return on Capital computed in Joel Greenblatt's Magic Formula?

    gpt-6-sol replied (162 input tokens, 53 output tokens):

        {"simple":false,"subqueries":["How do homeownership and mortgage payments fit into family savings and investment?","How is Return on Capital calculated in Joel Greenblatt's Magic Formula?"]}

    plan() keeps non-empty strings (at most 3). Two survive, so
        subs = ["How do homeownership and mortgage payments fit into family savings and investment?", "How is Return on Capital calculated in Joel Greenblatt's Magic Formula?"]
    and the trace records decided_by = "planner:decomposed".

    --- 2. workers (parallel) ----------------------------------------------------------------

    queries = {0: sub A, 1: sub B, 2: the full question}. Index 2 is the slot filler: it is
    only used if the interleave in step 4 leaves slots empty. All three go to the thread pool
    at once, so this step takes about as long as the slowest search, not the sum of three.
    Each returns 10 hits (hybrid, 200-char snippets):

        0 (sub A):  402950, 322157, 184826, 64456*, 249628, 255414, 100683, 90009, 82986, 172722
        1 (sub B):  83987, 109148, 203091, 489706, 438547, 143302, 443689, 473963, 95278, 220025
        2 (full q): 505172, 184826, 332846, 402950, 64456*, 393591, 322157, 386128*, 107701, 372615

    Sub B's list has no gold. Its top 5 are about computing returns in general (cap rate,
    a total-return index, a monthly savings plan); none mentions Greenblatt or the Magic Formula.

    --- 3. verify ----------------------------------------------------------------------------

    At least one sub-query worker succeeded, so verify() runs. It shows the model the TOP 5
    of each sub-query list (not the full-question list), all in one call. System prompt:
    prompts.VERIFIER. User prompt, exactly (2448 characters):

        Question: How does owning a home and paying a mortgage fit into family savings and investment, and how is Return on Capital computed in Joel Greenblatt's Magic Formula?

        Sub-question 0: How do homeownership and mortgage payments fit into family savings and investment?
        [402950] "Math says invest in the Market (But paying off your mortgage early is a valid option if you are very risk averse.) You are going to get a better return by investing in the stock market. In the US in
        [322157] "There are a number of reasons I'm in agreement with ""A house that is worth $300,000, or $50,000 of equity in a house and $225,000 in the bank."" So, the update to the first comment should be ""A pa
        [184826] "Home ownership is thought to be a lifetime investment and is touted as a big part of the ""American Dream."" That's changing now, as a lot of younger people have seen their parents' lifetime investm
        [64456] 1) How does owning a home fit into my financial portfolio? Most seem to agree that at best it is a hedge against rent or dollar inflation, and at worst it should be viewed as a liability, and has no p
        [249628] "The homeowner gets all the profit from the price rise, because it's their asset. The bank will charge early repayment fees, but these are often a small fraction of the profits. This is why homeowners

        Sub-question 1: How is Return on Capital calculated in Joel Greenblatt's Magic Formula?
        [83987] Thanks for showing me that. I can see it now. I have always used my formula, and even a senior at another company confirmed the way I calculated the returns. Luckily, I do not work with that manager
        [109148] "Capitalization rate and ""Net Profit margin"" are two different things. In Capitalization rate note that we are taking the ""total value"" in the denominator and in Net profit margin we are taking ""
        [203091] With 10% return over three years, depositing $900 each month, in three years $34,039.30. Re. downvote. I guess this is too brief and without explanation, but I was rushing. If you want further expla
        [489706] The point of a total return index is that it already has accounted for the capital gains + coupon income. If you want to calculate it yourself you'll have to find the on-the-run 10y bond for each dist
        [438547] There are many ways to calculate the return, and every way will give you a different results in terms of a percentage-value. One way to always get something meaningful - count the cash. You had 977 (+

    gpt-6-sol replied (677 input tokens, 70 output tokens):

        {"checks":[{"sub":0,"relevant":true,"rewrite":""},{"sub":1,"relevant":false,"rewrite":"Joel Greenblatt Magic Formula return on capital formula EBIT divided by net working capital plus net fixed assets"}]}

    verify() keeps a check only if relevant is false, the rewrite is non-empty and "sub" is an
    index it actually showed. Result: {1: "<the rewrite above>"}. _run_workers re-searches
    it, and the new list REPLACES list 1:

        1 (rewrite): 109148, 557607*, 83987, 489706, 523303, 563169, 473963, 204917, 438547, 69506

    The re-search found gold passage 557607 at rank 2. It was not in the first list for sub B.

    --- 4. allocate the 10 slots -------------------------------------------------------------

    sub_lists = [list 0, new list 1]. interleave() takes rank 1 of A, rank 1 of B, rank 2 of
    A, rank 2 of B, and so on, skipping ids it already took (none repeat here):

        rank 1:  A -> 402950    B -> 109148 
        rank 2:  A -> 322157    B -> 557607*
        rank 3:  A -> 184826    B ->  83987 
        rank 4:  A ->  64456*   B -> 489706 
        rank 5:  A -> 249628    B -> 523303 

    That fills 10 slots after rank 5, so ranks 6 to 10 of both lists are dropped:
        interleave = ['402950', '109148', '322157', '557607', '184826', '83987', '64456', '489706', '249628', '523303']
    tail (the full-question list) is appended and dict.fromkeys removes repeats, but the
    10 slots are already full, so the tail adds nothing here. Final answer:
        ['402950', '109148', '322157', '557607', '184826', '83987', '64456', '489706', '249628', '523303']

    --- What the example shows ---------------------------------------------------------------

    * Coverage: both halves are in the top 10 (64456 for half a, 557607 for half b). Each sub-
      query got exactly 5 slots, which is what round-robin promises.
    * The verifier earned its call: without the rewrite, sub B's 5 slots would have held no
      gold, and half b would be missing.
    * The cost of using the full-question list only as a filler: 386128 (gold, half a) is at
      rank 8 of that list, but the interleave had already filled all 10 slots, so it never got in.
    * Budget: 2 LLM calls (planner + verifier), 839 input and 123 output tokens, 4
      searches. The single agent takes up to 3 calls and re-sends every result block each
      step, which is why it costs more per query.

    A contrast, c023 ("How do I trade a large number of shares, and can we amend last year's
    Schedule C ..."): the verifier also marked sub-query 1 as off-topic, but its first list
    already had gold passage 362060 at rank 1, and the re-search returned it at rank 1 again.
    That was a false alarm. Here it was harmless, but each one costs a search and could
    replace a good list with a worse one, which is why the VERIFIER prompt sets a high bar
    ("ONLY if none of the shown results is on-topic").
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
