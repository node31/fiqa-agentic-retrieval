r"""Single agent (Part C2): one LLM, one tool, at most 3 steps, then a final top-10.

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

WORKED EXAMPLE (a real run: compound question c020 from the Azure gpt-6-sol eval)

    Every prompt below is EXACTLY what the model received. I rebuilt each one with this code
    and found its sha256 in .llm_cache/azure/ (the cache key is a hash of the full prompt,
    see src/agent/llm.py), so a single changed character would have missed the cache.

    The question joins two unrelated questions (that is what makes it "compound"):
        half a: "Can stock brokerage firms fail?"
        half b: "(How) can I print my own checks on my printer on regular paper?"
        gold passages: 526260, 122540, 370194 (half a) and 384251 (half b)

    --- Step 0: search the raw question (no LLM call) ---------------------------------------

    searches = [("hybrid", question, tool(question, "hybrid"))]

    The tool returns 10 hit dicts. The router gives the full passage; SearchTool keeps only
    doc_id, score and the first 200 characters of the text. The second hit, for example:
        {"doc_id": "384251", "score": 0.917544,
         "text": "There are certain standards that modern checks need to meet. These aren't
                  required by law, ... print your own checks a"}     (200 chars, cut mid-word)

    So after step 0:
        searches = [
            ("hybrid", "Can stock brokerage firms fail, and can I print my own checks on regular paper with my printer?", [10 hit dicts]),
        ]

    --- Step 1: the first LLM call -----------------------------------------------------------

    1. enumerate(searches, 1) yields (1, ("hybrid", <question>, <10 hits>)). The tuple is
       unpacked as i=1, s="hybrid", q=<question>, h=<10 hits>.
    2. The f-string turns that into one block:
           "Search 1 (hybrid): " + repr(q) + "\n" + render_hits(h)
    3. {q!r} is repr(q): the query wrapped in quotes. Python picks single quotes, and switches
       to double quotes when the text itself has an apostrophe:
           repr("Can stock ...")     -> 'Can stock ...'
           repr("What's a Roth IRA?") -> "What's a Roth IRA?"
       The quotes show the model exactly where the query starts and ends, so it can tell what
       was already searched and write a DIFFERENT query next.
    4. render_hits(h) (src/agent/tools.py) makes one line per hit, "[doc_id] snippet":
       * the score is dropped (the model judges by text; fewer characters = fewer tokens),
       * " ".join(text.split()) collapses newlines and runs of spaces, so a hit can never
         spill onto a second line. Hit 20791's raw snippet has a double space:
             " and on line the offer check printing.  From what I've seen,"
         and its rendered line has a single one ("printing. From").
       * an empty list renders as "(no results)".
    5. "\n\n".join(blocks) puts a blank line between blocks (only one block so far), and
       "Question: ...\n\n" goes in front.

    The system prompt is prompts.SINGLE_AGENT with "{remaining}" replaced by 3 (3 actions
    left, counting this one). The USER prompt, exactly as sent (2316 characters):

        Question: Can stock brokerage firms fail, and can I print my own checks on regular paper with my printer?

        Search 1 (hybrid): 'Can stock brokerage firms fail, and can I print my own checks on regular paper with my printer?'
        [500403] Not illegal. With respect to littleadv response, the printing of a check isn't illegal. I can order checks from cheap check printers, and they have no relationship to any bank, so long as they have my
        [384251] There are certain standards that modern checks need to meet. These aren't required by law, but banks today generally insist on them. If you are able to meet these standards and print your own checks a
        [135196] "Checks sold as ""business checks"" are larger than checks sold as ""personal checks"". Personal checks are usually 6"" x 2 1/2"" while business checks are 8 1/2 "" x 3 to 4 "". Also, business checks
        [258419] For some very small private companies I know of (and am part of), paper stocks do exist. You can sit at the table with the damn things in your hand and wave them in people's faces. They tell everyone
        [229546] "Legally, a check just needs to have a certain list of things (be an instruction to one's bank to pay a specific amount of money to bearer or to a specific entity, have a date, have a signature, etc.)
        [362215] It might be illegal for the very reason you stated: The process of printing checks may seem like check forgery. Banks in the US are allowed to do that, and the only condition under which you can do it
        [20791] There is no reason you must buy the bank's printed check. There are many places both physical stores and on line the offer check printing. From what I've seen, the requirement is the use of a magneti
        [260499] I don't know, ask the various companies I'm forced to do business with why in the hell they want me to stop by their office so I can drop a check off vs just using some means of digital payment. I wo
        [8854] "Technically it doesn't matter what size the check is. In fact, it doesn't even have to be written on paper. While writing it on a cow may not always fly, almost any object actually will. That said, m
        [564870] "The answer to your question is ""no"". Unless you specifically ask to receive paper share certificates, then brokers will hold your shares with a custodian company in the broker's own nominee account

    gpt-6-sol replied (749 input tokens, 46 output tokens, 1.43 s):
        {"action":"search","query":"Can stock brokerage firms go bankrupt fail what happens to customers shares investments SIPC protection","strategy":"hybrid"}

    8 of the 10 hits are about checks and the brokerage half is missing. The model's next query
    targets that half only. The code runs that search and appends it:
        searches = [
            ("hybrid", <question>, <10 hits about checks>),
            ("hybrid", "Can stock brokerage firms go bankrupt fail what happens to c...", <10 new hits>),
        ]

    --- Step 2: the second LLM call ----------------------------------------------------------

    enumerate now yields TWO tuples, so the prompt has two blocks. The system prompt says
    "at most 2 more action(s)". The USER prompt (4504 characters) is the step-1 prompt,
    byte for byte, plus one new block at the end:

        Question: Can stock brokerage firms fail, and can I print my own checks on regular paper with my printer?

        Search 1 (hybrid): 'Can stock brokerage firms fail, and can I print my own checks on regular paper with my printer?'
        [500403] Not illegal. With respect to littleadv response, the printing of a check isn't illegal. I can order checks from cheap check printers, and they have no relationship to any bank, so long as they have my
        [... the other 9 hit lines of Search 1, identical to step 1 ...]

        Search 2 (hybrid): 'Can stock brokerage firms go bankrupt fail what happens to customers shares investments SIPC protection'
        [304851] You are asking about what happens when an ETF/mutual fund company goes bankrupt. If you were asking about a bank account you would be asking about FDIC coverage. Investment funds are different, the c
        [322645] There is a measure of protection for investors. It is not the level of protection provided by FDIC or NCUA but it does exist: Securities Investor Protection Corporation What SIPC Protects SIPC protect
        [122540] Yes, any company can go under. SIPC offers a level of protection. They don't guarantee against stocks dropping, but will replace stocks that you owned, but the broker stole from you. (overgeneralizati
        [418336] Not sure if I follow your question completely. Re: What if some fraud takes place that's too big even for it to fund? SIPC does not fund anything. What it does is takes over the troubled brokerage f
        [415915] "SIPC is a corporation - a legal entity separate from its owners. In the case of SIPC, it is funded through the fees paid by its members. All the US brokers are required to be members and to contribut
        [55443] "I have received a response from SIPC, confirming littleadv's answer: For a brief background, the protections available under the Securities Investor Protection Act (""SIPA""), are only available in
        [556549] "Careful with the ""stock stolen from your account"" thing. SIPC protects investors against broker/dealer insolvency. Don't think they provide protection against theft."
        [526260] Brokerages are supposed to keep your money separate from theirs. So, even if they fail as a company, your money and investments are still there, and can be transferred to another brokerage. It doesn't
        [226337] If you are using a US broker, you are protected by SIPC up to $500,000. SIPC also oversees the liquidation of the broker itself, either by appointing a trustee, or by directly contacting clients. If t
        [217042] "Look at the link to the SIPC. I don't know exactly what you mean by ""runs out of funds,"" but the SIPC will replace shares of stock stolen from your account, and up to $100,000 in cash. The real ri

    gpt-6-sol replied (1,248 input tokens, 66 output tokens, 2.32 s):
        {"action":"finish","doc_ids":["122540","526260","226337","322645","229546","384251",
                                     "500403","20791","8854","418336"]}

    --- Finish: validate and fill ------------------------------------------------------------

    * Every id the model returned appears in one of the two blocks, so all 10 survive the
      "d in seen" check. Had it invented an id, that id would have been dropped silently.
    * The model's list already has 10 ids, so the RRF fill adds nothing here.
    * Look at where the picks come from: 122540, 526260, 226337, 322645, 418336 are from
      Search 2 (brokerages), and 229546, 384251, 500403, 20791, 8854 are from Search 1
      (checks). The final answer covers both halves, and 3 of the 4 gold passages are in it
      (370194 was in neither search). The whole run used 2 LLM calls and 2,109 tokens.

WHY THE WHOLE HISTORY IS SENT AGAIN ON EVERY STEP

    1. The model API is stateless. Each llm.complete() is an independent HTTP request; the model
       keeps nothing from the previous call. Chat products work the same way underneath: the
       model only reads what is in its input, so the app (client or server) keeps the
       conversation and puts earlier turns back into the input on every turn, often trimmed or
       summarised once it gets long. If step 2 only got the new block, the model
       would not know that Search 1 happened, which half it already covered, or what it asked
       for last time.
    2. The finish action picks ids from ALL searches. In c020, half of the final list comes from
       Search 1. Without Search 1's lines in the step-2 prompt, the model could not choose the
       check-printing passages (it would not even know their ids), and half b would be lost.
       The last step is effectively a rerank over everything seen so far, so everything seen
       so far must be in front of it.
    3. The code checks the model's ids against every search shown (the "seen" set). Showing
       every search keeps "what the model could read" and "what we accept" the same set.

    The cost is that input grows with each step: 749 input tokens at step 1, 1,248 at step 2,
    and a third step would add about another 500. The 3-step cap keeps that bounded. I didn't
    use the cheaper options:
        * re-sending only the doc ids of earlier searches: the model can't judge passages it
          can't read, so its final ordering would be guesswork for those ids;
        * a provider-side conversation (e.g. OpenAI's previous_response_id): the provider
          stores the history, but it still goes into the model's input on every call and is
          billed as input tokens, so it saves request bytes, not cost, and it ties the code to
          one provider's API.
    Because each new prompt starts with the previous prompt unchanged (only appended to), a
    provider's prompt-prefix caching can reuse the repeated part. This code does not depend on
    that.
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
        # Build the USER prompt for this step. The model has no memory between calls (each
        # llm.complete is a fresh request), so the prompt is rebuilt from scratch every step and
        # carries everything the model should know: the question + EVERY search done so far.
        #
        # Piece by piece:
        #   searches              list of (strategy, query, hits) tuples. Step 0 put the original
        #                         question there; each "search" action appends one more (below).
        #   enumerate(..., 1)     numbers the searches 1, 2, 3 (start=1 reads better than 0).
        #   (s, q, h)             unpacks one tuple: s = strategy, q = query text, h = hit list.
        #   {q!r}                 repr(q): the query in quotes ('...'), so the model sees exactly
        #                         what was searched, including spaces, and can avoid repeating it.
        #   render_hits(h)        (src/agent/tools.py) turns the hits into one line per hit:
        #                         "[doc_id] snippet". The snippet is the passage text already cut
        #                         to 200 chars by SearchTool, with newlines and runs of spaces
        #                         collapsed (" ".join(text.split())) so every hit stays on ONE
        #                         line. Scores are left out on purpose: the model judges by text,
        #                         and fewer characters means fewer tokens. No hits -> "(no results)".
        #   "\n\n".join(...)      a blank line between search blocks; the question goes first.
        #
        # The module docstring has a real run (WORKED EXAMPLE) showing the exact prompt at each
        # step, and WHY THE WHOLE HISTORY IS SENT AGAIN ON EVERY STEP.
        #
        # The bracketed ids are what the model copies back in a finish action; the validation
        # further down keeps only ids that appear in these blocks. Cost: the prompt grows with
        # every search (up to 10 hits x ~200 chars each), so the 3-step cap also bounds its size.
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
