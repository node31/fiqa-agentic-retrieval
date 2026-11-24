"""Model cascade: a cheap tier answers first; only the cases it is unsure about go to the strong tier.

WHY
    Most questions are easy. Paying the strong model's price for every one of them wastes money, so a
    cascade (sometimes called "LLM routing" or FrugalGPT-style escalation) tries a cheaper configuration
    first and pays for the strong one only when the cheap answer looks unreliable. Quality stays close to
    strong-only if the escalation check is good; cost drops by the share of questions that never escalate.

THE TWO TIERS (``default_tiers``)
    cheap   ONE call, no extra searches: the question's hybrid results with TRIMMED context (top 5 hits,
            120-character snippets), and a prompt that also asks for a confidence ("high" / "low").
            Version 1 let the cheap tier search again; with less context it did, on 80 of 100 plain
            questions (3 calls each), and cost almost as much as the strong tier (docs/topics/19_llm_cost.md).
    strong  the single agent exactly as before: 10 hits, 200 characters, the original prompt. Because
            the prompt is byte-for-byte the same, answers already in .llm_cache replay for free.
    Each tier has its OWN LLM and its own prices (CASCADE_CHEAP_* / CASCADE_STRONG_* env vars). Today
    both tiers call the same Azure deployment, so the saving comes only from the cheap tier reading fewer
    tokens. Point the cheap tier at a cheaper model (CASCADE_CHEAP_BACKEND / CASCADE_CHEAP_DEPLOYMENT)
    and its prices, and the same code saves on the price per token too.

MEASURED (Azure gpt-6-sol in both tiers; results/agent_eval_cascade.json, results/cascade_policies.json)
    plain     R@10 0.546 vs 0.547 strong-only, $0.00240 vs $0.00251 per question (-4%)
    compound  every question routed to strong: identical to strong-only
    With one model in both tiers the cascade saves little; scripts/cascade_policies.py shows the saving
    with a tier-1 model at a tenth of the price (-18% on plain with the default rule).

ROUTING BEFORE THE CASCADE (``route_compound``, on by default)
    A question that looks compound (orchestrator.looks_simple says no: long, or joined by "and", ";", two
    question marks) goes STRAIGHT to the strong tier: one cheap call it would almost surely fail costs
    money and latency for nothing. The check is free (no LLM).

WHEN IT ESCALATES (``escalation_reason``), cheapest signal first
    error            the cheap tier failed (bad JSON, timeout, budget)        -> nothing trustworthy to keep
    no finish        it ran out of steps without finishing
    low confidence   it said so itself (the prompt sets a strict bar for "high")
    too few picks    fewer than 3 valid ids: it found little that answers the question
    A wrong "keep" loses quality, a wrong "escalate" only costs one strong run, so every rule errs toward
    escalating.

    >>> from src.agent.llm import FakeLLM
    >>> from src.agent.fake_policies import heuristic_responder
    >>> hits = [{"doc_id": str(i), "score": 1.0 / i, "text": f"passage {i}"} for i in range(1, 11)]
    >>> tool = lambda q, s="hybrid": hits
    >>> cheap, strong = default_tiers(FakeLLM(heuristic_responder), FakeLLM(heuristic_responder))
    >>> r = run_cascade("what is a roth ira?", cheap, strong, tool)
    >>> r.path, r.trace[0]["tier"], r.trace[0]["escalate"]          # one topic: the fake is confident
    ('cascade', 'cheap', None)
    >>> r = run_cascade("what is a roth ira and should I pay off my mortgage?", cheap, strong, tool)
    >>> [t["tier"] for t in r.trace], r.trace[0]["escalate"]                  # compound: routed, no cheap call
    (['strong'], 'routed: compound')
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from src.agent import prompts
from src.agent.llm import LLM
from src.agent.single_agent import AgentResult, run_single_agent
from src.agent.tools import SearchTool
from src.config import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT

MIN_PICKS = 3


@dataclass
class Tier:
    """One rung of the cascade: which LLM, at what price, reading how much context."""
    name: str
    llm: LLM
    price_in: float                 # USD per 1M input tokens
    price_out: float                # USD per 1M output tokens
    system_prompt: str = prompts.SINGLE_AGENT
    max_hits: int | None = None     # context trimming, see run_single_agent
    snippet_chars: int | None = None
    dedupe: bool = False
    max_steps: int = 3              # 1 = answer from the first search, no extra searches

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * self.price_in + output_tokens / 1e6 * self.price_out

    def run(self, question: str, tool: SearchTool, k: int = 10) -> AgentResult:
        return run_single_agent(question, self.llm, tool, max_steps=self.max_steps, k=k,
                                system_prompt=self.system_prompt, max_hits=self.max_hits,
                                snippet_chars=self.snippet_chars, dedupe=self.dedupe)


def _price(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def default_tiers(cheap_llm: LLM, strong_llm: LLM) -> tuple[Tier, Tier]:
    """The cheap tier (trimmed context + confidence) and the strong tier (the agent as it always was).

    Prices default to the project's PRICE_PER_1M_* for both (the same deployment today); set
    CASCADE_CHEAP_PRICE_IN/OUT and CASCADE_STRONG_PRICE_IN/OUT when the tiers are different models.
    """
    cheap = Tier("cheap", cheap_llm, _price("CASCADE_CHEAP_PRICE_IN", PRICE_PER_1M_INPUT),
                 _price("CASCADE_CHEAP_PRICE_OUT", PRICE_PER_1M_OUTPUT), prompts.SINGLE_AGENT_CHEAP,
                 max_hits=5, snippet_chars=120, dedupe=True, max_steps=1)
    strong = Tier("strong", strong_llm, _price("CASCADE_STRONG_PRICE_IN", PRICE_PER_1M_INPUT),
                  _price("CASCADE_STRONG_PRICE_OUT", PRICE_PER_1M_OUTPUT))
    return cheap, strong


def cascade_llms(backend: str, confirm_spend: bool = False) -> tuple[LLM, LLM]:
    """(cheap LLM, strong LLM). Both default to ``backend``. CASCADE_CHEAP_BACKEND and
    CASCADE_CHEAP_DEPLOYMENT (an Azure deployment name) point the cheap tier somewhere else."""
    from src.agent.llm import AzureLLM, get_llm

    strong = get_llm(backend, confirm_spend=confirm_spend)
    cheap_backend = os.environ.get("CASCADE_CHEAP_BACKEND", backend)
    cheap = get_llm(cheap_backend, confirm_spend=confirm_spend)
    if isinstance(cheap, AzureLLM) and os.environ.get("CASCADE_CHEAP_DEPLOYMENT"):
        cheap.model = os.environ["CASCADE_CHEAP_DEPLOYMENT"]
    return cheap, strong


def escalation_reason(res: AgentResult, min_picks: int = MIN_PICKS) -> str | None:
    """Why the cheap tier's result should NOT be kept, or None to keep it."""
    if res.errors:
        return "error"
    finishes = [t["action"] for t in res.trace if t["action"].get("action") == "finish"]
    if not finishes:
        return "no finish"
    final = finishes[-1]
    if str(final.get("confidence", "")).lower() != "high":
        return "low confidence"
    picks = [d for d in dict.fromkeys(map(str, final.get("doc_ids") or [])) if d in res.doc_ids]
    if len(picks) < min_picks:
        return "too few picks"
    return None


def corpus_unanswered(res: AgentResult, min_picks: int = MIN_PICKS) -> bool:
    """True if even the strong tier found little: the question reached the strong tier (escalated or routed),
    and the strong tier failed or picked fewer than ``min_picks`` passages. That is the agent's signal for the web fallback."""
    tiers = [t for t in res.trace if t.get("tier") in ("cheap", "strong") and not t.get("shadow")]
    if not tiers or tiers[-1]["tier"] != "strong":   # kept by the cheap tier, or a cache hit
        return False
    strong = tiers[-1]
    if strong["errors"]:
        return True
    finishes = [s["action"] for s in strong["trace"] if s["action"].get("action") == "finish"]
    picks = [d for d in dict.fromkeys(map(str, finishes[-1].get("doc_ids") or [])) if d in res.doc_ids] if finishes else []
    return len(picks) < min_picks


def run_cascade(question: str, cheap: Tier, strong: Tier, tool: SearchTool, k: int = 10,
                cache=None, tenant: str | None = None, route_compound: bool = True,
                shadow: bool = False) -> AgentResult:
    """Run the cheap tier; escalate to the strong tier if ``escalation_reason`` says so.

    ``cache`` (a SemanticCache, src/agent/semantic_cache.py) is checked FIRST: a question close enough to
    one answered before (same tenant) returns that answer with no LLM call and no search, and a fresh
    answer is stored for next time. The cheapest LLM call is the one you never make.
    ``shadow`` (evaluation only): also run the tier that did NOT run, and record it with "shadow": True,
    outside every total, so scripts/cascade_policies.py can compare escalation rules on the same questions.

    The returned AgentResult carries the FINAL tier's doc ids, the SUM of both tiers' calls, tokens and
    latency (an escalated question pays for both), and its cost priced per tier. ``trace`` has one entry
    per tier that ran: {"tier", "escalate" (cheap only), "llm_calls", "tokens", "usd", "trace"}.
    """
    t0 = time.perf_counter()
    if cache is not None:
        hit = cache.lookup(question, tenant)
        if hit:
            return AgentResult(list(hit["value"]), path="cascade", priced_usd=0.0,
                               latency_s=time.perf_counter() - t0,
                               trace=[{"tier": "cache", "escalate": None, "similarity": hit["similarity"],
                                       "cached_question": hit["question"], "llm_calls": 0, "input_tokens": 0,
                                       "output_tokens": 0, "usd": 0.0}])
    from src.agent.orchestrator import looks_simple

    if route_compound and not looks_simple(question):
        reason, runs = "routed: compound", [(strong, strong.run(question, tool, k))]
    else:
        runs = [(cheap, cheap.run(question, tool, k))]
        reason = escalation_reason(runs[0][1])
        if reason:
            runs.append((strong, strong.run(question, tool, k)))
    final = runs[-1][1]
    out = AgentResult(final.doc_ids, path="cascade", errors=list(final.errors))
    usd = 0.0
    for tier, r in runs:
        cost = tier.usd(r.input_tokens, r.output_tokens)
        usd += cost
        out.llm_calls += r.llm_calls
        out.input_tokens += r.input_tokens
        out.output_tokens += r.output_tokens
        out.cached_input_tokens += r.cached_input_tokens
        out.searches += r.searches
        out.trace.append({"tier": tier.name, **({"escalate": reason} if tier is runs[0][0] else {}),
                          "llm_calls": r.llm_calls, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                          "usd": round(cost, 6), "doc_ids": r.doc_ids, "errors": r.errors, "trace": r.trace})
    out.priced_usd = usd
    out.latency_s = time.perf_counter() - t0
    if shadow:
        ran = {tier.name for tier, _ in runs}
        for tier in (cheap, strong):
            if tier.name not in ran:
                r = tier.run(question, tool, k)
                out.trace.append({"tier": tier.name, "shadow": True, "llm_calls": r.llm_calls,
                                  "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                                  "usd": round(tier.usd(r.input_tokens, r.output_tokens), 6), "doc_ids": r.doc_ids,
                                  "errors": r.errors, "trace": r.trace,
                                  **({"escalate": escalation_reason(r)} if tier is cheap else {})})
    if cache is not None and not out.errors:
        cache.store(question, out.doc_ids, tenant)
    return out
