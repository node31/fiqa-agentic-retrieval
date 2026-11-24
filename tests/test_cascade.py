"""Model cascade, semantic cache and provider-cached tokens, with fake LLMs (no network, ~1 s)."""
import json

import pytest

from src.agent import prompts
from src.agent.cascade import Tier, corpus_unanswered, default_tiers, escalation_reason, run_cascade
from src.agent.fake_policies import heuristic_responder
from src.agent.llm import Budget, FakeLLM, LLM
from src.agent.semantic_cache import SemanticCache
from src.agent.single_agent import run_single_agent
from src.retrieval.dense import HashEmbedder

HITS = [{"doc_id": str(i), "score": 1.0 / i, "text": f"passage number {i} " + "x" * 300} for i in range(1, 11)]
SIMPLE, COMPOUND = "what is a roth ira?", "what is a roth ira and should I pay off my mortgage early?"


def tool(query, strategy="hybrid"):
    return [dict(h, text=h["text"][:200]) for h in HITS]


class Recorder(FakeLLM):
    """A fake that also records every prompt it was sent."""

    def __init__(self, responder=heuristic_responder):
        super().__init__(responder)
        self.use_cache = False
        self.prompts: list[tuple[str, str]] = []

    def _call(self, system, user, max_output_tokens):
        self.prompts.append((system, user))
        return super()._call(system, user, max_output_tokens)


def scripted(*replies):
    it = iter(replies)
    return Recorder(lambda s, u: next(it))


def test_kept_when_confident_and_escalated_when_not():
    cheap_llm, strong_llm = Recorder(), Recorder()
    cheap, strong = default_tiers(cheap_llm, strong_llm)
    r = run_cascade(SIMPLE, cheap, strong, tool)
    assert [t["tier"] for t in r.trace] == ["cheap"] and not strong_llm.prompts
    r = run_cascade(COMPOUND, cheap, strong, tool, route_compound=False)
    assert [t["tier"] for t in r.trace] == ["cheap", "strong"] and r.trace[0]["escalate"] == "low confidence"
    assert r.llm_calls == r.trace[0]["llm_calls"] + r.trace[1]["llm_calls"]  # an escalated question pays twice
    assert r.trace[0]["llm_calls"] == 1                                      # the cheap tier is one-shot


def test_compound_questions_skip_the_cheap_tier_and_shadow_runs_stay_out_of_the_totals():
    cheap_llm, strong_llm = Recorder(), Recorder()
    cheap, strong = default_tiers(cheap_llm, strong_llm)
    r = run_cascade(COMPOUND, cheap, strong, tool)
    assert [t["tier"] for t in r.trace] == ["strong"] and r.trace[0]["escalate"] == "routed: compound"
    assert not cheap_llm.prompts
    s = run_cascade(COMPOUND, cheap, strong, tool, shadow=True)
    assert [(t["tier"], t.get("shadow", False)) for t in s.trace] == [("strong", False), ("cheap", True)]
    assert s.llm_calls == r.llm_calls and s.cost_usd == pytest.approx(r.cost_usd)


def test_cheap_tier_reads_less_and_strong_tier_prompt_is_unchanged():
    cheap_llm, strong_llm = Recorder(), Recorder()
    cheap, strong = default_tiers(cheap_llm, strong_llm)
    run_cascade(COMPOUND, cheap, strong, tool, route_compound=False)
    cheap_user, strong_user = cheap_llm.prompts[0][1], strong_llm.prompts[0][1]
    assert cheap_user.count("\n[") == 5 and strong_user.count("\n[") == 10    # 5 hits shown vs 10
    assert len(cheap_user) < len(strong_user) / 2                             # 120-char snippets, not 200
    # The strong tier sends exactly what the plain single agent sends, so the disk cache replays old answers.
    plain = Recorder()
    run_single_agent(COMPOUND, plain, tool)
    assert strong_llm.prompts == plain.prompts
    assert strong_llm.prompts[0][0].startswith(prompts.SINGLE_AGENT.split("{remaining}")[0])


@pytest.mark.parametrize("replies,reason", [
    (["not json at all"], "error"),
    ([json.dumps({"action": "finish", "doc_ids": ["1", "2", "3"], "confidence": "low"})], "low confidence"),
    ([json.dumps({"action": "finish", "doc_ids": ["1"], "confidence": "high"})], "too few picks"),
    ([json.dumps({"action": "finish", "doc_ids": ["1", "2", "99"], "confidence": "high"})], "too few picks"),  # 99 unseen
    ([json.dumps({"action": "finish", "doc_ids": ["1", "2", "3"], "confidence": "high"})], None),
])
def test_escalation_reasons(replies, reason):
    cheap = Tier("cheap", scripted(*replies), 1, 4, prompts.SINGLE_AGENT_CHEAP, max_hits=5)
    assert escalation_reason(cheap.run(SIMPLE, tool)) == reason


def test_cost_is_priced_per_tier():
    cheap = Tier("cheap", Recorder(), price_in=0.10, price_out=0.40, system_prompt=prompts.SINGLE_AGENT_CHEAP,
                 max_hits=5, snippet_chars=120, dedupe=True)
    strong = Tier("strong", Recorder(), price_in=10.0, price_out=40.0)
    kept, escalated = (run_cascade(SIMPLE, cheap, strong, tool),
                       run_cascade(COMPOUND, cheap, strong, tool, route_compound=False))
    assert kept.cost_usd == pytest.approx(cheap.usd(kept.input_tokens, kept.output_tokens))
    c, s = escalated.trace
    assert escalated.cost_usd == pytest.approx(cheap.usd(c["input_tokens"], c["output_tokens"])
                                               + strong.usd(s["input_tokens"], s["output_tokens"]))


def test_semantic_cache_skips_every_llm_call_and_respects_tenants():
    emb = HashEmbedder(seed="t")
    cache = SemanticCache(lambda q: emb.encode_queries([q])[0], threshold=0.95)
    cheap_llm, strong_llm = Recorder(), Recorder()
    cheap, strong = default_tiers(cheap_llm, strong_llm)
    first = run_cascade(SIMPLE, cheap, strong, tool, cache=cache, tenant="acme")
    calls = len(cheap_llm.prompts)
    again = run_cascade("What is a Roth IRA?", cheap, strong, tool, cache=cache, tenant="acme")
    assert again.trace[0]["tier"] == "cache" and again.llm_calls == 0 and again.cost_usd == 0
    assert again.doc_ids == first.doc_ids and len(cheap_llm.prompts) == calls
    other = run_cascade(SIMPLE, cheap, strong, tool, cache=cache, tenant="globex")   # never across tenants
    assert other.trace[0]["tier"] == "cheap"
    assert cache.stats()["hits"] == 1


def test_provider_cached_tokens_are_recorded():
    class Cached(LLM):
        backend, model = "test", "m"

        def _call(self, system, user, max_output_tokens):
            return '{"action": "finish", "doc_ids": []}', 1500, 20, 1024   # 4th value: provider cache

    b = Budget(3, 10_000)
    Cached(use_cache=False).complete("s", "u", budget=b)
    assert b.cached_input_tokens == 1024
    assert run_single_agent(SIMPLE, Cached(use_cache=False), tool).cached_input_tokens == 1024


def test_corpus_unanswered_needs_an_escalation_and_a_weak_strong_tier():
    cheap = Tier("cheap", scripted(json.dumps({"action": "finish", "doc_ids": [], "confidence": "low"})), 1, 4,
                 prompts.SINGLE_AGENT_CHEAP)
    weak = Tier("strong", scripted(json.dumps({"action": "finish", "doc_ids": ["1"]})), 1, 4)
    assert corpus_unanswered(run_cascade(SIMPLE, cheap, weak, tool))
    cheap = Tier("cheap", scripted(json.dumps({"action": "finish", "doc_ids": [], "confidence": "low"})), 1, 4,
                 prompts.SINGLE_AGENT_CHEAP)
    good = Tier("strong", scripted(json.dumps({"action": "finish", "doc_ids": ["1", "2", "3"]})), 1, 4)
    assert not corpus_unanswered(run_cascade(SIMPLE, cheap, good, tool))
