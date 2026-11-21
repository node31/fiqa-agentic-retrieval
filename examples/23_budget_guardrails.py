"""Toy 23: budget guardrails. A shared call cap + token budget across all agents of one query.

Concept: check BEFORE spending; degrade gracefully when the budget runs out.
Real code: src/agent/llm.py::Budget, Orchestrator.run    Doc: docs/topics/13_multi_agent_orchestration.md
"""
import json

from _common import banner, note, step, try_this

from src.agent.llm import Budget, BudgetExceeded, FakeLLM, estimate_tokens
from src.agent.orchestrator import Orchestrator
from src.agent.tools import SearchTool

banner("Toy 23: guardrails in action")
b = Budget(max_calls=3, max_tokens=1000)
llm = FakeLLM(lambda s, u: "ok " * 50)
step(1, f"Budget(max_calls=3, max_tokens=1000). Each call: ~{len('x' * 800) // 4} prompt tokens + up to 100 output")
for i in range(1, 5):
    try:
        llm.complete("sys", "x" * 800, max_output_tokens=100, budget=b)
        print(f"     call {i}: OK      calls={b.calls} tokens={b.tokens}")
    except BudgetExceeded as e:
        print(f"     call {i}: REFUSED ({e}) - nothing was spent")
worst = b.tokens + estimate_tokens("sys" + "x" * 800) + 100
note(f"Call 4 hit the call cap first, but it would ALSO fail the token check: reserve() counts the WORST case "
     f"({b.tokens} spent + ~{estimate_tokens('sys' + 'x' * 800)} in + 100 max out = {worst} > {b.max_tokens}), even "
     f"though the real reply would only use ~{b.output_tokens // b.calls} output tokens. Checking before the call means "
     "even parallel workers can't jointly overshoot. A post-hoc check would only tell you AFTER the money is gone.")
step(2, "The orchestrator degrades instead of failing: with max_llm_calls=1 the verifier is skipped")
tool = SearchTool(lambda q, s, k: [{"doc_id": f"{q.split()[0]}{i}", "score": 1 / (i + 1), "text": q} for i in range(k)])
plan = json.dumps({"simple": False, "subqueries": ["shorting risks", "mortgage payoff"]})
r = Orchestrator(FakeLLM(lambda s, u: plan), tool, max_llm_calls=1).run("How does shorting work, and should I prepay my mortgage?")
print(f"     llm_calls={r.llm_calls}, results={r.doc_ids[:4]}..., errors={r.errors}")
note("Policy: budgets protect cost, and partial agent work is still a valid answer. Our defaults: single agent 3 "
     "calls / 12K tokens, multi-agent 3 calls / 8K tokens per query.")
try_this("Set max_tokens so the planner fits but the verifier doesn't. Which trace entry shows it?",
         "Add a per-RUN dollar cap (scripts/agent_eval.py --max-usd does this). Why do you need both?",
         "Should budget exhaustion be logged as an error or a metric? Argue both sides.")
