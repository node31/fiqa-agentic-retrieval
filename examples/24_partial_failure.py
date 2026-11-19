"""Toy 24: partial failure. One worker crashes, one hangs, and the user still gets an answer.

Concept: isolate sub-agent failures with timeouts; never let one sub-task fail the whole request.
Real code: Orchestrator._run_workers, tests/test_agents.py::test_orchestrator_partial_failure...
Doc: docs/13_multi_agent_orchestration.md
"""
import json
import time

from _common import banner, note, step, try_this

from src.agent.llm import FakeLLM
from src.agent.orchestrator import Orchestrator
from src.agent.tools import SearchTool

banner("Toy 24: 3 parallel workers - one fails, one is slow, one is fine")


def flaky_search(q, strategy, k):
    if q.startswith("crash"):
        raise ConnectionError("search shard unreachable")
    if q.startswith("slow"):
        time.sleep(3)
    return [{"doc_id": f"{q.split()[0]}-{i}", "score": 1 / (i + 1), "text": q} for i in range(k)]


plan = json.dumps({"simple": False, "subqueries": ["crash me please", "slow query here", "bonds yield curve"]})
llm = FakeLLM(lambda s, u: plan if "planner" in s else json.dumps({"checks": []}))
orch = Orchestrator(llm, SearchTool(flaky_search), worker_timeout_s=0.5)
t0 = time.perf_counter()
r = orch.run("Tell me about crashes, slow markets, and bond yields, please?")
step(1, f"returned in {time.perf_counter() - t0:.2f}s (the slow worker needed 3 s)")
step(2, f"doc_ids = {r.doc_ids}")
step(3, "errors reported to the caller:")
for e in r.errors:
    print(f"     - {e}")
note("The healthy worker's results fill the slots, and the full-question search (always run for decomposed "
     "questions) fills the rest. If EVERY worker fails, the orchestrator falls back to one direct search.")
note("PITFALL: Python threads can't be killed. The slow worker keeps running in the background after we stop "
     "waiting. In production, propagate a deadline into the tool call (e.g. an HTTP timeout) instead.")
try_this("Make all three sub-queries crash. What does the user get?",
         "Lower worker_timeout_s to 0.01. What breaks, and how would you pick the value? (hint: p99 of search)",
         "Return the partial-failure info to the end user like the router's `partial` flag. Where?")
