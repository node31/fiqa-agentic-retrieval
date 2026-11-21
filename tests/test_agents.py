"""Agent tests with a FakeLLM and a fake search tool: no network, no money, deterministic."""
import json
import time

import pytest

from src.agent.llm import Budget, BudgetExceeded, FakeLLM, parse_json
from src.agent.orchestrator import Orchestrator, interleave, looks_simple
from src.agent.single_agent import run_single_agent
from src.agent.tools import SearchTool


def fake_search(query, strategy, k):
    """Returns ids derived from the query words, so different sub-queries give different lists."""
    if "FAIL" in query:
        raise ConnectionError("shard down")
    if "SLOW" in query:
        time.sleep(2)
    w = query.lower().split()[0]
    return [{"doc_id": f"{w}{i}", "score": 1.0 / (i + 1), "text": f"passage about {query} #{i}"} for i in range(10)]


TOOL = lambda: SearchTool(fake_search)  # noqa: E731
COMPOUND = "how does short selling work, and should I pay my mortgage early"


def scripted(*replies):
    it = iter(replies)
    return FakeLLM(lambda s, u: next(it))


def test_parse_json_handles_fences_and_escaped_quotes():
    assert parse_json('x ```json\n{"a": "b\\"}"}\n``` y') == {"a": 'b"}'}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_budget_refuses_before_spending():
    b = Budget(max_calls=1, max_tokens=10_000)
    b.reserve(100, 100)
    with pytest.raises(BudgetExceeded):
        b.reserve(100, 100)
    with pytest.raises(BudgetExceeded):
        Budget(5, 150).reserve(100, 100)  # 100 in + 100 worst-case out > 150


def test_single_agent_caps_at_three_llm_calls_and_returns_ten():
    llm = scripted(*[json.dumps({"action": "search", "query": f"rewrite{i} q", "strategy": "dense"})
                     for i in range(10)])
    r = run_single_agent("what is a roth ira", llm, TOOL())
    assert r.llm_calls == 3 and r.searches == 3  # initial search + 2 rewrites; 3rd action forced finish
    assert len(r.doc_ids) == 10


def test_single_agent_validates_ids_and_fills():
    llm = scripted(json.dumps({"action": "finish", "doc_ids": ["what3", "INVENTED", "what3"]}))
    r = run_single_agent("what is a roth ira", llm, TOOL())
    assert r.doc_ids[0] == "what3" and "INVENTED" not in r.doc_ids and len(r.doc_ids) == 10
    assert r.llm_calls == 1


def test_single_agent_survives_garbage_llm_output():
    r = run_single_agent("what is a roth ira", scripted("I refuse to answer in JSON"), TOOL())
    assert len(r.doc_ids) == 10 and r.errors


def test_orchestrator_skips_planner_for_simple_question():
    llm = scripted()  # any LLM call would raise StopIteration -> test fails
    r = Orchestrator(llm, TOOL()).run("What is short selling?")
    assert r.llm_calls == 0 and r.trace[0]["decided_by"] == "gate:simple" and len(r.doc_ids) == 10


def test_orchestrator_decomposes_and_interleaves_for_coverage():
    llm = scripted(json.dumps({"simple": False, "subqueries": ["short selling mechanics", "mortgage prepayment"]}),
                   json.dumps({"checks": [{"sub": 0, "relevant": True}, {"sub": 1, "relevant": True}]}))
    r = Orchestrator(llm, TOOL()).run(COMPOUND)
    assert r.doc_ids[:4] == ["short0", "mortgage0", "short1", "mortgage1"]
    assert r.llm_calls == 2


def test_orchestrator_partial_failure_still_returns_results():
    """C3 requirement: one sub-agent fails, another times out -> the query still returns."""
    llm = scripted(json.dumps({"simple": False, "subqueries": ["FAIL this one", "SLOW one", "bonds yield"]}),
                   json.dumps({"checks": []}))
    t0 = time.perf_counter()
    r = Orchestrator(llm, TOOL(), worker_timeout_s=0.5).run(COMPOUND)
    assert time.perf_counter() - t0 < 1.9  # did not wait for the 2 s slow worker
    assert r.doc_ids[0] == "bonds0" and len(r.doc_ids) == 10
    assert any("failed" in e for e in r.errors) and any("timed out" in e for e in r.errors)


def test_orchestrator_planner_garbage_falls_back_to_plain_search():
    r = Orchestrator(scripted("not json"), TOOL()).run(COMPOUND)
    assert r.trace[0]["decided_by"] == "planner:failed" and len(r.doc_ids) == 10


def test_orchestrator_respects_call_cap():
    llm = scripted(json.dumps({"simple": False, "subqueries": ["short selling", "mortgage early"]}))
    r = Orchestrator(llm, TOOL(), max_llm_calls=1).run(COMPOUND)
    assert r.llm_calls == 1 and any("verifier" in e for e in r.errors)  # verifier skipped by budget


def test_gate_and_interleave_helpers():
    assert looks_simple("Is gold a good hedge?")
    assert not looks_simple(COMPOUND)
    assert interleave([["a"], ["b", "c"]], k=10) == ["a", "b", "c"]


def _load_ask():
    """scripts/ask.py is a script, not a package module: load it from its path."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ask", Path(__file__).resolve().parents[1] / "scripts" / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    return ask


@pytest.mark.parametrize("strategy", ["bm25", "dense", "hybrid", "single", "multi"])
def test_ask_cli_prints_every_strategy(strategy, monkeypatch, capsys):
    """scripts/ask.py end to end with the fake search and the fake LLM (heuristic policy)."""
    ask = _load_ask()
    monkeypatch.setattr(ask, "make_search_fn", lambda url: fake_search)
    assert ask.main([COMPOUND, "--strategy", strategy, "--backend", "fake", "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "1. " in out and "passage about" in out
    if strategy in ("single", "multi"):
        assert "LLM calls" in out
    if strategy == "multi":
        assert "plan:" in out


def test_ask_cli_refuses_azure_without_confirm_spend():
    ask = _load_ask()
    with pytest.raises(SystemExit):
        ask.main(["q", "--strategy", "single", "--backend", "azure"])
