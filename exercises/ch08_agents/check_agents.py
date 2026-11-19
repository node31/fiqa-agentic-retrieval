"""Checker for ch08. Don't edit. Run: python -m pytest exercises/ch08_agents"""
import pytest

import agent_loop as mine

INDEX = {"short": ["d1", "d10", "d2"], "mortgage": ["d3", "d9", "d8"], "borrow": ["d10", "d1"]}
Q = "How does shorting work, and should I pay off my mortgage early?"


def search(q):
    out = []
    for word, ids in INDEX.items():
        if word in q.lower():
            out += [i for i in ids if i not in out]
    return out


# ---- parse_action ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,want", [
    ('{"action": "search", "query": "x"}', {"action": "search", "query": "x"}),
    ('Sure!\n```json\n{"action": "finish", "doc_ids": []}\n```', {"action": "finish", "doc_ids": []}),
    ('thinking... {"action": "search", "query": "a {b} c"} done', {"action": "search", "query": "a {b} c"}),
    ('{"a": {"b": 1}} {"c": 2}', {"a": {"b": 1}}),
])
def check_parse_action(text, want):
    assert mine.parse_action(text) == want


@pytest.mark.parametrize("bad", ["Ready.", "", "{not json", "{'single': 'quotes'}"])
def check_parse_action_rejects(bad):
    with pytest.raises(ValueError):
        mine.parse_action(bad)


# ---- interleave ---------------------------------------------------------------------------------
def check_interleave_doc_example():
    assert mine.interleave([["a1", "a2", "a3"], ["b1", "a1", "b2"]], k=4) == ["a1", "b1", "a2", "a3"]


def check_interleave_gives_both_halves_slots():
    a = [f"a{i}" for i in range(10)]
    b = [f"b{i}" for i in range(10)]
    got = mine.interleave([a, b], k=10)
    assert got == ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"]


def check_interleave_uneven_and_empty():
    assert mine.interleave([["a"], [], ["c1", "c2"]], k=10) == ["a", "c1", "c2"]
    assert mine.interleave([], k=3) == []


# ---- run_agent -----------------------------------------------------------------------------------
def check_happy_path():
    llm = mine.ScriptedLLM(['{"action": "search", "query": "mortgage early payoff"}',
                            '{"action": "finish", "doc_ids": ["d3", "d1"]}'])
    res = mine.run_agent(Q, llm, search, k=5)
    assert res["doc_ids"][:2] == ["d3", "d1"], "the agent's own picks come first"
    assert len(res["doc_ids"]) == 5 and len(set(res["doc_ids"])) == 5, "fill to k with no duplicates"
    assert res["llm_calls"] == 2 and res["searches"] == 2 and res["errors"] == []


def check_first_search_happens_before_any_llm_call():
    llm = mine.ScriptedLLM(['{"action": "finish", "doc_ids": ["d1"]}'])
    mine.run_agent("what is short selling", llm, search, k=3)
    assert "d1" in llm.prompts[0], "the first prompt must already contain the step-0 search results"


def check_invented_ids_are_dropped():
    llm = mine.ScriptedLLM(['{"action": "finish", "doc_ids": ["INVENTED", "d2"]}'])
    res = mine.run_agent("what is short selling", llm, search, k=3)
    assert "INVENTED" not in res["doc_ids"] and res["doc_ids"][0] == "d2"


def check_bad_json_degrades_gracefully():
    llm = mine.ScriptedLLM(["Ready."])
    res = mine.run_agent("what is short selling", llm, search, k=3)
    assert res["errors"], "record the parse failure in errors"
    assert res["doc_ids"] == ["d1", "d10", "d2"], "fall back to the step-0 results, never return nothing"


def check_step_cap_forces_finish():
    llm = mine.ScriptedLLM(['{"action": "search", "query": "short borrow"}'])  # never says finish
    res = mine.run_agent("what is short selling", llm, search, k=3, max_steps=3)
    assert res["llm_calls"] == 3, "exactly max_steps LLM calls, then stop"
    assert res["searches"] <= 3 and len(res["doc_ids"]) == 3


# ---- FULL check (python -m pytest exercises\\ch08_agents --full): 100 real two-part questions ------------------
@pytest.mark.full
def check_full_decomposition_raises_coverage():
    """100 compound dev questions ("X, and Y?") over the full corpus, with the project's hybrid search as the tool.
    Baseline: one search for the whole question (the project measured Coverage@10 = 0.10). Agent: YOUR run_agent with a
    scripted 'planner' that searches each half, then finishes. Interleaving the halves must cover BOTH halves far more
    often. (The project's real multi-agent system reached 0.26; this scripted planner splits perfectly, so ~0.3.)"""
    import json as _json

    import fullkit
    from src.metrics import coverage_at_k

    search = fullkit.hybrid_search

    base = agent = 0.0
    recs = fullkit.compound(100)
    for rec in recs:
        halves = fullkit.halves(rec)
        base += coverage_at_k(search(rec["question"]), halves)
        llm = mine.ScriptedLLM([_json.dumps({"action": "search", "query": rec["a"]["text"]}),
                                _json.dumps({"action": "search", "query": rec["b"]["text"]}),
                                _json.dumps({"action": "finish", "doc_ids": []})])
        out = mine.run_agent(rec["question"], llm, search)
        assert out["llm_calls"] == 3 and out["searches"] == 3 and len(out["doc_ids"]) <= 10
        agent += coverage_at_k(out["doc_ids"], halves)
    base, agent = base / len(recs), agent / len(recs)
    print(f"\nCoverage@10 on {len(recs)} compound questions: one search {base:.2f} -> agent {agent:.2f}")
    assert agent >= base + 0.10, f"decomposition should clearly help: {base:.2f} -> {agent:.2f}"
