"""Toy 22: planner decomposition + the free "is it simple?" gate (don't over-orchestrate).

Concept: spend an LLM call on planning only when the question looks compound.
Real code: src/agent/orchestrator.py::looks_simple, plan    Doc: docs/13_multi_agent_orchestration.md
"""
import json
import random

from _common import banner, note, step, try_this

from src.agent.fake_policies import split_question
from src.agent.orchestrator import looks_simple
from src.beir_data import load_split
from src.config import ROOT

banner("Toy 22: who gets decomposed?")
examples = [
    "What is short selling?",
    "Is a Roth IRA better than a 401k for early retirement?",
    "How does short selling work, and should I pay off my mortgage early or invest?",
    "Unvested options after an acquisition: what happens and how are they taxed?",
]
for q in examples:
    simple = looks_simple(q)
    step("q", f"{q}\n     gate says simple={simple} -> " + (
        "skip planner (0 LLM calls)" if simple else f"call planner; naive split would give {split_question(q)}"))
note("The gate is a cheap regex+length rule. It may send a simple question to the planner (wasting one call, "
     "after which the planner can still answer 'simple'), but it never decomposes on its own.")
try:
    queries, _ = load_split("dev")
    plain = random.Random(42).sample(sorted(queries), 100)
    skip_plain = sum(looks_simple(queries[q]) for q in plain)
    comp = [json.loads(l)["question"] for l in open(ROOT / "data" / "compound_dev.jsonl", encoding="utf-8")]
    skip_comp = sum(looks_simple(q) for q in comp)
    step("stats", f"gate skips the planner for {skip_plain}/100 plain dev queries and {skip_comp}/100 compound queries")
    note("Ideal: skip all plain, skip no compound. The gap is what the planner LLM call is paying for.")
except FileNotFoundError:
    print("  (FiQA not downloaded: skipping dataset stats)")
try_this("Tune max_words in looks_simple. Plot skip-rate(plain) vs skip-rate(compound) for 8..20 words.",
         "Example 4 has no 'and'. How should the planner split it? Is it really two needs?",
         "What does decomposition cost when the planner says 'simple'? (1 LLM call, ~150 tokens)")
