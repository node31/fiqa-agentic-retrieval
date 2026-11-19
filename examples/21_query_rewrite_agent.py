"""Toy 21: the real single agent (src/agent/single_agent.py) on a toy corpus with a fake LLM.

Concept: grounded first search, <=3 LLM steps, query rewrite, validated ids, RRF fill.
Doc: docs/12_single_agent.md
"""
import json

from _common import TOY_DOCS, banner, note, step, try_this

from src.agent.llm import FakeLLM
from src.agent.single_agent import run_single_agent
from src.agent.tools import SearchTool, render_hits
from src.retrieval.bm25 import BM25Index, tokenize

banner("Toy 21: single agent = 1 free search + up to 3 LLM decisions")
ix = BM25Index().build(list(TOY_DOCS), list(TOY_DOCS.values()))
tool = SearchTool(lambda q, s, k: [{"doc_id": d, "score": sc, "text": TOY_DOCS[d]}
                                   for d, sc in ix.search(list(dict.fromkeys(tokenize(q))), k)], k=5)
q = "How does shorting work, and is paying off my mortgage early smart?"
step(0, f"Question: {q}\n    Step 0 (no LLM): search(question) ->\n" + render_hits(tool(q))[:400])
seen_prompts = []


def model(system, user):
    seen_prompts.append(user)
    if user.count("Search ") == 1:  # only the original search so far -> cover the 2nd topic
        return json.dumps({"action": "search", "query": "mortgage early payoff return", "strategy": "hybrid"})
    return json.dumps({"action": "finish", "doc_ids": ["d1", "d3", "INVENTED-ID"]})


r = run_single_agent(q, FakeLLM(model), tool, k=5)
for t in r.trace:
    step(t["step"], f"LLM action: {t['action']}")
step("final", f"{r.doc_ids}  (llm_calls={r.llm_calls}, searches={r.searches}, tokens~{r.input_tokens + r.output_tokens})")
note("'INVENTED-ID' was dropped: the agent may only return ids it has seen. The remaining slots were filled by "
     "RRF over both searches, so the mortgage passages AND the shorting passages are present: coverage.")
note(f"The 2nd LLM call's prompt was {len(seen_prompts[-1])} chars vs {len(seen_prompts[0])} for the 1st: every "
     "step re-sends all previous results. That is where agent token costs come from.")
try_this("Make the model always reply 'search'. Confirm the loop stops after 3 LLM calls (forced finish).",
         "Change step 0 to NOT search first. How many LLM calls does a simple question now need?",
         "Run scripts/agent_eval.py --backend fake --systems single --limit 10 and read a trace.")
