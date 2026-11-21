"""Toy 20: a hand-written tool-calling loop with a scripted fake LLM. No framework, no magic.

Concept: an "agent" = while loop: ask LLM -> parse action -> run tool -> append observation -> repeat.
Real code: src/agent/single_agent.py, src/agent/llm.py::parse_json   Doc: docs/topics/11_llm_tool_calling.md
"""
import json

from _common import TOY_DOCS, banner, note, step, try_this

from src.agent.llm import FakeLLM, parse_json
from src.retrieval.bm25 import BM25Index, tokenize

banner("Toy 20: the agent loop, step by step")
index = BM25Index().build(list(TOY_DOCS), list(TOY_DOCS.values()))


def search(query: str) -> list[str]:  # the TOOL: plain Python
    return [d for d, _ in index.search(list(dict.fromkeys(tokenize(query))), 3)]


SCRIPT = [  # what a model might reply, turn by turn
    'Let me look that up.\n```json\n{"action": "search", "query": "short selling risk"}\n```',
    '{"action": "search", "query": "borrow fee shorting"}',
    '{"action": "finish", "doc_ids": ["d2", "d10", "d1"]}',
]
llm = FakeLLM(lambda system, user: SCRIPT[user.count("Observation")])
SYSTEM = ('Tools: search(query). Reply with ONE JSON: {"action":"search","query":...} '
          'or {"action":"finish","doc_ids":[...]}')
transcript = "Question: what are the risks and costs of shorting?"
for turn in range(1, 4):
    reply = llm.complete(SYSTEM, transcript).text
    step(turn, f"LLM replied: {reply!r}")
    action = parse_json(reply)  # robust to ```json fences and chatter
    print(f"     parsed action: {action}")
    if action["action"] == "finish":
        print(f"     FINAL ANSWER: {action['doc_ids']}")
        break
    obs = search(action["query"])
    print(f"     tool search({action['query']!r}) -> {obs}")
    transcript += f"\nObservation {turn}: {json.dumps(obs)}"
note("That's all an agent is. 'Native function calling' in the OpenAI or Anthropic APIs is the same loop. The "
     "provider just enforces the JSON schema and returns the call in a structured field instead of text.")
note("Things a real loop needs that this toy skips: a step cap (we use 3), a token budget, validation of "
     "tool arguments and of returned ids, and a fallback when the reply is not valid JSON.")
try_this("Make the 2nd scripted reply invalid JSON. What should the loop do? Compare with run_single_agent.",
         "Add a 'strategy' argument to the tool and validate it (see src/agent/tools.py).",
         "Count the characters of `transcript` per turn. Why does cost grow ~quadratically with steps?")
