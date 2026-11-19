# ch08: An agent loop yourself (with a scripted fake LLM: free, instant, deterministic)

Guide: `docs/from_zero/08_part_c_agents.md`.

> 🎯 **Why:** an "agent" sounds magical, but it's a loop in *your* code. Building it against a scripted fake LLM
> teaches you the guardrails (bad JSON, invented ids, step caps) *before* you spend a cent on a real model.
> Those guardrails are what make an agent safe to ship.

## Steps
1. `PAPER.md`.
2. Implement in `agent_loop.py`, in order: `parse_action` → `interleave` → `run_agent`.
3. `python -m pytest exercises\ch08_agents`.
4. `python exercises\ch08_agents\agent_loop.py` runs a scripted conversation and prints every step.
5. **Optional (real LLM):** swap `ScriptedLLM` for a function that calls a real model (your Azure deployment, via the
   `openai` package). Your loop must not change at all. That's the point of the interface.

## The contract of `run_agent` (what the checker expects)
1. **Step 0 (free):** `search_fn(question)` before any LLM call; remember the results.
2. Up to `max_steps` LLM calls. Each prompt includes the question and all results so far.
   The reply must contain one JSON object:
   `{"action": "search", "query": "..."}` or `{"action": "finish", "doc_ids": [...]}`.
3. `search` with steps left → run it, append the results, continue. On the **last** step, treat anything as finish.
4. `finish` → keep only ids the agent has **seen** in some search (drop invented ones), dedupe.
5. Unparseable reply or unknown action → record an error string and stop looping (no exception!).
6. Fill the remaining slots up to `k` with `interleave(all search result lists)`, skipping duplicates.
7. Return `{"doc_ids": [...], "llm_calls": int, "searches": int, "errors": [str, ...]}`.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch08_agents` | JSON parsing, interleave, and the 7-point contract against scripted LLM scenarios | seconds |
| full | `python -m pytest exercises\ch08_agents --full` | your `run_agent` on **100 real two-part questions** (`data/compound_dev.jsonl`) over the full corpus, with a scripted "planner" that searches each half: Coverage@10 must clearly beat a single search | ~30 s |

The full check uses a scripted LLM, so it's still free. To try a *real* LLM, replace `ScriptedLLM` with a function that
calls your model. Your loop shouldn't change.
