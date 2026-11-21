# 12 · The single agent

Code: `src/agent/single_agent.py`, `src/agent/prompts.py::SINGLE_AGENT`. Tests: `tests/test_agents.py`

My single agent looks at the search results, notices when part of the question is unanswered or the results drifted,
rewrites the query and searches again. Then it picks the best 10 passages from everything it saw.

```
step 0 (free):  results₁ = search(question, "hybrid")                  # ground the first decision
step 1..3:      action = LLM(SYSTEM, question + all results so far)     # exactly 1 LLM call per step
                search  → results_n = search(action.query, action.strategy)   (if steps remain)
                finish  → chosen ids (validated against "seen")
                last step → the prompt says "you MUST finish"
final top-10 = chosen ids  ++  RRF(results₁, results₂, …) fill, deduplicated
```

```mermaid
flowchart TD
  Q["question"] --> S0["search hybrid, no LLM"]
  S0 --> L1{"LLM step 1"}
  L1 -- finish --> F["validate ids + RRF fill"]
  L1 -- "search rewrite" --> S1["search"] --> L2{"LLM step 2"}
  L2 -- finish --> F
  L2 -- search --> S2["search"] --> L3{"LLM step 3: must finish"} --> F
  F --> T["top-10"]
```

The budget is 3 LLM calls (the assignment's cap) and 12K tokens. The model sees 200-character snippets of 10 hits per
search (~550 tokens). Step n re-sends all n result blocks, so tokens grow roughly quadratically, and the prompt says to
search again only if part of the question is missing.

| I chose | because | alternative | why I didn't use it |
|---|---|---|---|
| search before the first LLM call | the LLM decides with evidence and easy queries save a call | LLM writes the first query | an extra call, made blind |
| the model returns doc ids | it acts as a light reranker; ids are validated against what it saw | model only says "done" | no reranking |
| RRF fill over all searches | balanced across rewrites | concatenate lists | the first search would dominate |
| any error ends the loop with what it has | an agent must degrade | raise | one bad call would fail the query |

## Results

Final run on Azure gpt-6-sol (`results/agent_eval.json`, 100 plain + 100 compound dev queries):

| set | system | R@10 | MRR@10 | Cov@10 | calls/q | tokens/q | p50 / p95 s |
|---|---|---|---|---|---|---|---|
| plain | baseline (hybrid) | 0.502 | 0.485 | - | 0 | 0 | 0.06 / 0.08 |
| plain | single agent | 0.547 | 0.642 | - | 1.96 | 2,118 | 4.09 / 7.27 |
| compound | baseline (hybrid) | 0.204 | 0.365 | 0.10 | 0 | 0 | 0.08 / 0.10 |
| compound | single agent | 0.400 | 0.609 | 0.40 | 2.70 | 3,380 | 5.75 / 9.28 |

It was the best system on quality in both sets. On plain queries it improved recall on 7 and lost on none; it used 1, 2
and 3 steps on 29, 46 and 25 of them, and all 3 steps on 72% of compound queries.

For comparison, the free dev run with claude-haiku-4.5 (`results/agent_eval_dev_copilot.json`) gave plain 0.494 / 0.432
and compound 0.327 / 0.513 / Cov 0.29. The gap is the final step, an LLM rerank over 200-character snippets: haiku
lowered plain MRR from 0.485 to 0.432, gpt-6-sol raised it to 0.642. Whether that step helps depends on the model, so I
re-run the eval when the model changes.

Routing (DESIGN.md §6.11): plain traffic stays on non-agentic hybrid, with the single agent as an opt-in "deep search" at
about 4 s p50 and 2K tokens per query. For compound questions it is the quality choice, and the multi-agent path
([13](13_multi_agent_orchestration.md)) takes over under a tight latency or cost budget.

Not built: skipping the LLM when the first search's top score is high, a smaller decision model, shorter snippets.
