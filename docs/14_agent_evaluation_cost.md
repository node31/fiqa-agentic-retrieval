# 14 · Evaluating agents: quality, coverage, latency, cost, and a ship decision

Code: `scripts/agent_eval.py`, `data/build_compound.py`. Results: `results/agent_eval.json` (Azure, final),
`results/agent_eval_dev_copilot.json` (free dev run), `results/agent_eval_dryrun.json` (cost estimate)

An agent has to beat the cheap baseline on the queries it handles by enough to pay for seconds of latency and a per-query
LLM bill. So I measure quality, latency and cost per query type and route on the result.

## Evaluation design

- plain: 100 dev queries, `random.Random(42).sample(sorted(dev_qids), 100)`.
- compound: 100 two-topic questions (`data/compound_dev.jsonl`). I sampled 200 dev queries with seed 42, paired them
  across topics (MiniLM cosine below 0.25, else the least similar partner; mean pair cosine 0.088) and joined each pair
  into one question, written from the two query texts only so no gold wording leaks in. Gold is the union of both halves'
  qrels, each doc tagged with its half.

Systems: hybrid linear α = 0.7 (one search), the single agent ([12](12_single_agent.md)) and the multi-agent
([13](13_multi_agent_orchestration.md)). Metrics: Recall@10, MRR@10, Coverage@10 (compound: at least one gold doc from
each half in the top 10), cost per query, latency p50/p95, LLM calls, tokens and planner decisions.

| backend | cost | used for |
|---|---|---|
| fake | $0 | tests, plumbing, the `--dry-run` estimate |
| copilot (claude-haiku-4.5) | $0 to me | dev evaluation; latency is CLI start-up (~30 to 40 s), so I ignore it |
| azure (Foundry gpt-6-sol) | real $ | final numbers, gated by `--confirm-spend` and `--max-usd`, every call cached |

The real gpt-6-sol prices were unknown, so cost is tokens × placeholder prices ($1 / $4 per 1M input / output,
`PRICE_PER_1M_INPUT` / `PRICE_PER_1M_OUTPUT`). The token counts are the solid part.

```bash
python scripts/agent_eval.py --backend fake                         # 1 min, free, plumbing check only
python scripts/agent_eval.py --backend copilot --limit 30 --workers 8   # free, slow
python scripts/agent_eval.py --backend azure --dry-run              # estimate before spending
python scripts/agent_eval.py --backend azure --confirm-spend --max-usd 5
```

## Final results (Azure gpt-6-sol)

600 system-query runs, 0 errors, ≈ 655K tokens, $0.79 at placeholder prices.

| set | system | R@10 | MRR@10 | Cov@10 | calls/q | tokens/q | $/q | p50 / p95 s |
|---|---|---|---|---|---|---|---|---|
| plain | baseline | 0.502 | 0.485 | - | 0 | 0 | 0 | 0.06 / 0.08 |
| plain | single | 0.547 | 0.642 | - | 1.96 | 2,118 | 0.0025 | 4.09 / 7.27 |
| plain | multi | 0.481 | 0.466 | - | 0.51 | 130 | 0.0002 | 0.03 / 3.59 |
| compound | baseline | 0.204 | 0.365 | 0.10 | 0 | 0 | 0 | 0.08 / 0.10 |
| compound | single | 0.400 | 0.609 | 0.40 | 2.70 | 3,380 | 0.0039 | 5.75 / 9.28 |
| compound | multi | 0.355 | 0.518 | 0.32 | 2.00 | 925 | 0.0013 | 3.40 / 4.75 |

Dev comparison with haiku: plain single 0.494 / 0.432, plain multi 0.476 / 0.472, compound single 0.327 / 0.513 / Cov
0.29, compound multi 0.335 / 0.493 / Cov 0.26. On those numbers I would have shipped only the multi-agent path;
gpt-6-sol made the single agent the quality winner. Even the fake backend, a regex splitter, showed decomposition raising
compound coverage and slightly hurting plain recall, which is why the free gate exists.

At n = 100 a ±0.03 Recall@10 difference is noise. Single minus multi on compound is R@10 +0.045 (bootstrap 95% CI
[+0.007, +0.085]) and Cov@10 +0.08 (CI [+0.01, +0.15]), real but small. I hand-checked 10 of the compound
questions (`data/compound_review_sheet.md`); all 10 read well and none needed a fix.

Cost lessons: the dry run under-estimated by about 40% ($0.47 vs $0.79) because the fake policy makes fewer, shorter calls
than a real model, so I treat it as a lower bound. About 610K of the 655K tokens were input, since prompts re-send search
results, so shorter snippets save more than a lower output cap. Routing is the biggest lever: on plain queries the single
agent spends 2,118 tokens and multi 130, because the free gate skips the LLM.

## Routing decision (DESIGN.md §6.11)

1. The free gate says simple: non-agentic hybrid (~30 ms p50 through the router). The single agent is an opt-in "deep
   search" (+0.045 R@10, +0.157 MRR, at ~4 s p50 and ~2K tokens).
2. Otherwise one planner call; if it says simple, non-agentic results.
3. Compound: the single agent when quality matters (R@10 0.400, Cov@10 0.40, p95 9.3 s, 3,380 tokens), the multi-agent
   under a tight latency or cost budget (p95 4.7 s, about a quarter of the tokens, R@10 0.355, Cov@10 0.32). Both beat one
   search (R@10 0.204, Cov@10 0.10).
