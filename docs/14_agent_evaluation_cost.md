# 14 · Evaluating agents: quality, coverage, latency, cost, and a ship decision

**Code:** `scripts/agent_eval.py`, `data/build_compound.py` · **Toys:** 22, 25, 26
**Results:** `results/agent_eval_dev_*.json` (free dev backends), `results/agent_eval.json` (Azure)

## 1. Intuition

An agent is a *product decision*, not just a model trick. It must beat the cheap baseline on the
queries it's used for, by enough to justify extra latency (seconds, not milliseconds) and money
(per-query LLM cost). So you measure all three, **per query type**, and route accordingly.

## 2. The evaluation design

**Two sets**
- *plain*: 100 dev queries, `random.Random(42).sample(sorted(dev_qids), 100)`.
- *compound*: 100 two-topic questions (`data/compound_dev.jsonl`): 200 dev queries sampled with seed 42,
  paired across topics (MiniLM cosine < 0.25, mean pair cosine 0.088), joined into one natural
  question, with gold = union of both halves' qrels, **each gold doc tagged with its half**.

**Three systems**: non-agentic best (hybrid linear α = 0.7, one search), single agent, multi-agent.

**Metrics**: Recall@10, MRR@10, Coverage@10 (compound: ≥ 1 gold from *each* half in the top 10), cost/query,
latency p50/p95, average LLM calls (+ tokens, searches, error count, planner decisions).

## 3. Diagram

```
for each set × system: ThreadPool(workers) over queries → AgentResult(doc_ids, calls, tokens, latency, trace, errors)
                        → summarize → results/agent_eval*.json (summary + per-query traces)
```

## 4. Cost discipline (`docs/adr/0011-llm-backends-and-cost.md`)

| backend | cost | used for |
|---|---|---|
| fake | $0 | tests, toys, plumbing check, `--dry-run` estimate |
| copilot (GitHub Copilot CLI, claude-haiku-4.5) | $0 to us | real-LLM dev evaluation; its latency is dominated by CLI start-up (~30–40 s), so **ignore its latency** |
| azure (Foundry gpt-6-sol) | real $ | the final numbers. Gated by `--confirm-spend` and `--max-usd`, every call cached |

Prices for gpt-6-sol were unknown, so `PRICE_PER_1M_INPUT/OUTPUT` are placeholders in `.env`. Cost
columns are "tokens × your prices", and the token counts are the robust part.

## 5. Run it

```bash
python scripts/agent_eval.py --backend fake                         # 1 min, free
python scripts/agent_eval.py --backend copilot --limit 30 --workers 8   # free, slow
python scripts/agent_eval.py --backend azure --dry-run              # estimate before spending
python scripts/agent_eval.py --backend azure --confirm-spend --max-usd 5
```

## 6. Observe it

Numbers and the ship/don't-ship decision are in DESIGN.md §7. The fake-backend run is informative
by itself, because its "LLM" is just a regex splitter. On compound queries, decomposition alone takes
Coverage@10 from 0.10 (one search) to 0.31 (multi-agent). On plain queries it slightly *hurts* recall
(0.502 → 0.476), because it splits questions that should not be split. That is why the gate and the routing policy exist.

**The real run (Azure gpt-6-sol, `results/agent_eval.json`).** 600 system-query runs, 0 errors, ≈ 655K tokens,
**$0.79 at placeholder prices** ($1 / $4 per 1M input / output tokens).

| set | system | R@10 | MRR@10 | Cov@10 | calls/q | tokens/q | $/q | p50 / p95 s |
|---|---|---|---|---|---|---|---|---|
| plain | baseline | 0.502 | 0.485 | – | 0 | 0 | 0 | 0.06 / 0.08 |
| plain | single | 0.547 | 0.642 | – | 1.96 | 2,118 | 0.0025 | 4.09 / 7.27 |
| plain | multi | 0.481 | 0.466 | – | 0.51 | 130 | 0.0002 | 0.03 / 3.59 |
| compound | baseline | 0.204 | 0.365 | 0.10 | 0 | 0 | 0 | 0.08 / 0.10 |
| compound | single | 0.400 | 0.609 | 0.40 | 2.70 | 3,380 | 0.0039 | 5.75 / 9.28 |
| compound | multi | 0.355 | 0.518 | 0.32 | 2.00 | 925 | 0.0013 | 3.40 / 4.75 |

Three cost lessons from this run:
- **The dry run under-estimated by ~40%** ($0.47 estimated vs $0.79 actual). `--dry-run` replays the *fake* policy, which
  makes fewer and shorter calls than a real model (the real single agent averaged 2.70 calls on compound queries). Treat a
  dry-run estimate as a lower bound and set `--max-usd` with headroom.
- **Input tokens dominate.** Roughly 610K of the 655K tokens were input: prompts re-send search results, while replies are a
  short JSON. Shorter snippets and fewer results per step cut cost more than a lower output cap does.
- **Cost per useful query.** The single agent costs ~19× multi on plain queries (2,118 vs 130 tokens), because multi's free
  gate skips the LLM for most of them. Routing is the biggest cost lever.

## 7. Pitfalls and interview questions

- **Gold leakage into rewrites**: the compound rewrites must not add words from gold passages. Ours were written
  from the two query texts only.
- **Same-topic pairs** make compound queries trivially answerable by one search. Hence the cosine filter.
- **Small n**: with 100 queries, a ±0.03 Recall@10 difference is within noise (bootstrap it! exercise 3).
- **Cost per query vs. cost per *useful* query**: if 90% of traffic is simple, route it to the cheap path.

## 8. Exercises

1. (Concept) Why is Coverage@10 a better primary metric than Recall@10 for compound questions?
2. (Predict) Would a 7/3 slot split help Coverage@10 or hurt it? When?
3. (Do) Bootstrap 95% confidence intervals for Recall@10 differences between systems (resample queries 1,000×).
4. (Do) Build a compound *test* set: `python data/build_compound.py --split test` (Project FAQaration). Do NOT tune on it.

Solutions: [solutions/14_agent_eval.md](solutions/14_agent_eval.md)

## 9. Further reading

- Anthropic, *Building effective agents* (2024): start simple, add agents only when measured
- Zaharia et al., *The Shift from Models to Compound AI Systems* (2024)
- Efron & Tibshirani, *An Introduction to the Bootstrap*
