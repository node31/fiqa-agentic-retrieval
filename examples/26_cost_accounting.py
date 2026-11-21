"""Toy 26: cost accounting. Tokens to dollars, why agent prompts grow, and what caching saves.

Concept: estimate BEFORE you spend; per-query cost = sum over LLM calls of (in x p_in + out x p_out).
Real code: src/agent/llm.py (cost_usd, estimate_tokens, cache), scripts/agent_eval.py --dry-run
Doc: docs/topics/14_agent_evaluation_cost.md
"""
from _common import banner, note, step, try_this

from src.agent import prompts
from src.agent.llm import cost_usd, estimate_tokens
from src.config import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT

banner("Toy 26: what does one agentic query cost?")
step(1, f"Prices (placeholders in .env): ${PRICE_PER_1M_INPUT}/1M input, ${PRICE_PER_1M_OUTPUT}/1M output tokens")
snip = 10 * (200 + 10)  # 10 hits x (200-char snippet + id)
single = [("step 1", len(prompts.SINGLE_AGENT) + snip, 60), ("step 2", len(prompts.SINGLE_AGENT) + 2 * snip, 60),
          ("step 3", len(prompts.SINGLE_AGENT) + 3 * snip, 120)]
step(2, "Single agent, worst case (3 steps; each re-sends ALL earlier results):")
tot_in = tot_out = 0
for name, chars, out in single:
    tin = estimate_tokens("x" * chars)
    tot_in, tot_out = tot_in + tin, tot_out + out
    print(f"     {name}: ~{tin:5d} in + {out:3d} out tokens")
print(f"     total ~{tot_in} in + {tot_out} out = ${cost_usd(tot_in, tot_out):.5f}/query")
multi_in = estimate_tokens(prompts.PLANNER) + 40 + estimate_tokens(prompts.VERIFIER) + estimate_tokens("x" * 2 * 5 * 210)
multi_usd, single_usd = cost_usd(multi_in, 250), cost_usd(tot_in, tot_out)
step(3, f"Multi-agent: planner (~{estimate_tokens(prompts.PLANNER) + 40} in) + verifier on top-5 x 2 subs "
        f"(~{multi_in - estimate_tokens(prompts.PLANNER) - 40} in): ${cost_usd(multi_in, 250):.5f}/query")
note(f"The multi-agent system is {'CHEAPER' if multi_usd < single_usd else 'MORE EXPENSIVE'} per query than the "
     f"single agent here (${multi_usd:.5f} vs ${single_usd:.5f}, {single_usd / multi_usd:.1f}x): 2 LLM calls vs 3, and "
     "the verifier sees only top-5 snippets once, while the single agent re-reads its growing transcript. "
     "Structure beats verbosity.")
step(4, "Scale: 200 eval queries x 2 agentic systems x 2 runs (dev + debrief)")
print(f"     ~${200 * 2 * (single_usd + multi_usd):.2f} at worst case (both runs uncached)")
print(f"     ~${200 * (single_usd + multi_usd):.2f} if the 2nd run repeats the 1st: the cache makes an identical re-run free")
note("Hidden costs: reasoning models bill 'reasoning tokens' as output, often 2-5x the visible answer. Always run "
     "`scripts/agent_eval.py --backend azure --dry-run` first and set --max-usd.")
try_this("Cut snippet length from 200 to 100 chars. Re-run the agent eval: does recall drop?",
         "Compute the break-even: how much Recall@10 gain per $ makes an agent worth shipping?",
         "Why must a cache key include max_output_tokens and the model name?")
