"""Print the Part C results table (markdown) from an agent-eval JSON, for DESIGN.md section 7.

    python scripts/render_agent_table.py results/agent_eval_dev_copilot.json
    python scripts/render_agent_table.py results/agent_eval.json          # after the Azure run
"""
import json
import sys
from pathlib import Path


def main(path: str) -> None:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"backend `{j['backend']}`, model `{j['model']}`, total est. spend ${j.get('total_estimated_usd', 0):.4f}\n")
    print("| set | system | R@10 | MRR@10 | Cov@10 | LLM calls/q | tokens/q | $/q | p50 / p95 s | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, systems in j["results"].items():
        for name, r in systems.items():
            cov = f"{r['coverage@10']:.3f}" if "coverage@10" in r else "-"
            print(f"| {s} | {name} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} | {cov} | {r['avg_llm_calls']:.2f} | "
                  f"{r['avg_tokens']:.0f} | {r['cost_per_query_usd']:.5f} | {r['latency_p50_s']:.2f} / "
                  f"{r['latency_p95_s']:.2f} | {r['queries_with_errors']} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/agent_eval.json")
