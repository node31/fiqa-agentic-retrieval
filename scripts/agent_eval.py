"""Agent evaluation (Part C4): non-agentic vs single agent vs multi-agent.

    python scripts/agent_eval.py --backend fake                      # free plumbing check
    python scripts/agent_eval.py --backend copilot --limit 20        # free dev numbers (slow)
    python scripts/agent_eval.py --backend azure --dry-run           # estimate tokens/$ first
    python scripts/agent_eval.py --backend azure --confirm-spend     # the real run (costs money)

SETS
    plain     100 dev queries sampled with seed 42 (``random.Random(42).sample(sorted(dev), 100)``)
    compound  data/compound_dev.jsonl (100 two-topic questions, gold tagged by half)

METRICS (per set x system)
    Recall@10, MRR@10, Coverage@10 (compound only), cost/query (USD, at config prices),
    latency p50/p95 (s), avg LLM calls, avg tokens, error count, planner decisions.

SAFETY
    * azure requires --confirm-spend, and --max-usd (default 2.0) aborts the run once the
      running estimated spend crosses it.
    * Every LLM response is cached, so re-running costs nothing.
Writes results/agent_eval.json (azure) or results/agent_eval_dev_<backend>.json.
"""
import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.llm import cost_usd, get_llm  # noqa: E402
from src.agent.orchestrator import Orchestrator  # noqa: E402
from src.agent.single_agent import AgentResult, run_single_agent  # noqa: E402
from src.agent.tools import SearchTool, http_search_fn, local_search_fn  # noqa: E402
from src.beir_data import load_split  # noqa: E402
from src.config import RESULTS_DIR, SEED  # noqa: E402
from src.metrics import coverage_at_k, mrr_at_k, recall_at_k  # noqa: E402
from src.util import percentile  # noqa: E402


def load_sets(limit=None):
    queries, qrels = load_split("dev")
    plain_ids = random.Random(SEED).sample(sorted(queries), 100)
    plain = [{"id": q, "question": queries[q], "gold": {d: "a" for d in qrels[q]}} for q in plain_ids]
    compound = [json.loads(l) for l in (ROOT / "data" / "compound_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"plain": plain[:limit], "compound": compound[:limit]}


class Spend:
    def __init__(self, max_usd):
        self.max_usd, self.usd, self.lock = max_usd, 0.0, threading.Lock()

    def add(self, r: AgentResult):
        with self.lock:
            self.usd += r.cost_usd
            if self.usd > self.max_usd:
                raise SystemExit(f"ABORT: estimated spend ${self.usd:.2f} exceeded --max-usd {self.max_usd}")


def run_system(name, item, llm, tool, orch):
    q = item["question"]
    if name == "baseline":
        t0 = time.perf_counter()
        ids = [h["doc_id"] for h in tool(q, "hybrid")]
        return AgentResult(ids, path="baseline", searches=1, latency_s=time.perf_counter() - t0)
    if name == "single":
        return run_single_agent(q, llm, tool)
    return orch.run(q)


def summarize(set_name, items, results):
    gold = [it["gold"] for it in items]
    rel = [{d: 1 for d in g} for g in gold]
    out = {
        "n": len(items),
        "recall@10": sum(recall_at_k(r.doc_ids, g) for r, g in zip(results, rel)) / len(items),
        "mrr@10": sum(mrr_at_k(r.doc_ids, g) for r, g in zip(results, rel)) / len(items),
        "cost_per_query_usd": sum(r.cost_usd for r in results) / len(items),
        "latency_p50_s": percentile([r.latency_s for r in results], 50),
        "latency_p95_s": percentile([r.latency_s for r in results], 95),
        "avg_llm_calls": sum(r.llm_calls for r in results) / len(items),
        "avg_tokens": sum(r.input_tokens + r.output_tokens for r in results) / len(items),
        "avg_searches": sum(r.searches for r in results) / len(items),
        "queries_with_errors": sum(bool(r.errors) for r in results),
    }
    if set_name == "compound":
        halves = [{h: [d for d, x in g.items() if x == h] for h in ("a", "b")} for g in gold]
        out["coverage@10"] = sum(coverage_at_k(r.doc_ids, hv) for r, hv in zip(results, halves)) / len(items)
    decisions = Counter(t["decided_by"] for r in results for t in r.trace if "decided_by" in t)
    if decisions:
        out["planner_decisions"] = dict(decisions)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake", choices=["fake", "copilot", "azure"])
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens/cost with the fake policy; no LLM calls")
    ap.add_argument("--limit", type=int, help="queries per set")
    ap.add_argument("--systems", default="baseline,single,multi")
    ap.add_argument("--sets", default="plain,compound")
    ap.add_argument("--search", choices=["local", "service"], default="local")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    llm = get_llm("fake" if a.dry_run else a.backend, confirm_spend=a.confirm_spend)
    if a.search == "local":
        from src.service.router import single_node_router

        tool = SearchTool(local_search_fn(single_node_router()))
    else:
        tool = SearchTool(http_search_fn(a.url))
    orch = Orchestrator(llm, tool)
    spend = Spend(a.max_usd)
    sets = load_sets(a.limit)
    report = {"backend": a.backend, "model": getattr(llm, "model", "") or a.backend, "dry_run": a.dry_run,
              "search": a.search, "plain_sample_seed": SEED, "results": {}, "per_query": {}}
    t_start = time.time()
    for set_name in a.sets.split(","):
        items = sets[set_name]
        for system in a.systems.split(","):
            def one(it, system=system):
                r = run_system(system, it, llm, tool, orch)
                spend.add(r)
                return r
            with ThreadPoolExecutor(a.workers) as ex:
                results = list(ex.map(one, items))
            summ = summarize(set_name, items, results)
            report["results"].setdefault(set_name, {})[system] = summ
            report["per_query"][f"{set_name}/{system}"] = [
                {"id": it["id"], "doc_ids": r.doc_ids, "llm_calls": r.llm_calls, "errors": r.errors,
                 "trace": r.trace} for it, r in zip(items, results)]
            cov = f" Cov@10={summ['coverage@10']:.3f}" if "coverage@10" in summ else ""
            print(f"{set_name:8s} {system:8s} R@10={summ['recall@10']:.3f} MRR={summ['mrr@10']:.3f}{cov} "
                  f"calls={summ['avg_llm_calls']:.2f} tok={summ['avg_tokens']:.0f} "
                  f"$/q={summ['cost_per_query_usd']:.5f} p50/p95={summ['latency_p50_s']:.2f}/{summ['latency_p95_s']:.2f}s "
                  f"errors={summ['queries_with_errors']}", flush=True)
    report["total_estimated_usd"] = round(spend.usd, 4)
    report["wall_s"] = round(time.time() - t_start, 1)
    if a.dry_run:
        n = sum(len(v) for v in sets.values())
        print(f"\nDRY RUN: estimated total ${spend.usd:.4f} for {n} queries x systems at "
              f"config prices (reasoning tokens NOT included; a reasoning model may cost 2-5x more).")
    name = "agent_eval.json" if a.backend == "azure" and not a.dry_run else (
        f"agent_eval_dryrun.json" if a.dry_run else f"agent_eval_dev_{a.backend}.json")
    out = a.out or RESULTS_DIR / name
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
