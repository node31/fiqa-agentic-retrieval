"""Agent evaluation (Part C4): non-agentic vs single agent vs multi-agent.

    python scripts/agent_eval.py --backend fake                      # free plumbing check
    python scripts/agent_eval.py --backend copilot --limit 20        # free dev numbers (slow)
    python scripts/agent_eval.py --backend azure --dry-run           # estimate tokens/$ first
    python scripts/agent_eval.py --backend azure --confirm-spend     # the real run (costs money)
    python scripts/agent_eval.py --backend azure --confirm-spend --systems cascade --out results/agent_eval_cascade.json

SYSTEMS
    baseline  one hybrid search, no LLM
    single    the single agent (src/agent/single_agent.py)
    multi     planner + workers + verifier (src/agent/orchestrator.py)
    cascade   cheap tier first, strong tier only when it escalates (src/agent/cascade.py); the summary adds
              the escalation rate and reasons and the tokens each tier used

SETS
    plain     100 dev queries sampled with seed 42 (``random.Random(42).sample(sorted(dev), 100)``)
    compound  data/compound_dev.jsonl (100 two-topic questions, gold tagged by half)
    Needs the REAL FiQA data: unset FIQA_DATA_DIR / ARTIFACTS_DIR / RESULTS_DIR if a toy run set them.

WHERE THE AGENTS' SEARCHES RUN (--search)
    local    (default) NOT sharded: ``single_node_router()`` = the whole corpus in ONE in-process
             ShardStore, same Router code with 1 shard and no HTTP. Fast to start; quality is
             identical to the sharded service (svc3 matched single-node nDCG exactly).
    service  the real sharded system over HTTP: router + 3 shards, e.g. the Docker stack
             (``--search service --url http://127.0.0.1:8000``). Adds the network hop to latency.

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
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.cascade import cascade_llms, default_tiers, run_cascade  # noqa: E402
from src.agent.llm import cost_usd, get_llm  # noqa: E402
from src.agent.orchestrator import Orchestrator  # noqa: E402
from src.agent.single_agent import AgentResult, run_single_agent  # noqa: E402
from src.agent.tools import SearchTool, http_search_fn, local_search_fn  # noqa: E402
from src.beir_data import load_split  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR, SEED  # noqa: E402
from src.metrics import coverage_at_k, mrr_at_k, recall_at_k  # noqa: E402
from src.util import percentile  # noqa: E402


def load_sets(limit=None):
    queries, qrels = load_split("dev")
    # PITFALL: a shell left over from the toy runs (ch06) may still have FIQA_DATA_DIR pointing at
    # the 24-doc fixture (4 dev queries). Refuse clearly instead of crashing in random.sample, and
    # never shrink the sample: the compound set's gold docs are REAL FiQA ids, absent from a toy corpus.
    if len(queries) < 100:
        sys.exit(f"agent_eval needs the real FiQA dev split (500 queries), but {DATA_DIR} has only {len(queries)}.\n"
                 f"FIQA_DATA_DIR={os.environ.get('FIQA_DATA_DIR', '(unset)')} - probably left over from a toy run. Fix:\n"
                 "  PowerShell: Remove-Item Env:FIQA_DATA_DIR, Env:ARTIFACTS_DIR, Env:RESULTS_DIR -ErrorAction SilentlyContinue\n"
                 "  bash:       unset FIQA_DATA_DIR ARTIFACTS_DIR RESULTS_DIR")
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


def run_system(name, item, llm, tool, orch, cascade=None):
    q = item["question"]
    if name == "baseline":
        t0 = time.perf_counter()
        ids = [h["doc_id"] for h in tool(q, "hybrid")]
        return AgentResult(ids, path="baseline", searches=1, latency_s=time.perf_counter() - t0)
    if name == "single":
        return run_single_agent(q, llm, tool)
    if name == "cascade":
        return cascade(q)
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
        "avg_provider_cached_input_tokens": sum(r.cached_input_tokens for r in results) / len(items),
    }
    if any(r.path == "cascade" for r in results):
        cheap = [r.trace[0] for r in results]
        esc = [c for c in cheap if c.get("escalate")]
        out["escalation_rate"] = len(esc) / len(items)
        out["semantic_cache_hits"] = sum(c["tier"] == "cache" for c in cheap)
        out["escalation_reasons"] = dict(Counter(c["escalate"] for c in esc))
        for tier in ("cheap", "strong"):
            runs = [t for r in results for t in r.trace if t["tier"] == tier and not t.get("shadow")]
            out[f"{tier}_tier_avg_tokens"] = sum(t["input_tokens"] + t["output_tokens"] for t in runs) / len(items)
            out[f"{tier}_tier_avg_input_tokens"] = sum(t["input_tokens"] for t in runs) / len(items)
            out[f"{tier}_tier_avg_output_tokens"] = sum(t["output_tokens"] for t in runs) / len(items)
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
    ap.add_argument("--search", choices=["local", "service"], default="local",
                    help="local = single in-process node (NOT sharded, default); service = the sharded router at --url")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("FIQA_TOKEN"), help="bearer token for --search service")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cascade-shadow", action="store_true",
                    help="cascade: also run the tier that did not run (outside the totals), for scripts/cascade_policies.py")
    ap.add_argument("--no-route", action="store_true", help="cascade: send compound questions to the cheap tier too")
    a = ap.parse_args()

    llm = get_llm("fake" if a.dry_run else a.backend, confirm_spend=a.confirm_spend)
    if a.search == "local":
        from src.service.router import single_node_router

        tool = SearchTool(local_search_fn(single_node_router()))
    else:
        tool = SearchTool(http_search_fn(a.url, a.token))
    orch = Orchestrator(llm, tool)
    cascade = None
    if "cascade" in a.systems.split(","):
        # The strong tier IS the agent's usual LLM; the cheap one may be another backend/deployment
        # (CASCADE_CHEAP_BACKEND / CASCADE_CHEAP_DEPLOYMENT); by default the same.
        cheap_llm = llm if a.dry_run else cascade_llms(a.backend, a.confirm_spend)[0]
        tiers = default_tiers(cheap_llm, llm)

        def cascade(q):
            return run_cascade(q, *tiers, tool, route_compound=not a.no_route, shadow=a.cascade_shadow)
    spend = Spend(a.max_usd)
    sets = load_sets(a.limit)
    report = {"backend": a.backend, "model": getattr(llm, "model", "") or a.backend, "dry_run": a.dry_run,
              "search": a.search, "plain_sample_seed": SEED, "results": {}, "per_query": {}}
    t_start = time.time()
    for set_name in a.sets.split(","):
        items = sets[set_name]
        for system in a.systems.split(","):
            def one(it, system=system):
                try:
                    r = run_system(system, it, llm, tool, orch, cascade)
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001  WHY: one bad query must not lose an hour-long run
                    r = run_system("baseline", it, llm, tool, orch, cascade)
                    r.errors.append(f"{system} crashed, served baseline: {e!r}"[:300])
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
            if "escalation_rate" in summ:
                cov += f" escalated={summ['escalation_rate']:.0%} {summ['escalation_reasons']}"
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
