"""Compare cascade escalation rules on the SAME questions, offline, from one shadow run.

    python scripts/agent_eval.py --backend azure --confirm-spend --systems cascade --cascade-shadow \
        --out results/agent_eval_cascade.json                   # both tiers' answers for every question
    python scripts/cascade_policies.py                           # -> results/cascade_policies.json

WHY OFFLINE
    A cascade's quality and cost depend only on (a) each tier's answer to each question and (b) the rule that
    picks one. With --cascade-shadow the eval records BOTH tiers' answers and token costs for every question,
    so any rule can be scored without another LLM call, and every rule is compared on identical answers
    (no run-to-run noise).

RULES
    strong only        every question to the strong tier (= the single agent)
    cheap only         every question to the cheap tier
    confidence         cheap first; escalate on error / no finish / low confidence / < 3 picks
    picks only         cheap first; escalate on error / no finish / < 3 picks (ignore the self-reported confidence)
    route only         compound-looking questions to strong, the rest to cheap; no escalation
    route+confidence   the default: compound-looking -> strong; otherwise "confidence"
    oracle             per question, the better of the two answers (an upper bound no rule can beat)
    Cost = what the rule would have paid: the cheap tier's tokens whenever it runs, plus the strong tier's
    whenever it runs. ``what_if_cheap_price`` reprices the cheap tier's tokens at a fraction of the strong
    price, i.e. what the same rule would cost with a genuinely cheaper model in tier 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_eval import load_sets  # noqa: E402
from src.agent.orchestrator import looks_simple  # noqa: E402
from src.metrics import coverage_at_k, mrr_at_k, recall_at_k  # noqa: E402

IN = ROOT / "results" / "agent_eval_cascade.json"
OUT = ROOT / "results" / "cascade_policies.json"
HARD = ("error", "no finish", "too few picks")


def tiers_of(row: dict) -> tuple[dict, dict]:
    by = {t["tier"]: t for t in row["trace"]}
    return by["cheap"], by["strong"]


def cheap_reason(cheap: dict) -> str | None:
    return cheap.get("escalate")


def score(rule: str, q: str, cheap: dict, strong: dict) -> tuple[list[str], float, bool]:
    """(doc ids served, usd paid, went to strong?) for one question under ``rule``."""
    c, s, reason = cheap["usd"], strong["usd"], cheap_reason(cheap)
    if rule == "strong only":
        return strong["doc_ids"], s, True
    if rule == "cheap only":
        return cheap["doc_ids"], c, False
    if rule == "route only":
        return (strong["doc_ids"], s, True) if not looks_simple(q) else (cheap["doc_ids"], c, False)
    if rule in ("confidence", "picks only", "route+confidence"):
        if rule == "route+confidence" and not looks_simple(q):
            return strong["doc_ids"], s, True
        escalate = reason is not None if rule != "picks only" else reason in HARD
        return (strong["doc_ids"], c + s, True) if escalate else (cheap["doc_ids"], c, False)
    raise ValueError(rule)


RULES = ["strong only", "cheap only", "confidence", "picks only", "route only", "route+confidence"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=IN)
    ap.add_argument("--cheap-price-ratio", type=float, default=0.1,
                    help="what-if: the cheap tier's price as a fraction of the strong tier's (default 0.1)")
    a = ap.parse_args(argv)
    rep = json.loads(a.input.read_text(encoding="utf-8"))
    sets = load_sets()
    out = {"input": str(a.input.relative_to(ROOT)), "cheap_price_ratio": a.cheap_price_ratio, "sets": {}}
    for set_name, items in sets.items():
        rows = {r["id"]: r for r in rep["per_query"].get(f"{set_name}/cascade", [])}
        items = [it for it in items if it["id"] in rows]
        if not items:
            continue
        table = {}
        for rule in RULES + ["oracle"]:
            rec = mrr = cov = usd = usd_whatif = to_strong = 0.0
            for it in items:
                cheap, strong = tiers_of(rows[it["id"]])
                rel = {d: 1 for d in it["gold"]}
                if rule == "oracle":
                    better = recall_at_k(strong["doc_ids"], rel) > recall_at_k(cheap["doc_ids"], rel)
                    ids, paid, went = (strong["doc_ids"], cheap["usd"] + strong["usd"], True) if better else \
                        (cheap["doc_ids"], cheap["usd"], False)
                else:
                    ids, paid, went = score(rule, it["question"], cheap, strong)
                # what-if: same rule, cheap tier priced at ratio x the strong tier's price
                cheap_ran = rule != "strong only" and not (rule in ("route only", "route+confidence")
                                                           and not looks_simple(it["question"]))
                usd_whatif += paid - (cheap["usd"] * (1 - a.cheap_price_ratio) if cheap_ran else 0.0)
                rec += recall_at_k(ids, rel)
                mrr += mrr_at_k(ids, rel)
                if set_name == "compound":
                    cov += coverage_at_k(ids, {h: [d for d, x in it["gold"].items() if x == h] for h in ("a", "b")})
                usd += paid
                to_strong += went
            n = len(items)
            table[rule] = {"recall@10": round(rec / n, 4), "mrr@10": round(mrr / n, 4),
                           **({"coverage@10": round(cov / n, 4)} if set_name == "compound" else {}),
                           "usd_per_q": round(usd / n, 6), "usd_per_q_what_if_cheaper_tier1": round(usd_whatif / n, 6),
                           "share_to_strong": round(to_strong / n, 3)}
        out["sets"][set_name] = {"n": len(items), "rules": table}
        print(f"\n{set_name} (n={len(items)})")
        print(f"  {'rule':18s} {'R@10':>6s} {'MRR':>6s} {'$/q':>9s} {'$/q what-if':>11s} {'to strong':>9s}")
        for rule, m in table.items():
            print(f"  {rule:18s} {m['recall@10']:6.3f} {m['mrr@10']:6.3f} {m['usd_per_q']:9.5f} "
                  f"{m['usd_per_q_what_if_cheaper_tier1']:11.5f} {m['share_to_strong']:9.0%}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
