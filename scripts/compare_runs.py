"""Compare two eval runs on the same questions: per-query deltas and a paired bootstrap 95% CI.

    python scripts/compare_runs.py results/runs/hybrid-linear-minilm.json results/runs/hybrid-linear-minilm-ft.json
    -> prints the table and writes results/compare_<A>__<B>.json

WHY PAIRED
    Both runs answer the SAME 500 questions, so the honest question is "how much does B change each
    question's score", not "are two averages different". The bootstrap resamples QUESTIONS (with
    replacement, 10,000 times, seed 42), recomputes mean(B - A) each time, and reports the 2.5th and
    97.5th percentiles. If that interval excludes 0, the gain is unlikely to be luck of which 500
    questions dev happens to contain.

    Toy: per-question deltas [+0.2, 0.0, -0.1, +0.3]; mean +0.10. Resample 4 of them with replacement many
    times -> means like +0.20, +0.05, -0.03, ... -> the middle 95% of those means is the CI.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.beir_data import load_corpus, load_split  # noqa: E402
from src.config import RESULTS_DIR, SEED  # noqa: E402
from src.eval import long_doc_ids  # noqa: E402
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

METRICS = {"ndcg@10": ndcg_at_k, "recall@10": recall_at_k, "mrr@10": mrr_at_k}


def paired_bootstrap(deltas: list[float], n: int = 10000, seed: int = SEED) -> tuple[float, float]:
    """95% CI of mean(deltas) by resampling the questions with replacement."""
    rng = random.Random(seed)
    k = len(deltas)
    means = sorted(sum(deltas[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def compare(a: dict, b: dict, qrels: dict, long_docs: set[str]) -> dict:
    qids = [q for q in qrels if q in a["run"] and q in b["run"]]
    out = {"n": len(qids), "metrics": {}}
    for name, fn in METRICS.items():
        sa = [fn(a["run"][q], qrels[q], 10) for q in qids]
        sb = [fn(b["run"][q], qrels[q], 10) for q in qids]
        d = [y - x for x, y in zip(sa, sb)]
        lo, hi = paired_bootstrap(d)
        out["metrics"][name] = {"a": round(sum(sa) / len(qids), 4), "b": round(sum(sb) / len(qids), 4),
                                "delta": round(sum(d) / len(qids), 4), "ci95": [round(lo, 4), round(hi, 4)],
                                "better": sum(x > 1e-9 for x in d), "worse": sum(x < -1e-9 for x in d)}
    # The long-document slice from A2: does training on FiQA help the passages MiniLM truncates?
    gold_long = [q for q in qids if any(d in long_docs for d, g in qrels[q].items() if g > 0)]
    rest = [q for q in qids if q not in set(gold_long)]
    out["recall@10_by_slice"] = {
        name: {"n": len(qs),
               "a": round(sum(recall_at_k(a["run"][q], qrels[q], 10) for q in qs) / max(1, len(qs)), 4),
               "b": round(sum(recall_at_k(b["run"][q], qrels[q], 10) for q in qs) / max(1, len(qs)), 4)}
        for name, qs in (("gold_in_longest_10pct", gold_long), ("rest", rest))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", type=Path, help="baseline run file (results/runs/<name>.json)")
    ap.add_argument("b", type=Path, help="candidate run file")
    args = ap.parse_args()
    a, b = json.loads(args.a.read_text()), json.loads(args.b.read_text())
    if a.get("split") != b.get("split"):
        raise SystemExit(f"different splits: {a.get('split')} vs {b.get('split')}")
    _, qrels = load_split(a.get("split", "dev"))
    res = {"a": args.a.stem, "b": args.b.stem, "split": a.get("split", "dev"),
           **compare(a, b, qrels, long_doc_ids(load_corpus()))}
    print(f"{res['a']}  ->  {res['b']}   ({res['split']}, n = {res['n']})")
    for name, m in res["metrics"].items():
        print(f"  {name:10s} {m['a']:.4f} -> {m['b']:.4f}   delta {m['delta']:+.4f}  "
              f"95% CI [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]   better {m['better']}, worse {m['worse']}")
    for name, s in res["recall@10_by_slice"].items():
        print(f"  R@10 {name:22s} n={s['n']:3d}  {s['a']:.4f} -> {s['b']:.4f}")
    out = RESULTS_DIR / f"compare_{res['a']}__{res['b']}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
