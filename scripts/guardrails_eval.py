"""Measure the guardrails: what they catch, what they wrongly refuse, and where the scope threshold sits.

    python scripts/guardrails_eval.py                      # rules + corpus scan + scope (minilm-ft, minilm)
    python scripts/guardrails_eval.py --models minilm      # scope calibration for one model only

Writes results/guardrails_eval.json. docs/design/SECURITY.md quotes it.

WHAT IT MEASURES
    1. Rules on data/guardrails_redteam.jsonl (hand-written; 4 groups):
         harmful           how-to fraud questions            -> share caught       (recall; higher is better)
         benign_sensitive  the same topics, asked to learn   -> share refused      (false blocks; lower is better)
         injection         prompt-injection attempts         -> share caught
         off_topic         nothing to do with finance        -> share flagged by the SCOPE check
       The last 5 harmful rows are paraphrases written to dodge keywords: they measure the rules' limit.
       The rules were tuned once on this set's misses, so the HELD-OUT set (data/guardrails_heldout.jsonl,
       written after the rules were frozen) gives the unbiased numbers.
    2. False blocks on REAL questions: every FiQA question (train + dev + test, 6,648) through the
       harmful and injection rules. Each flagged one is listed, so you can judge it yourself.
    3. The corpus: how many of the 57,638 passages contain PII (by type) or instruction-like text.
       That is what pii_redaction and guard_injection would touch in returned passages.
    4. Scope calibration per model: the best dense cosine of each dev question against the corpus. The
       threshold is the 1st percentile of those (so ~1% of real questions would be flagged), then the
       share of off-topic questions it catches. A sweep shows the trade-off.
    5. Speed of the rules (microseconds per question).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.beir_data import load_all_queries, load_corpus, load_split  # noqa: E402
from src.security import guardrails, pii  # noqa: E402

REDTEAM = ROOT / "data" / "guardrails_redteam.jsonl"
HELDOUT = ROOT / "data" / "guardrails_heldout.jsonl"
# The first run of this script, BEFORE I tuned the rules on its misses (kept so the tuning is visible).
# After tuning, the red-team set is no longer an unbiased test: data/guardrails_heldout.jsonl was written
# after the rules were frozen and never used to change them. Its numbers are the honest ones.
BEFORE_TUNING = {"harmful": "36/45", "benign_sensitive_false_blocks": "0/40", "injection": "15/15",
                 "fiqa_question_false_blocks": "0/6648", "corpus_injection_flags": 14}
OUT = ROOT / "results" / "guardrails_eval.json"


def rules_on_redteam(rows: list[dict]) -> dict:
    out = {}
    for group in ("harmful", "benign_sensitive", "injection"):
        g = [r for r in rows if r["group"] == group]
        flagged = []
        for r in g:
            f = guardrails.check_injection(r["text"]) if group == "injection" else guardrails.check_harmful(r["text"])
            if f:
                flagged.append(r["id"])
        missed = [r for r in g if r["id"] not in flagged]
        out[group] = {"n": len(g), "flagged": len(flagged), "rate": round(len(flagged) / len(g), 3),
                      ("missed" if group != "benign_sensitive" else "false_blocks"):
                          [{"id": r["id"], "text": r["text"]} for r in (missed if group != "benign_sensitive"
                                                                        else [r for r in g if r["id"] in flagged])]}
    para = [r for r in rows if r.get("note")]
    if para:
        out["harmful"]["paraphrases_caught"] = f"{sum(1 for r in para if guardrails.check_harmful(r['text']))}/{len(para)}"
    # Cross-check: no benign or off-topic row trips the injection rule, no injection row needs the harmful rule.
    out["injection_false_alarms_on_other_groups"] = sum(
        1 for r in rows if r["group"] in ("harmful", "benign_sensitive", "off_topic") and guardrails.check_injection(r["text"]))
    return out


def rules_on_fiqa() -> dict:
    qs = load_all_queries()
    harm = [(qid, t, guardrails.check_harmful(t)) for qid, t in qs.items()]
    harm = [{"id": q, "text": t, "category": f["category"], "matched": f["matched"]} for q, t, f in harm if f]
    inj = [{"id": q, "text": t, "matched": guardrails.find_injection(t)} for q, t in qs.items() if guardrails.find_injection(t)]
    dev, _ = load_split("dev")
    return {"n_questions": len(qs), "harmful_flags": len(harm), "harmful_rate": round(len(harm) / len(qs), 5),
            "dev_harmful_flags": sum(1 for h in harm if h["id"] in dev), "n_dev": len(dev),
            "injection_flags": len(inj), "harmful_flagged": harm, "injection_flagged": inj}


def corpus_scan(corpus: dict[str, str]) -> dict:
    t0 = time.perf_counter()
    kinds, docs_with = Counter(), 0
    inj_docs = []
    for d, text in corpus.items():
        c = pii.find(text)
        if c:
            docs_with += 1
            kinds.update(c)
        if guardrails.find_injection(text):
            inj_docs.append(d)
    return {"n_passages": len(corpus), "pii_passages": docs_with, "pii_share": round(docs_with / len(corpus), 4),
            "pii_items_by_type": dict(kinds.most_common()), "injection_passages": len(inj_docs),
            "injection_examples": [{"doc_id": d, "matched": guardrails.find_injection(corpus[d])}
                                   for d in inj_docs[:15]],
            "scan_seconds": round(time.perf_counter() - t0, 1)}


def scope(models: list[str], rows: list[dict]) -> dict:
    import faiss

    from src.index.build_index import load_cache
    from src.retrieval.dense import get_embedder

    dev, _ = load_split("dev")
    groups = {"dev": list(dev.values()),
              **{g: [r["text"] for r in rows if r["group"] == g] for g in ("off_topic", "harmful", "benign_sensitive")}}
    out = {}
    for m in models:
        cache = load_cache(m)
        if cache is None:
            print(f"  {m}: no embedding cache, skipped", flush=True)
            continue
        _, _, mat = cache
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(np.ascontiguousarray(mat, dtype=np.float32))
        emb = get_embedder(m)
        top = {}
        for g, texts in groups.items():
            q = np.asarray(emb.encode_queries(texts), dtype=np.float32)
            top[g] = index.search(q, 1)[0][:, 0]
        tau = float(np.percentile(top["dev"], 1))
        sweep = []
        for pct in (0.5, 1, 2, 5, 10):
            t = float(np.percentile(top["dev"], pct))
            sweep.append({"dev_percentile": pct, "threshold": round(t, 4),
                          "dev_flagged": round(float((top["dev"] < t).mean()), 3),
                          "off_topic_caught": round(float((top["off_topic"] < t).mean()), 3)})
        out[m] = {"threshold": round(tau, 4),
                  "dev_flagged": round(float((top["dev"] < tau).mean()), 3),
                  "off_topic_caught": round(float((top["off_topic"] < tau).mean()), 3),
                  "harmful_flagged_off_topic": round(float((top["harmful"] < tau).mean()), 3),
                  "top_score": {g: {"p1": round(float(np.percentile(v, 1)), 4), "median": round(float(np.median(v)), 4),
                                    "max": round(float(v.max()), 4)} for g, v in top.items()},
                  "sweep": sweep}
        print(f"  {m}: threshold {tau:.4f} -> dev flagged {out[m]['dev_flagged']:.1%}, "
              f"off-topic caught {out[m]['off_topic_caught']:.1%}", flush=True)
    return out


def speed(rows: list[dict]) -> dict:
    texts = [r["text"] for r in rows] * 50
    t0 = time.perf_counter()
    for t in texts:
        guardrails.check_harmful(t)
        guardrails.check_injection(t)
    return {"rules_us_per_question": round((time.perf_counter() - t0) / len(texts) * 1e6, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="minilm-ft,minilm", help="models to calibrate the scope threshold for")
    ap.add_argument("--no-scope", action="store_true", help="skip the embedding part (rules + corpus only)")
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in REDTEAM.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("1. rules on the red-team set", flush=True)
    held = [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    res = {"before_tuning": BEFORE_TUNING, "redteam": rules_on_redteam(rows), "heldout": rules_on_redteam(held)}
    for name in ("redteam", "heldout"):
        for g in ("harmful", "benign_sensitive", "injection"):
            print(f"  {name} {g}: {res[name][g]['flagged']}/{res[name][g]['n']}", flush=True)
    print("2. rules on every FiQA question", flush=True)
    res["fiqa_questions"] = rules_on_fiqa()
    print(f"  harmful flags {res['fiqa_questions']['harmful_flags']}/{res['fiqa_questions']['n_questions']}, "
          f"injection flags {res['fiqa_questions']['injection_flags']}", flush=True)
    print("3. corpus scan (PII, injection)", flush=True)
    res["corpus"] = corpus_scan(load_corpus())
    print(f"  PII in {res['corpus']['pii_passages']} passages, injection-like text in "
          f"{res['corpus']['injection_passages']}", flush=True)
    if not a.no_scope:
        print("4. scope calibration", flush=True)
        res["scope"] = scope([m for m in a.models.split(",") if m], rows)
    res["speed"] = speed(rows)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
