"""Semantic cache: how often do rephrased questions hit, how often do DIFFERENT questions wrongly hit, and
what does a wrong hit cost in quality? Picks the threshold for src/agent/semantic_cache.py.

    python scripts/semantic_cache_eval.py --confirm-spend      # first run: writes the paraphrases (Azure, ~$0.05)
    python scripts/semantic_cache_eval.py                      # later runs reuse data/paraphrases_plain.jsonl

Writes results/semantic_cache_eval.json.

THE THREE MEASUREMENTS
    1. Benefit. data/paraphrases_plain.jsonl has one rewording per question of the agent eval's plain set
       (100 dev questions), written by the LLM ("ask it the way a different person would"). With every
       train + dev question in the cache, a paraphrase should hit its ORIGINAL. "true hit" = the nearest
       cached question is the original and the similarity clears the threshold; "wrong hit" = it clears the
       threshold but the nearest one is some other question.
    2. Harm. Each of the 500 dev questions against a cache of every OTHER train + dev question: a hit here
       is a different question answered from the cache. For those, the served results (the neighbour's
       hybrid top-10) are scored against the asking question's own relevance labels, next to its own
       top-10. Some FiQA questions really are duplicates, so not every such hit is harmful; the nDCG says
       how much is.
    3. Agent savings. The plain-set cascade answers (results/agent_eval_cascade.json) go into a cache; the
       paraphrases are looked up. A hit costs nothing and serves the original's answer, scored against the
       shared relevance labels.
The query encoder is the served one (minilm-ft): the cache reuses the embedding the search computes anyway.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.beir_data import load_split  # noqa: E402
from src.config import SEED  # noqa: E402
from src.metrics import ndcg_at_k, recall_at_k  # noqa: E402

PARA = ROOT / "data" / "paraphrases_plain.jsonl"
OUT = ROOT / "results" / "semantic_cache_eval.json"
CASCADE = ROOT / "results" / "agent_eval_cascade.json"
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
PARAPHRASE_PROMPT = """ROLE: paraphraser
Rewrite the user's personal-finance question the way a different person would ask it: the same meaning
and the same details, but different wording. Reply with the rewritten question only, nothing else."""


def plain_set() -> list[tuple[str, str]]:
    queries, _ = load_split("dev")
    ids = random.Random(SEED).sample(sorted(queries), 100)  # the agent eval's plain set
    return [(q, queries[q]) for q in ids]


def paraphrases(confirm_spend: bool) -> list[dict]:
    if PARA.exists():
        return [json.loads(line) for line in PARA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not confirm_spend:
        sys.exit(f"{PARA.relative_to(ROOT)} is missing: run once with --confirm-spend to write it (Azure, ~$0.05)")
    from src.agent.llm import get_llm

    llm = get_llm("azure", confirm_spend=True)
    rows = []
    for qid, q in plain_set():
        text = " ".join(llm.complete(PARAPHRASE_PROMPT, q, max_output_tokens=400).text.split()).strip('"')
        rows.append({"id": qid, "question": q, "paraphrase": text})
    PARA.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--model", default="minilm-ft")
    a = ap.parse_args(argv)
    rows = paraphrases(a.confirm_spend)
    from src.retrieval.dense import get_embedder

    train_q, _ = load_split("train")
    dev_q, dev_rel = load_split("dev")
    ids = list(train_q) + list(dev_q)
    texts = [train_q.get(i) or dev_q[i] for i in ids]
    emb = get_embedder(a.model)
    print(f"embedding {len(ids)} questions + {len(rows)} paraphrases with {a.model}", flush=True)
    M = unit(np.asarray(emb.encode_queries(texts), dtype=np.float32))
    P = unit(np.asarray(emb.encode_queries([r["paraphrase"] for r in rows]), dtype=np.float32))
    pos = {q: i for i, q in enumerate(ids)}

    # 1. benefit: paraphrase -> nearest cached question
    sims = P @ M.T
    nn = sims.argmax(1)
    para = [{"id": r["id"], "nn_is_original": ids[nn[j]] == r["id"], "sim": float(sims[j, nn[j]]),
             "sim_to_original": float(sims[j, pos[r["id"]]])} for j, r in enumerate(rows)]

    # 2. harm: each dev question vs every OTHER question
    dev_idx = [pos[q] for q in dev_q]
    S = M[dev_idx] @ M.T
    S[np.arange(len(dev_idx)), dev_idx] = -1.0  # not itself
    dnn = S.argmax(1)
    dev_hits = [(q, ids[dnn[j]], float(S[j, dnn[j]])) for j, q in enumerate(dev_q)]
    lowest = min(THRESHOLDS)
    risky = [(q, n, s) for q, n, s in dev_hits if s >= lowest]
    print(f"{len(risky)} dev questions have a different question at similarity >= {lowest}; searching both", flush=True)
    from src.service.router import single_node_router

    router = single_node_router(models=(a.model,))
    loop = asyncio.new_event_loop()
    top10 = {}
    for qid in {x for q, n, _ in risky for x in (q, n)}:
        text = train_q.get(qid) or dev_q[qid]
        top10[qid] = [r["doc_id"] for r in loop.run_until_complete(router.search(text, k=10))["results"]]
    harm = [{"id": q, "neighbour": n, "sim": round(s, 4), "own_ndcg": ndcg_at_k(top10[q], dev_rel[q]),
             "served_ndcg": ndcg_at_k(top10[n], dev_rel[q])} for q, n, s in risky]

    sweep = []
    for t in THRESHOLDS:
        hit = [p for p in para if p["sim"] >= t]
        wrong = [h for h in harm if h["sim"] >= t]
        sweep.append({"threshold": t,
                      "paraphrase_true_hits": sum(p["nn_is_original"] for p in hit) / len(para),
                      "paraphrase_wrong_hits": sum(not p["nn_is_original"] for p in hit) / len(para),
                      "dev_false_hits": len(wrong) / len(dev_q),
                      "false_hit_own_ndcg": round(float(np.mean([h["own_ndcg"] for h in wrong])), 4) if wrong else None,
                      "false_hit_served_ndcg": round(float(np.mean([h["served_ndcg"] for h in wrong])), 4) if wrong else None})
    # Default: the lowest threshold at which at most 1% of dev questions wrongly hit.
    chosen = next(s for s in sweep if s["dev_false_hits"] <= 0.01)
    print(json.dumps(sweep, indent=1), flush=True)

    # 3. agent savings at the chosen threshold
    agent = None
    if CASCADE.exists():
        rep = json.loads(CASCADE.read_text(encoding="utf-8"))
        answers = {r["id"]: r for r in rep["per_query"].get("plain/cascade", [])}
        tr = [p for p in para if p["id"] in answers]
        hits = [p for p in tr if p["nn_is_original"] and p["sim"] >= chosen["threshold"]]
        per = {r["id"]: r for r in rep["per_query"]["plain/cascade"]}
        # what the cascade actually paid per question: the non-shadow tiers' usd (shadow runs are eval-only)
        cost = {r["id"]: sum(t["usd"] for t in r["trace"] if not t.get("shadow"))
                for r in rep["per_query"]["plain/cascade"]}
        agent = {"threshold": chosen["threshold"], "paraphrases": len(tr), "hits": len(hits),
                 "llm_calls_saved": sum(per[p["id"]]["llm_calls"] for p in hits),
                 "usd_saved": round(sum(cost[p["id"]] for p in hits), 5),
                 "served_recall@10_on_hits": round(float(np.mean([recall_at_k(per[p["id"]]["doc_ids"], dev_rel[p["id"]])
                                                                  for p in hits])), 4) if hits else None}
        print(f"agent savings at {chosen['threshold']}: {agent}", flush=True)
    res = {"model": a.model, "n_cache": len(ids), "n_paraphrases": len(para), "chosen": chosen, "sweep": sweep,
           "paraphrase_sim_to_original": {"p10": round(float(np.percentile([p["sim_to_original"] for p in para], 10)), 4),
                                          "median": round(float(np.median([p["sim_to_original"] for p in para])), 4)},
           "dev_nn_sim": {"median": round(float(np.median([s for _, _, s in dev_hits])), 4),
                          "p99": round(float(np.percentile([s for _, _, s in dev_hits], 99)), 4)},
           "false_hit_examples": sorted(harm, key=lambda h: -h["sim"])[:10], "agent": agent}
    for ex in res["false_hit_examples"]:
        ex["question"], ex["neighbour_question"] = dev_q[ex["id"]], train_q.get(ex["neighbour"]) or dev_q[ex["neighbour"]]
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
