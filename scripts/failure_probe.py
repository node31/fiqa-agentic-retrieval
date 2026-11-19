"""Probe the three failure cases in results/failures.md: how common is each mode, and does the proposed fix help?

    python scripts/failure_probe.py      # needs results/failure_candidates.json (scripts/failure_cases.py)

Writes results/failure_probe.json. Everything is dev split, chosen config (hybrid linear alpha 0.7, MiniLM).
"""
import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.beir_data import load_corpus, load_split  # noqa: E402
from src.metrics import evaluate_run  # noqa: E402
from src.retrieval.bm25 import tokenize  # noqa: E402
from src.retrieval.dense import get_embedder  # noqa: E402
from src.service.router import single_node_router  # noqa: E402


def main():
    import torch

    torch.set_num_threads(4)
    cands = json.loads((ROOT / "results" / "failure_candidates.json").read_text(encoding="utf-8"))
    corpus = load_corpus()
    queries, qrels = load_split("dev")
    out = {"n_failures": len(cands)}

    # ---- how common is each mode among the Recall@10 = 0 queries -----------------------------------
    words = sorted(len(t.split()) for t in corpus.values())
    p90 = words[int(0.9 * len(words))]

    def rank(x):
        return x[0] if x else 10**9

    modes = Counter()
    for c in cands:
        g = c["gold"]
        if any(rank(x["bm25"]) <= 10 and rank(x["dense"]) > 100 for x in g):
            modes["A_lexical_only_hit_buried"] += 1
        if all(x["words"] > p90 for x in g):
            modes["B_all_gold_long"] += 1
        if all(rank(x["bm25"]) > 100 and rank(x["dense"]) > 100 for x in g):
            modes["C_neither_retriever_top100"] += 1
        if any(11 <= rank(x["fused"]) <= 20 for x in g):
            modes["near_miss_rank_11_20"] += 1
    out["p90_doc_words"] = p90
    out["mode_counts"] = dict(modes)

    # ---- case A: query-adaptive alpha (lower dense weight when the query has a very rare term) --------
    df = Counter()
    for t in corpus.values():
        df.update(set(tokenize(t)))
    n = len(corpus)

    def max_idf(q):
        return max((math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in tokenize(q) if df[t]), default=0.0)

    router = single_node_router()
    loop = asyncio.new_event_loop()

    def run(alpha_fn):
        res = {}
        for qid, q in queries.items():
            r = loop.run_until_complete(router.search(q, k=10, strategy="hybrid", fusion="linear", alpha=alpha_fn(q)))
            res[qid] = [x["doc_id"] for x in r["results"]]
        return res

    for qid in ("4746", "10054"):
        q = queries[qid]
        per = {}
        for a in (0.7, 0.5, 0.4, 0.3):
            ids = [x["doc_id"] for x in loop.run_until_complete(
                router.search(q, k=100, strategy="hybrid", fusion="linear", alpha=a))["results"]]
            per[str(a)] = {g: (ids.index(g) + 1 if g in ids else None) for g in qrels[qid]}
        out[f"caseA_{qid}"] = {"query": q, "max_idf": round(max_idf(q), 2), "gold_rank_by_alpha": per}
    base = evaluate_run(run(lambda q: 0.7), qrels)
    out["caseA_fix_eval"] = {"fixed_0.7": base}
    for thr in (8.0, 9.0):
        out["caseA_fix_eval"][f"adaptive_idf>{thr}_alpha0.5"] = evaluate_run(
            run(lambda q, thr=thr: 0.5 if max_idf(q) > thr else 0.7), qrels)
    out["caseA_fix_eval"]["queries_triggered"] = {
        str(thr): sum(max_idf(q) > thr for q in queries.values()) for thr in (8.0, 9.0)}

    # ---- case B: long passage -> chunk it and take max over chunks -----------------------------------
    emb = get_embedder("minilm")
    tok = emb.model.tokenizer
    qid = "4029"
    qv = emb.encode_queries([queries[qid]])[0]
    top10 = loop.run_until_complete(router.search(queries[qid], k=10, strategy="dense"))["results"]
    caseB = {"query": queries[qid], "dense_score_rank10": round(top10[-1]["score"], 4), "gold": {}}
    for g in qrels[qid]:
        text = corpus[g]
        w = text.split()
        chunks = [" ".join(w[i:i + 150]) for i in range(0, len(w), 100)] or [text]
        cv = emb.encode_docs(chunks)
        whole = float(emb.encode_docs([text])[0] @ qv)
        caseB["gold"][g] = {"words": len(w), "wordpiece_tokens": len(tok(text)["input_ids"]),
                            "dense_whole_truncated": round(whole, 4),
                            "dense_max_over_chunks": round(float(np.max(cv @ qv)), 4),
                            "best_chunk": int(np.argmax(cv @ qv)), "n_chunks": len(chunks)}
    out["caseB_4029"] = caseB

    # ---- case C: vocabulary of query vs gold ---------------------------------------------------------
    qid = "7490"
    qt = set(tokenize(queries[qid]))
    out["caseC_7490"] = {"query": queries[qid], "query_terms": sorted(qt), "gold": {
        g: {"shared_terms": sorted(qt & set(tokenize(corpus[g]))),
            "top_gold_terms": [t for t, _ in Counter(tokenize(corpus[g])).most_common(6)]} for g in qrels[qid]}}

    (ROOT / "results" / "failure_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
