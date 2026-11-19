"""Find failure cases for results/failures.md (Part A4), using the chosen config on dev.

    python scripts/failure_cases.py [--n 12]

For every dev query with Recall@10 = 0 under hybrid-linear-minilm (alpha 0.7), record:
  * the query text and the top-5 results with fused scores + snippets,
  * each gold passage: its rank and score in the fused list, the BM25 list and the dense list
    (searched to depth 1000, so we can see HOW far away it was), its length and snippet.
Writes results/failure_candidates.json. The 3 cases in failures.md were picked from it by hand,
choosing DIFFERENT failure modes (vocabulary mismatch, long-passage truncation, ambiguous query/qrels).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.beir_data import load_corpus, load_split  # noqa: E402
from src.metrics import recall_at_k  # noqa: E402
from src.service.router import single_node_router  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    corpus = load_corpus()
    queries, qrels = load_split("dev")
    router = single_node_router()
    loop = asyncio.new_event_loop()  # WHY one loop: asyncio.run per call leaks executor threads

    def s(q, strategy, k, **kw):
        return loop.run_until_complete(router.search(q, k=k, strategy=strategy, fusion="linear", alpha=0.7,
                                                     depth=max(k, 100), **kw))["results"]

    out = []
    for qid, q in queries.items():
        top = s(q, "hybrid", 10)
        if recall_at_k([r["doc_id"] for r in top], qrels[qid]) > 0:
            continue
        # PITFALL: keep depth = 100 exactly as in serving. Min-max fusion normalises over the candidate
        # pool, so a depth-200 list has different scores and a different order (we were bitten by this).
        fused = {r["doc_id"]: (i + 1, r["score"]) for i, r in enumerate(s(q, "hybrid", 100))}
        bm = {r["doc_id"]: (i + 1, r["score"]) for i, r in enumerate(s(q, "bm25", 1000))}
        dn = {r["doc_id"]: (i + 1, r["score"]) for i, r in enumerate(s(q, "dense", 1000))}
        gold = [{"doc_id": g, "words": len(corpus[g].split()), "fused": fused.get(g), "bm25": bm.get(g),
                 "dense": dn.get(g), "text": corpus[g][:600]} for g in qrels[qid]]
        out.append({"qid": qid, "query": q,
                    "top5": [{"doc_id": r["doc_id"], "score": r["score"], "text": corpus[r["doc_id"]][:300]}
                             for r in top[:5]],
                    "gold": gold})
    # rank candidates: most informative first = gold far away in BOTH lists, or found by only one
    out.sort(key=lambda c: min((g["fused"] or (999,))[0] for g in c["gold"]))
    (ROOT / "results" / "failure_candidates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                             encoding="utf-8")
    print(f"{len(out)} dev queries with Recall@10 = 0 (of {len(queries)}); wrote results/failure_candidates.json")
    for c in out[: a.n]:
        g = c["gold"][0]
        print(f"- {c['qid']}: {c['query'][:70]!r} gold {g['doc_id']} fused={g['fused']} bm25={g['bm25']} "
              f"dense={g['dense']} words={g['words']}")


if __name__ == "__main__":
    main()
