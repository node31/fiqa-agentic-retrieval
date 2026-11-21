"""Build the compound query set (Part C1): data/compound_dev.jsonl.

    python data/build_compound.py            # deterministic (seed 42)

STEPS
    1. Sample 200 dev queries with a fixed seed.
    2. Pair them so the two halves are on DIFFERENT topics. We embed all 200 with
       MiniLM, then greedily give each query (in seeded random order) a partner drawn
       at random from the still-unpaired queries with cosine similarity < 0.25
       (the least-similar one if none qualifies). Pairing by MINIMUM similarity
       instead tends to always pick the same few odd queries, so we use
       "random below a threshold".
    3. Join each pair into one natural question. Rewrites live in
       data/compound_rewrites.json (written in-session by GitHub Copilot; 10 are listed
       for a manual check in data/compound_review_sheet.md). A pair with no rewrite falls
       back to a template: "<q_a>? Also, <q_b>".
    4. Gold = union of both halves' qrels, with each gold doc tagged "a" or "b".
       That tag is what Coverage@10 needs.

OUTPUT (one JSON per line)
    {"id": "c000", "question": "...", "a": {"qid", "text"}, "b": {...}, "cos": 0.08,
     "gold": {"doc_id": "a" | "b"}, "rewrite_source": "copilot" | "template"}
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.beir_data import load_split  # noqa: E402
from src.config import SEED  # noqa: E402

OUT = ROOT / "data" / "compound_dev.jsonl"
REWRITES = ROOT / "data" / "compound_rewrites.json"
THRESHOLD = 0.25


def build(seed: int = SEED, n_queries: int = 200, split: str = "dev") -> list[dict]:
    queries, qrels = load_split(split)
    rng = random.Random(seed)
    qids = rng.sample(sorted(queries), n_queries)
    from src.retrieval.dense import get_embedder

    emb = get_embedder("minilm").encode_queries([queries[q] for q in qids])
    sims = emb @ emb.T
    order = list(range(n_queries))
    rng.shuffle(order)
    unpaired, pairs = set(order), []
    for i in order:
        if i not in unpaired:
            continue
        unpaired.discard(i)
        cands = sorted(j for j in unpaired if sims[i, j] < THRESHOLD)
        j = rng.choice(cands) if cands else min(unpaired, key=lambda x: sims[i, x])
        unpaired.discard(j)
        pairs.append((i, j))
    rewrites = json.loads(REWRITES.read_text(encoding="utf-8")) if REWRITES.exists() and split == "dev" else {}
    rows = []
    for n, (i, j) in enumerate(pairs):
        a, b = qids[i], qids[j]
        cid = f"c{n:03d}"
        gold = {d: "a" for d in qrels[a]}
        for d in qrels[b]:
            gold.setdefault(d, "b")  # a doc relevant to both halves (rare) counts for half a
        template = f"{queries[a].rstrip(' ?')}? Also, {queries[b]}"
        rows.append({"id": cid, "question": rewrites.get(cid, template),
                     "a": {"qid": a, "text": queries[a]}, "b": {"qid": b, "text": queries[b]},
                     "cos": round(float(sims[i, j]), 3), "gold": gold,
                     "rewrite_source": "copilot" if cid in rewrites else "template"})
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="test = held-out compound set (template rewrites; never tune on it)")
    split = ap.parse_args().split
    rows = build(split=split)
    out = OUT if split == "dev" else OUT.with_name("compound_test.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    cos = np.array([r["cos"] for r in rows])
    print(f"wrote {len(rows)} compound queries -> {out} (pair cosine mean={cos.mean():.3f}, max={cos.max():.3f}; "
          f"rewritten={sum(r['rewrite_source'] == 'copilot' for r in rows)})")


if __name__ == "__main__":
    main()
