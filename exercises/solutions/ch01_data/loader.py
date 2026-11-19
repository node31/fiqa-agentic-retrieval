"""SOLUTION for ch01: a BEIR loader. Standard library only.

Try the exercise yourself first (exercises/ch01_data/loader.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch01_data
Run the demo:    python exercises\\solutions\\ch01_data\\loader.py
"""
from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> dict[str, str]:
    """Read a BEIR .jsonl file (corpus or queries) into {_id: text}."""
    out: dict[str, str] = {}
    # WHY encoding="utf-8": Windows defaults to cp1252 and would crash (or garble) on "é", "€", emoji...
    with open(path, encoding="utf-8") as f:
        for line in f:                       # WHY line by line: the real corpus is 45 MB; no need to hold it twice
            line = line.strip()
            if not line:                     # PITFALL: a trailing blank line would make json.loads fail
                continue
            rec = json.loads(line)
            # WHY title + text: other BEIR sets (e.g. TREC-COVID) put real words in the title; FiQA's are "".
            # .strip() removes the leading space when the title is empty.
            out[str(rec["_id"])] = (rec.get("title", "") + " " + rec["text"]).strip()
    return out


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a qrels TSV (header: query-id, corpus-id, score) into {query_id: {doc_id: score}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)                              # WHY: line 1 is the header "query-id<TAB>corpus-id<TAB>score"
        for line in f:
            if not line.strip():
                continue
            qid, did, score = line.rstrip("\n").split("\t")
            # WHY keep ids as str: "08" and "8" would collide as ints, and the corpus ids are strings.
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_split(data_dir: str | Path, split: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels) for one split, keeping only that split's queries."""
    data_dir = Path(data_dir)
    qrels = load_qrels(data_dir / "qrels" / f"{split}.tsv")
    all_queries = load_jsonl(data_dir / "queries.jsonl")   # contains train + dev + test queries
    # WHY filter by qrels: the split is DEFINED by which query ids appear in qrels/<split>.tsv.
    queries = {qid: all_queries[qid] for qid in qrels}
    return queries, qrels


def stats(data_dir: str | Path, split: str) -> dict:
    """{"n_docs", "n_queries", "n_pairs", "avg_gold"} for a split."""
    data_dir = Path(data_dir)
    queries, qrels = load_split(data_dir, split)
    n_pairs = sum(len(docs) for docs in qrels.values())    # one pair per (query, relevant doc)
    return {"n_docs": len(load_jsonl(data_dir / "corpus.jsonl")),
            "n_queries": len(queries),
            "n_pairs": n_pairs,
            "avg_gold": n_pairs / len(queries)}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # exercises/ (for common.py)
    from common import dataset_from_argv

    name, data_dir = dataset_from_argv()           # add --full for the real FiQA
    for s in (("dev",) if name == "tiny" else ("train", "dev", "test")):
        print(f"{name} {s}:", stats(data_dir, s))  # tiny dev: 24 docs, 4 queries, 7 pairs, 1.75 · full dev: 57638/500/1238
