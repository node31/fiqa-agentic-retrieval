"""Load the BEIR FiQA dataset (corpus, queries, qrels) from the standard download.

WHAT
    Three tiny readers for the BEIR on-disk format:
      corpus.jsonl   {"_id", "title", "text"}         one line per passage
      queries.jsonl  {"_id", "text"}                  train + dev + test queries
      qrels/<split>.tsv  query-id  corpus-id  score     relevance judgements

WHY
    The ``beir`` pip package pulls many heavy dependencies just to read three
    files. Reading them ourselves takes 40 lines and makes the format obvious.

KEY CONCEPT: qrels
    "Query relevance judgements". For FiQA every judged pair has score 1 and
    unjudged documents are assumed non-relevant. A query has ~2.6 relevant
    passages on average. See docs/04_ir_metrics.md.

PITFALLS
    * Queries for all splits live in one file. Always filter by the split's
      qrels, otherwise you silently evaluate on train queries.
    * FiQA titles are empty; passage text alone is the document.

Run ``python -m src.beir_data`` to print dataset statistics.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from src.config import DATA_DIR


@lru_cache(maxsize=1)
def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Return {doc_id: text}. Title and text are joined because titles are usually empty."""
    corpus: dict[str, str] = {}
    with open(Path(data_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = (row.get("title", "") + " " + row["text"]).strip()
    return corpus


@lru_cache(maxsize=1)
def load_all_queries(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Return {query_id: text} for every split (filter with :func:`load_qrels`)."""
    out: dict[str, str] = {}
    with open(Path(data_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"]
    return out


@lru_cache(maxsize=4)
def load_qrels(split: str, data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} for ``split`` in {train, dev, test}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(Path(data_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            qrels[qid][did] = int(score)
    return dict(qrels)


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels) restricted to one split, in a stable (sorted) order."""
    qrels = load_qrels(split, data_dir)
    allq = load_all_queries(data_dir)
    queries = {qid: allq[qid] for qid in sorted(qrels, key=lambda x: int(x) if x.isdigit() else x)}
    return queries, qrels


def main() -> None:
    corpus = load_corpus()
    lens = sorted(len(t.split()) for t in corpus.values())
    print(f"corpus: {len(corpus):,} passages; words/passage median={lens[len(lens)//2]} "
          f"p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    for split in ("train", "dev", "test"):
        q, qr = load_split(split)
        n_rel = sum(len(v) for v in qr.values())
        print(f"{split:5s}: {len(q):5d} queries, {n_rel:6d} judged-relevant pairs "
              f"({n_rel/len(q):.2f} per query)")


if __name__ == "__main__":
    main()
