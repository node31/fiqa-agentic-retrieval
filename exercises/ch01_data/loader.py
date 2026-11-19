"""ch01: a BEIR loader, written by you. Standard library only.

Run the checker:   python -m pytest ch01_data
Run the demo:      python ch01_data/loader.py
"""
from __future__ import annotations

import json  # noqa: F401  (you'll need these)
from pathlib import Path


def load_jsonl(path: str | Path) -> dict[str, str]:
    """Read a BEIR .jsonl file (corpus or queries) into {_id: text}.

    Each line is a JSON object with "_id", "text" and (corpus only) "title".
    Return text as (title + " " + text).strip()  (FiQA titles are empty, other BEIR sets have them).
    Keep ids as STRINGS.

    >>> # corpus line:  {"_id": "3", "title": "", "text": "I'm not saying ..."}
    >>> # result:       {"3": "I'm not saying ..."}
    """
    path = Path(path)
    result: dict[str, str] = {}

    data = []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    

    raise NotImplementedError("TODO (ch01): implement load_jsonl. See README step 3")


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a qrels TSV (header: query-id, corpus-id, score) into {query_id: {doc_id: score}}."""
    raise NotImplementedError("TODO (ch01): implement load_qrels. Remember to skip the header line")


def load_split(data_dir: str | Path, split: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels) for one split ("train" / "dev" / "test").

    queries must contain ONLY the queries that appear in qrels/<split>.tsv
    (queries.jsonl contains every split's queries).
    """
    raise NotImplementedError("TODO (ch01): implement load_split using load_jsonl + load_qrels")


def stats(data_dir: str | Path, split: str) -> dict:
    """Return {"n_docs": int, "n_queries": int, "n_pairs": int, "avg_gold": float} for a split.

    n_pairs = number of (query, relevant doc) pairs; avg_gold = n_pairs / n_queries.
    """
    raise NotImplementedError("TODO (ch01): implement stats")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # exercises/ (for common.py)
    from common import dataset_from_argv

    name, data_dir = dataset_from_argv()           # add --full for the real FiQA
    for s in (("dev",) if name == "tiny" else ("train", "dev", "test")):
        print(f"{name} {s}:", stats(data_dir, s))  # tiny dev: 24 docs, 4 queries, 7 pairs, 1.75 · full dev: 57638/500/1238
