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

HOW TO READ THIS FILE
    load_corpus()       -> {doc_id: passage text}             57,638 passages
    load_all_queries()  -> {query_id: question text}          every split mixed
    load_qrels(split)   -> {query_id: {doc_id: grade}}        the answer key
    load_split(split)   -> (queries, qrels) for ONE split     what callers use
    A "split" (train / dev / test) is defined by its qrels file: the dev split
    is exactly the queries that appear in qrels/dev.tsv (500 dev, 648 test).

    IDs are kept as STRINGS everywhere (e.g. "4102"), never converted to int,
    so they match the ids in run files, indexes and the HTTP API.

    Caching: the loaders use ``functools.lru_cache``, which remembers a
    function's return value per argument tuple. The first call reads the file;
    later calls with the same arguments return the SAME object instantly.
    PITFALL: because it is the same object, never modify a returned dict in
    place: you would silently change it for every other caller in the process.
    PITFALL: the default ``data_dir=DATA_DIR`` is fixed when this module is
    imported, so FIQA_DATA_DIR must be set before the first import.
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
        # JSONL = one JSON object per line, so the file is read line by line.
        for line in f:
            row = json.loads(line)
            # .strip() removes the leading space left behind when the title is empty.
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


@lru_cache(maxsize=4)  # WHY 4: room for train, dev and test (+1) without re-reading files
def load_qrels(split: str, data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} for ``split`` in {train, dev, test}."""
    # defaultdict(dict): the first time a query id is seen, an empty inner dict is
    # created automatically, so we can write qrels[qid][did] without checking.
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(Path(data_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        # One line = one (query, gold passage) pair. A query with 3 gold passages
        # has 3 lines, e.g. "0\t18850\t1" -> qrels["0"]["18850"] = 1.
        for qid, did, score in reader:
            qrels[qid][did] = int(score)
    # Convert back to a plain dict so a typo'd lookup raises KeyError instead of
    # quietly inserting an empty entry.
    return dict(qrels)


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries, qrels) restricted to one split, in a stable (sorted) order."""
    qrels = load_qrels(split, data_dir)
    allq = load_all_queries(data_dir)
    # The qrels decide which queries belong to the split. Ids are sorted
    # NUMERICALLY when they look like numbers ("9" before "10"; a plain string
    # sort would put "10" first), so "the first N queries" (--limit, the 20
    # cold queries in eval.py) are the same N on every run.
    # PITFALL: if one split mixed numeric and non-numeric ids, sorted() would compare
    # int with str and raise TypeError. Today each dataset is uniform: FiQA ids are
    # all numeric, the toy dataset's ("q1", "q2", ...) all fall back to a string sort.
    queries = {qid: allq[qid] for qid in sorted(qrels, key=lambda x: int(x) if x.isdigit() else x)}
    return queries, qrels


def main() -> None:
    corpus = load_corpus()
    # Passage lengths in words, sorted, so the median / 90th percentile are just positions in the list.
    lens = sorted(len(t.split()) for t in corpus.values())
    print(f"corpus: {len(corpus):,} passages; words/passage median={lens[len(lens)//2]} "
          f"p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    for split in ("train", "dev", "test"):
        q, qr = load_split(split)
        # Each qrels entry is one judged (query, passage) pair, so this counts gold passages.
        n_rel = sum(len(v) for v in qr.values())
        print(f"{split:5s}: {len(q):5d} queries, {n_rel:6d} judged-relevant pairs "
              f"({n_rel/len(q):.2f} per query)")


if __name__ == "__main__":
    main()
