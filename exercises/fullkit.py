"""Reference helpers for the FULL checks (57,638 docs). You don't need to edit (or read) this.

The full checks compare YOUR code with the project's reference code on the real data. Reference results are
computed once and cached in exercises/.cache/ (delete that folder to recompute):
  bm25_run(split)   reference BM25 top-100 per query     (~1 min the first time: builds the index)
  dense_run(split)  reference MiniLM top-100 per query   (seconds, uses artifacts/emb_minilm.npy;
                                                          ~10 min the first time if that file is missing)
  hybrid_search(q)  the shipped hybrid (linear α=0.7) for one query: the search tool of the ch08 full check
"""
from __future__ import annotations

import functools
import json

import numpy as np

from common import CACHE, FIQA_DIR, REPO, use_repo

use_repo()
DEPTH = 100   # same per-list candidate depth as the real system (src/config.py CANDIDATE_DEPTH)


def _cached_json(name: str, compute):
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    CACHE.mkdir(exist_ok=True)
    value = compute()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@functools.cache
def corpus() -> dict[str, str]:
    from src.beir_data import load_corpus
    return dict(load_corpus.__wrapped__(FIQA_DIR))


@functools.cache
def split(name: str = "dev"):
    from src.beir_data import load_split
    return load_split(name, FIQA_DIR)


@functools.cache
def bm25_index():
    from src.retrieval.bm25 import BM25Index
    c = corpus()
    return BM25Index().build(list(c), list(c.values()))


def bm25_run(split_name: str = "dev", depth: int = DEPTH) -> dict[str, list[list]]:
    """{qid: [[doc_id, score], ...]} reference BM25, best first."""
    def compute():
        from src.retrieval.bm25 import tokenize
        idx = bm25_index()
        queries, _ = split(split_name)
        return {q: [[d, s] for d, s in idx.search(list(dict.fromkeys(tokenize(t))), depth)] for q, t in queries.items()}
    return _cached_json(f"ref_bm25_{split_name}_{depth}", compute)


@functools.cache
def dense_matrix():
    """(ids, matrix) of the reference MiniLM corpus embeddings (row i = ids[i])."""
    from src.index.build_index import build_cache, load_cache
    got = load_cache("minilm")
    if got is None:
        print("\n[fullkit] building reference MiniLM corpus embeddings (~10 min, once)...")
        build_cache("minilm")
        got = load_cache("minilm")
    ids, _hashes, mat = got
    return ids, np.asarray(mat)


def dense_run(split_name: str = "dev", depth: int = DEPTH) -> dict[str, list[list]]:
    """{qid: [[doc_id, score], ...]} reference MiniLM dense retrieval, best first."""
    def compute():
        from src.retrieval.dense import get_embedder
        ids, mat = dense_matrix()
        queries, _ = split(split_name)
        qids = list(queries)
        qv = get_embedder("minilm").encode_queries([queries[q] for q in qids])
        out = {}
        for i, q in enumerate(qids):
            s = mat @ qv[i]
            top = np.argpartition(-s, depth)[:depth]
            out[q] = sorted(([ids[j], float(s[j])] for j in top), key=lambda x: (-x[1], x[0]))
        return out
    return _cached_json(f"ref_dense_{split_name}_{depth}", compute)


def hybrid_search(query: str, k: int = 10) -> list[str]:
    """The project's shipped retrieval for ONE query: BM25 top-100 + MiniLM top-100 → linear fusion α=0.7 → top-k ids."""
    from src.retrieval.bm25 import tokenize
    from src.retrieval.dense import get_embedder
    from src.retrieval.hybrid import linear
    bm25 = bm25_index().search(list(dict.fromkeys(tokenize(query))), DEPTH)
    ids, mat = dense_matrix()
    s = mat @ get_embedder("minilm").encode_queries([query])[0]
    top = np.argpartition(-s, DEPTH)[:DEPTH]
    dense = sorted(((ids[j], float(s[j])) for j in top), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in linear(bm25, dense, 0.7, top=k)]


def compound(limit: int = 100) -> list[dict]:
    """The compound (two-part) dev questions: {"question", "a": {"text"}, "b": {"text"}, "gold": {doc: "a"|"b"}}."""
    lines = (REPO / "data" / "compound_dev.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()][:limit]


def halves(rec: dict) -> dict[str, list[str]]:
    """{"a": [gold docs of half a], "b": [...]} for coverage_at_k."""
    out: dict[str, list[str]] = {"a": [], "b": []}
    for doc, h in rec["gold"].items():
        out[h].append(doc)
    return out
