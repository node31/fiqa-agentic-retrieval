"""Source root for the FiQA distributed + agentic retrieval project.

Packages:
    retrieval/  BM25, dense, hybrid fusion, reranking  (Part A)
    index/      offline indexing: embedding cache, sharding  (Part A/B)
    service/    shard server, router, WAL, ingestion  (Part B)
    agent/      LLM backends, single agent, multi-agent orchestrator  (Part C)

Top-level modules (shared by all parts):
    config.py     paths, model registry, tuned defaults (read by everything)
    beir_data.py  load the FiQA corpus, queries and qrels (the answer key)
    metrics.py    nDCG@10 / Recall@10 / MRR@10 / Coverage@10, written by hand
    eval.py       the evaluation harness: quality + latency + memory for one configuration
    search.py     tiny CLI that sends one query to the running router
    util.py       percentile and memory (RSS) helpers

This file also makes ``src`` a Python package, which is why modules import each
other as ``from src.config import ...`` and are run as ``python -m src.eval``.

Start reading at docs/LEARNING_PATH.md.
"""
