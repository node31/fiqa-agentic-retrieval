"""Cross-encoder reranking: rescore a short candidate list with a slower, stronger model.

WHAT
    A bi-encoder (our dense models) embeds query and passage SEPARATELY, so
    passages can be pre-computed and searched in 1 ms. A cross-encoder reads
    "[CLS] query [SEP] passage" TOGETHER, so every token of the query can attend
    to every token of the passage. It is much more accurate, but it costs a full
    transformer pass PER (query, passage) pair, so it can only be used on the top-k.

THE TRADE-OFF WE MEASURE (results/ablation_rerank_minilm.json)
    quality vs. rerank depth k ∈ {10, 20, 50}, and latency grows ~linearly in k.
    See docs/06_reranking.md and examples/09_bi_vs_cross_encoder.py.

PITFALL
    ms-marco cross-encoders output unbounded logits, not probabilities. Only
    their ORDER matters here, so we never mix them with other scores.

RESULT ON FIQA
    ms-marco-MiniLM-L-6-v2 costs ~29 ms per (query, passage) pair on CPU, and on
    FiQA dev it did NOT help: nDCG@10 0.4058 when reranking the fused top-10,
    below the 0.419 of linear fusion alone. (MS MARCO is web search, not finance;
    a stronger model is not automatically a better one out of domain.) So reranking
    is optional and off by default (``rerank_k=0`` in src/service/router.py).

HOW TO READ THIS FILE
    get_reranker() loads the model once per process (thread-safe, lazy);
    rerank() scores every candidate that has text and re-sorts them.
"""
from __future__ import annotations

import threading

from src.config import RERANKER_NAME

_MODEL = None  # the loaded CrossEncoder, created on first use
_LOCK = threading.Lock()


def get_reranker():
    # Lazy, process-wide singleton (same pattern as dense.get_embedder).
    # WHY the lock: without it, two request threads arriving together could both
    # see _MODEL is None and load the model twice.
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            # WHY import here: sentence_transformers pulls in torch (seconds to import);
            # services that never rerank should not pay that at startup.
            from sentence_transformers import CrossEncoder

            # max_length=256: query + passage together are cut to 256 model tokens,
            # which bounds the cost of each pair (long passages lose their tail).
            _MODEL = CrossEncoder(RERANKER_NAME, device="cpu", max_length=256)
        return _MODEL


def rerank(query: str, candidates: list[tuple[str, float]], texts: dict[str, str]) -> list[tuple[str, float]]:
    """Reorder ``candidates`` by cross-encoder score. Candidates without text keep their order at the end."""
    # candidates: [(doc_id, fused_score), ...] from hybrid.fuse, usually the top-k.
    # texts: doc_id -> passage text. Cost is one model pass per candidate with
    # text (~29 ms each on CPU), so the caller decides k to bound latency.
    have = [d for d, _ in candidates if d in texts]
    if not have:
        return list(candidates)
    # predict() takes (query, passage) pairs and returns one relevance logit per
    # pair (higher = more relevant), in the same order as the input.
    scores = get_reranker().predict([(query, texts[d]) for d in have], batch_size=32)
    # Sort by cross-encoder score, ties by doc id. The old fused scores are dropped.
    ranked = sorted(zip(have, map(float, scores)), key=lambda x: (-x[1], x[0]))
    # Candidates we have no text for cannot be scored; they keep their ORIGINAL
    # fused score and go after all reranked docs.
    # PITFALL: the returned scores therefore mix two scales (logits first, fused
    # scores after) and may not be descending; trust the list order, not the scores.
    rest = [(d, s) for d, s in candidates if d not in texts]
    return ranked + rest
