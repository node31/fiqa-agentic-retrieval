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
"""
from __future__ import annotations

import threading

from src.config import RERANKER_NAME

_MODEL = None
_LOCK = threading.Lock()


def get_reranker():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from sentence_transformers import CrossEncoder

            _MODEL = CrossEncoder(RERANKER_NAME, device="cpu", max_length=256)
        return _MODEL


def rerank(query: str, candidates: list[tuple[str, float]], texts: dict[str, str]) -> list[tuple[str, float]]:
    """Reorder ``candidates`` by cross-encoder score. Candidates without text keep their order at the end."""
    have = [d for d, _ in candidates if d in texts]
    if not have:
        return list(candidates)
    scores = get_reranker().predict([(query, texts[d]) for d in have], batch_size=32)
    ranked = sorted(zip(have, map(float, scores)), key=lambda x: (-x[1], x[0]))
    rest = [(d, s) for d, s in candidates if d not in texts]
    return ranked + rest
