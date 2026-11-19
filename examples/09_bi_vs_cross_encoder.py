"""Toy 09: bi-encoder vs cross-encoder. Why rerankers are accurate, slow, and used only on the top-k.

Concept: two-stage retrieval (cheap recall, then expensive precision).
Real code: src/retrieval/rerank.py      Doc: docs/06_reranking.md
"""
import time

import numpy as np
from _common import TOY_DOCS, banner, load_st_model, note, step, try_this

banner("Toy 09: bi-encoder (embed separately) vs cross-encoder (read together)")
bi = load_st_model("sentence-transformers/all-MiniLM-L6-v2")
try:
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
except Exception as e:  # noqa: BLE001
    ce = None
    print(f"  cross-encoder unavailable offline ({type(e).__name__}); run the rerank ablation once to cache it.")
if bi is None or ce is None:
    raise SystemExit(0)
ids, texts = list(TOY_DOCS), list(TOY_DOCS.values())
q = "what is the risk if I short a stock"
t0 = time.perf_counter()
D = bi.encode(texts, normalize_embeddings=True)  # done OFFLINE in real life
t_docs = time.perf_counter() - t0
t0 = time.perf_counter()
s_bi = D @ bi.encode([q], normalize_embeddings=True)[0]
t_bi = time.perf_counter() - t0
step(1, f"Bi-encoder: docs embedded once ({t_docs * 1e3:.0f} ms, offline); query costs ONE encode: {t_bi * 1e3:.1f} ms")
t0 = time.perf_counter()
s_ce = ce.predict([(q, t) for t in texts])
t_ce = time.perf_counter() - t0
step(2, f"Cross-encoder: one transformer pass PER (query, doc) pair: {t_ce * 1e3:.1f} ms for {len(texts)} docs")
print(f"     {'doc':4s} {'bi cos':>7s} {'CE logit':>9s}  text")
for i in np.argsort(-s_ce)[:5]:
    print(f"     {ids[i]:4s} {s_bi[i]:7.3f} {s_ce[i]:9.2f}  {texts[i][:55]}")
note("The cross-encoder sees query and passage together, so 'risk' in the query can attend to 'loss is "
     "unlimited' in d2. Its scores are logits: only their ORDER means anything.")
note(f"Cost scales with k: reranking 100 FiQA passages = 100 pair passes ~ {t_ce / len(texts) * 100 * 1e3:.0f} ms "
     "here. That's why we rerank only the top-k and sweep k (results/ablation_rerank_minilm.json).")
try_this("Rerank only the top-3 bi-encoder hits. Did the final #1 change?",
         "Time ce.predict with batch_size=1 vs 32. Why does batching help on CPU?",
         "Can a cross-encoder index a corpus offline the way a bi-encoder can? Why not?")
