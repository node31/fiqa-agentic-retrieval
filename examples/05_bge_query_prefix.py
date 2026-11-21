"""Toy 05: the BGE query instruction prefix. Asymmetric retrieval in one line of config.

Concept: some embedding models were trained with an instruction on the QUERY side only.
Real code: src/config.py MODELS['bge-small'].query_prefix    Doc: docs/topics/02_dense_retrieval.md
"""
import numpy as np
from _common import TOY_DOCS, banner, load_st_model, note, step, try_this

from src.config import MODELS

banner("Toy 05: does the BGE query prefix change rankings?")
spec = MODELS["bge-small"]
model = load_st_model(spec.hf_name)
if model is None:
    print("  bge-small not available offline: run `python -m src.index.build_index --models bge-small` once.")
    raise SystemExit(0)
ids, texts = list(TOY_DOCS), list(TOY_DOCS.values())
D = model.encode(texts, normalize_embeddings=True)
step(1, f"Prefix used for queries: {spec.query_prefix!r}")
changes = []
for q in ["what happens if I short a stock", "is mortgage interest tax deductible"]:
    plain = D @ model.encode([q], normalize_embeddings=True)[0]
    pref = D @ model.encode([spec.query_prefix + q], normalize_embeddings=True)[0]
    step(2, f"Query {q!r}")
    print(f"     no prefix  : top-3 {[ids[i] for i in np.argsort(-plain)[:3]]}  scores {np.sort(plain)[::-1][:3].round(3)}")
    print(f"     with prefix: top-3 {[ids[i] for i in np.argsort(-pref)[:3]]}  scores {np.sort(pref)[::-1][:3].round(3)}")
    changes.append((ids[np.argmax(plain)] != ids[np.argmax(pref)],
                    list(np.argsort(-plain)[:3]) != list(np.argsort(-pref)[:3]), float(np.mean(pref - plain))))
note(f"Top-1 changed for {sum(c[0] for c in changes)}/{len(changes)} queries, top-3 for "
     f"{sum(c[1] for c in changes)}/{len(changes)}; the prefix shifted cosines by "
     f"{', '.join(f'{c[2]:+.3f}' for c in changes)} on average. Absolute cosines are not comparable across query "
     "encodings; only the ORDER matters. The BGE card recommends the prefix for short-query -> passage retrieval.")
note("The prefix is QUERY-side only: document vectors (and the embedding cache) do not depend on it, so changing it "
     "needs no re-index. That is why it lives in the model registry next to the model name: queries must be "
     "encoded the way the stored documents expect.")
try_this("Measure dense-bge-small nDCG@10 on dev with query_prefix='' (edit config, rebuild nothing: "
         "only queries change!). What is the delta?",
         "Why must the prefix NOT be added to documents? Try it and compare cosine scores.",
         "MiniLM has no prefix. What would happen if you added BGE's prefix to MiniLM queries?")
