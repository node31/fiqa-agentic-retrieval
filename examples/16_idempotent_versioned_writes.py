"""Toy 16: idempotent, versioned writes. Replays and out-of-order delivery are harmless.

Concept: (doc_id, version) makes every write safe to retry; tombstones stop deleted docs resurrecting.
Real code: src/service/store.py::ShardStore._write     Doc: docs/09_live_updates_consistency.md
"""
from _common import TOY_DOCS, banner, note, step, try_this

from src.retrieval.dense import HashEmbedder
from src.service.store import ShardStore

banner("Toy 16: versioned upserts + deletes on a real ShardStore (hash embedder, no WAL)")
emb = HashEmbedder()
store = ShardStore(embed=lambda m, t: emb.encode_docs(list(t)), use_cache=False).load(TOY_DOCS, ["minilm"])


def w(label, res):
    print(f"     {label:44s} -> applied={res['applied']}" + (f" ({res.get('reason')})" if not res["applied"] else ""))


step(1, "A client upserts v1, then RETRIES it because the ack was lost (network blip)")
w("upsert(x, v1, 'cheap ETF fees')", store.upsert("x", 1, "cheap ETF fees"))
w("upsert(x, v1, 'cheap ETF fees')  [retry]", store.upsert("x", 1, "cheap ETF fees"))
step(2, "Messages arrive out of order: v3 is delivered before v2")
w("upsert(x, v3, 'ETF expense ratios explained')", store.upsert("x", 3, "ETF expense ratios explained"))
w("upsert(x, v2, 'OLD text')  [late]", store.upsert("x", 2, "OLD text"))
print(f"     stored text: {store.docs['x']!r}  (v3 kept: older never overwrites newer)")
step(3, "Delete at v4, then a delayed v3 upsert shows up")
w("delete(x, v4)", store.delete("x", 4))
w("upsert(x, v3, ...)  [very late]", store.upsert("x", 3, "ETF expense ratios explained"))
print(f"     'x' in docs: {'x' in store.docs}; tombstone: versions['x'] = {store.versions['x']}")
note("Without the tombstone (version, deleted=True) the late v3 would RESURRECT a deleted document.")
note(f"Counters: {store.counters}. 'noop' writes did no work: no embedding, no WAL append, no lock.")
note("Idempotency is what makes the router's retry loop safe: it can resend a write whose ack timed out.")
try_this("Who should assign versions: client, router or shard? What breaks with wall-clock timestamps?",
         "Tombstones live forever here. Design a GC rule (hint: 'older than the max replay window').",
         "Send the same (id, version) with DIFFERENT text. What should happen? What does happen?")
