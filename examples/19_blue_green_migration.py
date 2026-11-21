"""Toy 19: blue/green embedding migration. Dual-write, backfill race, cutover, rollback.

Concept: build the new index alongside the old one, flip a flag, and keep the old one warm for rollback.
Real code: ShardStore.start_build/_backfill, Router.cutover/rollback    Doc: docs/topics/10_model_migration.md
"""
import threading

from _common import TOY_DOCS, banner, note, step, try_this

from src.retrieval.dense import HashEmbedder
from src.service.store import ShardStore

banner("Toy 19: migrate 'minilm' -> 'bge-small' (hash embedders stand in for the models)")
embs = {"minilm": HashEmbedder(seed="minilm"), "bge-small": HashEmbedder(seed="bge-small")}
s = ShardStore(embed=lambda m, t: embs[m].encode_docs(list(t)), use_cache=False).load(TOY_DOCS, ["minilm"])
step(1, f"Serving with minilm: indexes = {list(s.dense)}")

# Capture the backfill thread so WE control when it runs (to show the race deterministically).
captured, Orig = {}, threading.Thread
threading.Thread = lambda target=None, args=(), **kw: captured.update(fn=target, args=args) or Orig(target=lambda: None)
s.start_build("bge-small")
threading.Thread = Orig
step(2, f"start_build: empty bge-small index created, ready={s.ready['bge-small']}. Dual-write is ON:")
s.upsert("d3", 1, "UPDATED: prepaying a mortgage is a risk-free return")
s.upsert("new", 1, "brand new passage written during migration")
print(f"     bge-small index now holds {len(s.dense['bge-small'])} docs (only the 2 written since migrate)")
step(3, "Backfill runs with the snapshot taken at migrate time (d3 still at version 0 there)")
captured["fn"](*captured["args"])
v_new = embs["bge-small"].encode_queries(["UPDATED: prepaying a mortgage is a risk-free return"])[0]
top = s.dense["bge-small"].search(v_new, 1)[0]
print(f"     ready={s.ready['bge-small']}, size={len(s.dense['bge-small'])}/{len(s.docs)}; d3 vector matches "
      f"NEW text: {top[0] == 'd3' and top[1] > 0.999}")
note("The race: the backfill read d3@v0 before the update, and must NOT overwrite the v1 vector from the "
     "dual-write. The check `doc not in index and version unchanged` makes that safe.")
step(4, "Cutover = router flips active_model. Rollback = flip back. Both indexes are live, so no query fails.")
note("In-flight queries keep the model they started with (captured once per request). Query vectors from "
     "model A are never compared with doc vectors from model B: that would be silent garbage, not an error.")
try_this("Run `python scripts/migration_demo.py`: 3 shards, live queries and writes, migrate/cutover/rollback.",
         "When is rollback no longer instant? (hint: `finalize` drops the old index).",
         "What if a shard restarts mid-migration? Read ShardStore.load + models.json.")
