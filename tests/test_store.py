"""In-process tests of the shard core: idempotent versioned writes, WAL recovery,
lexical/dense consistency, migration backfill race, sharded == single-node.

Uses HashEmbedder (no model download) and a 24-passage fixture corpus.
"""
import asyncio
import json
from pathlib import Path

import pytest

from src.index.sharding import shard_of
from src.retrieval.bm25 import tokenize
from src.retrieval.dense import HashEmbedder
from src.service.router import LocalShardClient, Router
from src.service.store import ShardStore

FIX = Path(__file__).parent / "fixtures" / "tinyfiqa"
CORPUS = {json.loads(l)["_id"]: json.loads(l)["text"] for l in (FIX / "corpus.jsonl").read_text().splitlines()}
EMB = {"minilm": HashEmbedder(seed="minilm"), "bge-small": HashEmbedder(seed="bge-small")}


def embed(model, texts):
    return EMB[model].encode_docs(list(texts))


def qembed(model, text):
    return EMB[model].encode_queries([text])[0]


def make_store(tmp_path=None, corpus=CORPUS, sid=0, n=1, models=("minilm",)):
    return ShardStore(sid, n, state_dir=tmp_path, embed=embed, use_cache=False).load(corpus, models)


def search_ids(store, text, model="minilm"):
    terms = list(dict.fromkeys(tokenize(text)))
    r = store.search(terms, 10, vector=qembed(model, text), model=model)
    return [d for d, _ in r["bm25"]], [d for d, _ in r["dense"]]


def test_upsert_is_idempotent_and_versioned(tmp_path):
    s = make_store(tmp_path)
    assert s.upsert("new1", 1, "zebra bonds yield")["applied"]
    assert not s.upsert("new1", 1, "zebra bonds yield")["applied"]  # exact replay
    assert s.upsert("new1", 3, "giraffe stocks")["applied"]
    assert not s.upsert("new1", 2, "OLD text must not win")["applied"]  # out-of-order older write
    assert s.docs["new1"] == "giraffe stocks"
    assert s.counters == {"applied": 2, "noop": 2, "replayed": 0}


def test_delete_tombstone_blocks_older_upsert(tmp_path):
    s = make_store(tmp_path)
    s.upsert("x", 1, "zebra")
    assert s.delete("x", 2)["applied"]
    assert not s.upsert("x", 1, "zebra resurrected")["applied"]
    assert "x" not in search_ids(s, "zebra")[0]


def test_lexical_and_dense_stay_consistent(tmp_path):
    s = make_store(tmp_path)
    s.upsert("z", 1, "zebra stripes finance")
    bm, dn = search_ids(s, "zebra stripes finance")
    assert "z" in bm and dn[0] == "z"
    s.delete("z", 2)
    bm, dn = search_ids(s, "zebra stripes finance")
    assert "z" not in bm and "z" not in dn


def test_wal_replay_recovers_after_crash(tmp_path):
    s = make_store(tmp_path)
    s.upsert("w1", 1, "walrus pension")
    s.upsert("w2", 1, "walrus annuity")
    s.delete("w2", 2)
    # simulate a crash in the middle of appending a record: torn last line
    with open(tmp_path / "wal.jsonl", "a") as f:
        f.write('{"op":"upsert","doc_id":"w3","vers')
    s.wal.close()
    s2 = make_store(tmp_path)  # "restart": base corpus + WAL replay
    assert s2.docs.get("w1") == "walrus pension"
    assert "w2" not in s2.docs and s2.versions["w2"] == (2, True)
    assert "w3" not in s2.docs  # torn, never acknowledged -> correctly dropped
    assert s2.counters["replayed"] == 3
    # replay is idempotent: replaying again changes nothing
    assert not s2.upsert("w1", 1, "walrus pension")["applied"]


def test_migration_backfill_dual_write_and_race(tmp_path):
    s = make_store(tmp_path)
    s.upsert("m1", 1, "mongoose margin call")
    # start the build but capture the background thread so the test controls the race
    import threading
    orig = threading.Thread
    captured = {}

    class Capture(orig):
        def __init__(self, target=None, args=(), **kw):
            captured["fn"], captured["args"] = target, args
            super().__init__(target=lambda: None)

    threading.Thread = Capture
    try:
        s.start_build("bge-small")
    finally:
        threading.Thread = orig
    # writes during migration are dual-written...
    s.upsert("m1", 2, "mongoose updated text")
    s.upsert("m2", 1, "meerkat new doc")
    assert "m2" in s.dense["bge-small"] and not s.ready["bge-small"]
    # ...then the backfill runs with its OLD snapshot (m1 at version 1)
    captured["fn"](*captured["args"])
    assert s.ready["bge-small"]
    assert len(s.dense["bge-small"]) == len(s.docs)
    # m1's vector in the new index must be the NEW text's vector, not the stale snapshot's
    top = s.dense["bge-small"].search(qembed("bge-small", "mongoose updated text"), 1)
    assert top[0][0] == "m1" and top[0][1] == pytest.approx(1.0, abs=1e-5)


def test_sharded_router_equals_single_node():
    single = Router([LocalShardClient(make_store())], query_embed=qembed)
    n = 3
    stores = [make_store(corpus={d: t for d, t in CORPUS.items() if shard_of(d, n) == i}, sid=i, n=n)
              for i in range(n)]
    sharded = Router([LocalShardClient(s) for s in stores], query_embed=qembed)
    for q in ["how does short selling work", "roth ira taxes", "emergency fund months"]:
        for strat in ("bm25", "dense", "hybrid"):
            a = asyncio.run(single.search(q, strategy=strat))["results"]
            b = asyncio.run(sharded.search(q, strategy=strat))["results"]
            assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b], (q, strat)
            assert [r["score"] for r in a] == pytest.approx([r["score"] for r in b], rel=1e-4)


def test_router_reports_partial_results_when_a_shard_fails():
    n = 3
    stores = [make_store(corpus={d: t for d, t in CORPUS.items() if shard_of(d, n) == i}, sid=i, n=n)
              for i in range(n)]

    class Dead:
        async def call(self, op, payload, timeout):
            raise ConnectionError("shard down")

    r = Router([LocalShardClient(stores[0]), Dead(), LocalShardClient(stores[2])], query_embed=qembed)
    out = asyncio.run(r.search("short selling"))
    assert out["partial"] and out["missing_shards"] == [1] and out["results"]
    strict = Router(r.shards, query_embed=qembed, failure_policy="fail")
    with pytest.raises(Exception):
        asyncio.run(strict.search("short selling"))


def test_wal_write_after_torn_tail_survives_next_restart(tmp_path):
    """Regression: crash mid-append -> recover -> ack a new write -> restart. The new write must survive."""
    from src.service.wal import WAL

    p = tmp_path / "wal.jsonl"
    w = WAL(p, fsync=False)
    w.append({"op": "upsert", "doc_id": "a", "version": 1})
    w.close()
    with open(p, "ab") as f:  # simulate the crash: half a record, no newline
        f.write(b'{"op":"upsert","doc_id":"b","ver')
    w = WAL(p, fsync=False)  # recovery
    assert w.truncated_bytes > 0
    assert [r["doc_id"] for r in w.replay()] == ["a"]
    w.append({"op": "upsert", "doc_id": "c", "version": 1})  # acked after recovery
    w.close()
    assert [r["doc_id"] for r in WAL(p, fsync=False).replay()] == ["a", "c"]
