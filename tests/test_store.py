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


# ---------------------------------------------------------------- async apply mode (ADR 0013)
def make_async_store(tmp_path=None, embed_fn=embed, **kw):
    return ShardStore(0, 1, state_dir=tmp_path, embed=embed_fn, use_cache=False, apply_mode="async",
                      **kw).load(CORPUS, ("minilm",))


def test_async_versions_compare_with_accepted_not_applied(tmp_path):
    s = make_async_store(tmp_path)
    assert s.upsert("a1", 1, "zebra bonds yield")["applied"]
    assert not s.upsert("a1", 1, "zebra bonds yield")["applied"]  # retry, maybe still queued
    assert s.upsert("a1", 3, "giraffe stocks")["applied"]
    late = s.upsert("a1", 2, "OLD text must not win")
    assert not late["applied"] and late["current_version"] == 3
    assert s.flush(5)
    assert s.docs["a1"] == "giraffe stocks"
    bm, dn = search_ids(s, "giraffe stocks")
    assert "a1" in bm and dn[0] == "a1"  # both indexes updated together
    s.close()


def test_async_acked_write_survives_crash_before_apply(tmp_path):
    s = make_async_store(tmp_path)
    s._applier.stop()  # simulate a crash: the applier never runs again
    assert s.upsert("c1", 1, "walrus pension crash probe")["applied"]  # acked = in the WAL
    assert s.delete("x-never", 1)["applied"]
    assert "c1" not in s.docs  # acked but not applied
    s.wal.close()
    s2 = make_store(tmp_path)  # restart (any mode): WAL replay embeds the text-only records
    assert s2.docs["c1"] == "walrus pension crash probe"
    bm, dn = search_ids(s2, "walrus pension crash probe")
    assert "c1" in bm and dn[0] == "c1"


def test_async_rejects_bad_payload_before_ack(tmp_path):
    s = make_async_store(tmp_path)
    with pytest.raises(ValueError):
        s.upsert("bad", 1, "   ")
    assert s._applier.counters["accepted"] == 0
    s.close()


def test_async_backpressure_keeps_visible_lag_under_one_second(tmp_path):
    import threading
    import time as _t

    def slow_embed(model, texts):  # ~50 ms per batch: the applier is the bottleneck
        _t.sleep(0.05)
        return embed(model, texts)

    # No WAL (state_dir=None): writers are instant, so only back-pressure can hold them back.
    s = make_async_store(None, embed_fn=slow_embed, max_lag_s=0.2, batch_max=4)

    def writer(w):
        for i in range(25):
            s.upsert(f"bp-{w}-{i}", 1, f"backpressure probe {w} {i}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.flush(10)
    st = s._applier.stats()
    assert st["applied"] == 100 and st["backpressure_waits"] > 0  # writers were slowed down...
    assert st["lag_ms"]["max"] < 1000, st  # ...so every write became searchable within 1 s
    s.close()


def test_async_migration_with_concurrent_writes_covers_every_doc(tmp_path):
    """Migration under async apply: queued writes, dual-write and backfill must cover every doc,
    and each doc's vector in the NEW index must be the vector of its LATEST text."""
    import threading
    import time as _t

    def slow_embed(model, texts):  # slow enough that writes queue up during the build
        _t.sleep(0.01)
        return embed(model, texts)

    s = make_async_store(tmp_path, embed_fn=slow_embed, batch_max=8)
    stop = threading.Event()

    def writer(w):
        i = 0
        while not stop.is_set():
            s.upsert(f"mig-{w}-{i % 15}", i // 15 + 1, f"migration writer {w} doc {i % 15} round {i // 15}")
            i += 1

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(3)]
    for t in threads:
        t.start()
    _t.sleep(0.2)
    assert s.start_build("bge-small", batch=8)["status"] == "started"
    deadline = _t.time() + 10
    while not s.ready["bge-small"] and _t.time() < deadline:
        _t.sleep(0.02)
    _t.sleep(0.2)  # keep writing after the build is ready (dual-write only from here)
    stop.set()
    for t in threads:
        t.join()
    assert s.ready["bge-small"] and s.flush(10)
    new, old = s.dense["bge-small"], s.dense["minilm"]
    assert len(new) == len(old) == len(s.docs)
    for d, text in s.docs.items():
        assert d in new
        top = new.search(qembed("bge-small", text), 1)
        assert top[0][1] == pytest.approx(1.0, abs=1e-5), d  # an exact match exists for the latest text
    s.close()
