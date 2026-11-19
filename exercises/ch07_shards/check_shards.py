"""Checker for ch07. Don't edit. Run: python -m pytest exercises/ch07_shards"""
import json

import pytest

import mini_shards as mine
from common import TINY
from src.index.sharding import shard_of as ref_shard_of
from src.retrieval.bm25 import BM25Index, tokenize

QUERIES = ["how does short selling work", "roth ira vs traditional 401k taxes",
           "how big should an emergency fund be", "pay off mortgage early or invest", "dividend etf fees"]


def _tiny():
    out = {}
    for line in (TINY / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["_id"]] = (r.get("title", "") + " " + r["text"]).strip()
    return out


# ---- 1. partitioning ----------------------------------------------------------------------------
def check_shard_of_matches_crc32():
    for i in range(500):
        for n in (2, 3, 4):
            assert mine.shard_of(str(i), n) == ref_shard_of(str(i), n)


# ---- 2. router with global stats == single node ------------------------------------------------------
@pytest.mark.parametrize("n_shards", [2, 3])
def check_router_equals_single_node(n_shards):
    corpus = _tiny()
    shards = [mine.Shard({d: t for d, t in corpus.items() if mine.shard_of(d, n_shards) == i}) for i in range(n_shards)]
    single = BM25Index().build(list(corpus), list(corpus.values()))
    for q in QUERIES:
        got = mine.router_search(shards, q, 10, use_global_stats=True)
        want = single.search(list(dict.fromkeys(tokenize(q))), 10)
        assert [d for d, _ in got] == [d for d, _ in want], f"{n_shards} shards, query {q!r}: ranking differs"
        assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-4)


def check_local_stats_bug_and_global_fix():
    a = {"a1": "dividend stocks pay cash", "a2": "dividend yield explained", "a3": "dividend tax rules",
         "a4": "dividend reinvestment plan", "a5": "dividend etf fees"}
    b = {"b1": "dividend etf fees", "b2": "bond ladder basics", "b3": "roth ira limits",
         "b4": "mortgage rate lock", "b5": "credit card rewards"}
    shards = [mine.Shard(a), mine.Shard(b)]
    loc = dict(mine.router_search(shards, "dividend etf", 10, use_global_stats=False))
    glob = dict(mine.router_search(shards, "dividend etf", 10, use_global_stats=True))
    assert loc["a5"] != pytest.approx(loc["b1"]), "identical texts on different shards should DIFFER with local stats"
    assert glob["a5"] == pytest.approx(glob["b1"]), "...and be EQUAL with global stats"


# ---- 3. versioned writes (toy 16's sequence) --------------------------------------------------------
def check_versioned_writes_sequence():
    s = mine.VersionedStore()
    assert s.upsert("x", 1, "cheap ETF fees") is True
    assert s.upsert("x", 1, "cheap ETF fees") is False, "a retry of the same version must be a no-op"
    assert s.upsert("x", 3, "ETF expense ratios explained") is True
    assert s.upsert("x", 2, "OLD text") is False, "an older version must never overwrite a newer one"
    assert s.get("x") == "ETF expense ratios explained"
    assert s.delete("x", 4) is True
    assert s.upsert("x", 3, "late") is False, "the tombstone must block a late older upsert (no resurrection)"
    assert s.get("x") is None
    assert s.get("never-seen") is None


# ---- 4. WAL ------------------------------------------------------------------------------------------
def check_wal_append_and_replay(tmp_path):
    w = mine.WAL(tmp_path / "wal.jsonl")
    w.append({"op": "upsert", "doc_id": "a", "version": 1})
    w.append({"op": "delete", "doc_id": "a", "version": 2})
    w.close()
    assert [r["op"] for r in mine.WAL(tmp_path / "wal.jsonl").replay()] == ["upsert", "delete"]


def check_wal_survives_crash_then_write_then_restart(tmp_path):
    p = tmp_path / "wal.jsonl"
    w = mine.WAL(p)
    w.append({"op": "upsert", "doc_id": "a", "version": 1})
    w.close()
    with open(p, "ab") as f:  # the crash: half a record, no newline
        f.write(b'{"op":"upsert","doc_id":"b","ver')
    w = mine.WAL(p)  # recovery #1
    assert [r["doc_id"] for r in w.replay()] == ["a"], "the torn record was never acked: it must not come back"
    w.append({"op": "upsert", "doc_id": "c", "version": 1})  # acked AFTER recovery
    w.close()
    assert [r["doc_id"] for r in mine.WAL(p).replay()] == ["a", "c"], \
        "the write acked after recovery was lost: did you truncate the torn tail when opening?"


# ---- FULL checks (python -m pytest exercises\\ch07_shards --full): 3 shards over all 57,638 docs ---------------
@pytest.fixture(scope="module")
def full_shards():
    import fullkit
    corpus = fullkit.corpus()
    return [mine.Shard({d: t for d, t in corpus.items() if mine.shard_of(d, 3) == i}) for i in range(3)]


@pytest.mark.full
def check_full_shards_are_balanced(full_shards):
    import fullkit
    sizes = [sum(1 for d in fullkit.corpus() if mine.shard_of(d, 3) == i) for i in range(3)]
    assert sum(sizes) == 57_638
    assert max(sizes) / min(sizes) < 1.03, f"crc32 % 3 should split evenly, got {sizes}"


@pytest.mark.full
def check_full_router_equals_single_node_on_dev(full_shards):
    """500 dev queries: 3 shards + global stats must give the single-node BM25 top-10 (same docs, same scores)."""
    import fullkit
    queries, _ = fullkit.split("dev")
    ref = fullkit.bm25_run("dev")
    same = total = 0
    for qid, q in queries.items():
        got = dict(mine.router_search(full_shards, q, 10, use_global_stats=True))
        want = dict((d, sc) for d, sc in ref[qid][:10])
        common_ids = set(got) & set(want)
        same += len(common_ids)
        total += len(want)
        for d in common_ids:
            assert got[d] == pytest.approx(want[d], rel=1e-4), f"{qid}/{d}: score differs from single node"
    # WHY not 100%: the reference uses float32, so exact ties at rank 10 can swap.
    assert same / total > 0.99, f"only {same / total:.1%} of the top-10 docs match the single node"


@pytest.mark.full
def check_full_local_stats_change_rankings(full_shards):
    """With local stats the same 500 queries get DIFFERENT top-10s for a noticeable share. The bug is real."""
    import fullkit
    queries, _ = fullkit.split("dev")
    changed = sum(
        [d for d, _ in mine.router_search(full_shards, q, 10, True)]
        != [d for d, _ in mine.router_search(full_shards, q, 10, False)] for q in queries.values())
    print(f"\nlocal stats changed the top-10 order for {changed}/500 dev queries")
    assert changed > 0
