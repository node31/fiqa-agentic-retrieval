"""Integration tests over REAL processes: 3 shard servers + 1 router talking HTTP.

Proves the Part B guarantees end to end:
  * idempotent writes (replay + out-of-order versions) through the router
  * freshness: a write is searchable as soon as its ack returns (we assert < 1 s)
  * a dead shard -> partial results, flagged to the caller
  * crash recovery: kill a shard, restart it, acknowledged writes are still there (WAL)

Runs on the 24-doc fixture corpus with EMBED_BACKEND=hash, so it needs no model
download and takes a few seconds.
"""
import time
from pathlib import Path

import httpx
import pytest

from src.index.sharding import shard_of
from src.service.cluster import Cluster

FIX = Path(__file__).parent / "fixtures" / "tinyfiqa"


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("cluster")
    env = {"FIQA_DATA_DIR": str(FIX), "EMBED_BACKEND": "hash", "ARTIFACTS_DIR": str(root / "artifacts")}
    c = Cluster(3, base_port=18101, router_port=18000, state_root=root, env=env, threads=1, timeout_s=2.0)
    c.start(wait_s=120)
    # PITFALL: a fresh httpx.Client per call builds a new SSL context (~0.6 s on Windows).
    # Reusing one client (keep-alive) is what makes latency numbers meaningful.
    c.http = httpx.Client(timeout=10)
    yield c
    c.http.close()
    c.stop()


def post(c, path, **body):
    r = c.http.post(f"{c.router_url}{path}", json=body)
    r.raise_for_status()
    return r.json()


def ids(c, query, strategy="hybrid"):
    return [r["doc_id"] for r in post(c, "/search", query=query, strategy=strategy)["results"]]


def test_idempotent_writes_through_router(cluster):
    a = post(cluster, "/upsert", doc_id="idem-1", version=1, text="quokka dividend strategy")
    b = post(cluster, "/upsert", doc_id="idem-1", version=1, text="quokka dividend strategy")
    assert a["applied"] and not b["applied"]  # replay of the same (id, version) = no-op
    post(cluster, "/upsert", doc_id="idem-1", version=5, text="quokka bond ladder")
    old = post(cluster, "/upsert", doc_id="idem-1", version=4, text="quokka STALE")
    assert not old["applied"] and old["current_version"] == 5  # older never overwrites newer
    top = post(cluster, "/search", query="quokka", strategy="bm25", include_text=True)["results"][0]
    assert top["doc_id"] == "idem-1" and top["text"] == "quokka bond ladder"


def test_freshness_write_searchable_immediately_after_ack(cluster):
    """Freshness guarantee: a write is searchable by the FIRST search issued after its ack.

    We also bound write-ack + 3 searches at < 1 s to catch regressions (e.g. async apply).
    """
    lags = []
    for i in range(5):
        doc = f"fresh-{i}"
        t0 = time.perf_counter()
        post(cluster, "/upsert", doc_id=doc, version=1, text=f"axolotl{i} annuity freshness probe")
        # no sleep, no retry: the very first search after the ack must see it
        for strategy in ("bm25", "dense", "hybrid"):
            assert doc in ids(cluster, f"axolotl{i} annuity freshness probe", strategy), strategy
        lags.append(time.perf_counter() - t0)
    assert max(lags) < 1.0, lags


def test_delete_is_visible_to_both_indexes(cluster):
    post(cluster, "/upsert", doc_id="del-1", version=1, text="narwhal municipal bonds")
    assert "del-1" in ids(cluster, "narwhal municipal bonds", "bm25")
    post(cluster, "/delete", doc_id="del-1", version=2)
    for strategy in ("bm25", "dense"):
        assert "del-1" not in ids(cluster, "narwhal municipal bonds", strategy)


def test_crash_recovery_and_partial_results(cluster):
    doc = next(f"crash-{i}" for i in range(100) if shard_of(f"crash-{i}", 3) == 1)
    post(cluster, "/upsert", doc_id=doc, version=1, text="pangolin treasury inflation")
    cluster.kill("shard1")  # hard kill: no graceful shutdown, in-memory state is gone
    out = post(cluster, "/search", query="short selling borrowing shares")
    assert out["partial"] and out["missing_shards"] == [1] and out["results"]
    cluster.start_shard(1)  # restart -> base corpus + WAL replay
    cluster.wait_healthy([cluster.shard_url(1)], 120)
    out = post(cluster, "/search", query="pangolin treasury inflation", strategy="bm25")
    assert not out["partial"] and out["results"][0]["doc_id"] == doc
