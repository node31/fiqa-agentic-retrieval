"""Integration tests for APPLY_MODE=async over REAL processes: 3 shards + 1 router.

The sync-mode guarantees live in test_integration_service.py ("searchable by the first search after
the ack"). Async mode promises something weaker and different, and this file proves exactly that:
  * the ack comes back BEFORE the write is applied (reply says "visible": False)
  * every acked write becomes searchable within 1 s (we poll and measure the lag)
  * retries / older versions are still no-ops, even while the newer write is only queued
  * crash recovery: writes acked but not yet applied (text-only WAL records) survive a hard kill

Tiny fixture corpus + EMBED_BACKEND=hash, so no model download. Ports 18200 / 18301+ (the sync
integration tests use 18000 / 18101+).
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
    root = tmp_path_factory.mktemp("cluster_async")
    env = {"FIQA_DATA_DIR": str(FIX), "EMBED_BACKEND": "hash", "ARTIFACTS_DIR": str(root / "artifacts"),
           "APPLY_MODE": "async"}
    c = Cluster(3, base_port=18301, router_port=18200, state_root=root, env=env, threads=1, timeout_s=2.0)
    c.start(wait_s=120)
    c.http = httpx.Client(timeout=10)
    yield c
    c.http.close()
    c.stop()


def post(c, path, **body):
    r = c.http.post(f"{c.router_url}{path}", json=body)
    r.raise_for_status()
    return r.json()


def ids(c, query, strategy="bm25"):
    return [r["doc_id"] for r in post(c, "/search", query=query, strategy=strategy)["results"]]


def visible_after(c, doc, query, timeout=2.0):
    """Seconds from now until ``doc`` shows up in BOTH bm25 and dense results (polling)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if all(doc in ids(c, query, s) for s in ("bm25", "dense")):
            return time.perf_counter() - t0
        time.sleep(0.01)
    return None


def test_shards_report_async_mode(cluster):
    shards = cluster.http.get(f"{cluster.router_url}/admin/status").json()["shards"]
    assert {s["apply_mode"] for s in shards.values()} == {"async"}


def test_empty_text_is_rejected_by_the_router_not_reported_as_unavailable(cluster):
    t0 = time.perf_counter()
    r = cluster.http.post(f"{cluster.router_url}/upsert", json={"doc_id": "empty-1", "version": 1, "text": "  "})
    assert r.status_code == 422  # a client error, answered at once...
    assert time.perf_counter() - t0 < 1.0  # ...not retried 3x and turned into "503 shards unavailable"


def test_acked_write_is_searchable_within_one_second(cluster):
    lags = []
    for i in range(5):
        doc, text = f"afresh-{i}", f"okapiprobe{i} annuity freshness"
        ack = post(cluster, "/upsert", doc_id=doc, version=1, text=text)
        assert ack["applied"] and ack["visible"] is False  # acked = durable, not yet searchable
        lag = visible_after(cluster, doc, text)
        assert lag is not None, f"{doc} not visible within 2 s"
        lags.append(lag)
    assert max(lags) < 1.0, lags


def test_versions_checked_against_accepted_writes(cluster):
    assert post(cluster, "/upsert", doc_id="aidem", version=5, text="tapirprobe bond ladder")["applied"]
    # Immediately (v5 may still be queued): a retry and an older version must both be no-ops.
    assert not post(cluster, "/upsert", doc_id="aidem", version=5, text="tapirprobe bond ladder")["applied"]
    old = post(cluster, "/upsert", doc_id="aidem", version=4, text="tapirprobe STALE")
    assert not old["applied"] and old["current_version"] == 5
    assert post(cluster, "/admin/flush")["drained"]
    top = post(cluster, "/search", query="tapirprobe", strategy="bm25", include_text=True)["results"][0]
    assert top["doc_id"] == "aidem" and top["text"] == "tapirprobe bond ladder"


def test_acked_writes_survive_a_hard_kill(cluster):
    docs = [d for d in (f"acrash-{i}" for i in range(200)) if shard_of(d, 3) == 1][:20]
    for d in docs:  # a burst: some of these are still queued (acked, not applied) at kill time
        post(cluster, "/upsert", doc_id=d, version=1, text=f"{d.replace('-', '')}probe pangolin treasury")
    cluster.kill("shard1")  # hard kill: the in-memory queue is lost, the WAL is not
    cluster.start_shard(1)  # restart -> base corpus + WAL replay (re-embeds text-only records)
    cluster.wait_healthy([cluster.shard_url(1)], 120)
    for d in docs:
        assert d in ids(cluster, f"{d.replace('-', '')}probe"), d


def test_migration_under_async_writes(cluster):
    """migrate -> (writes keep flowing) -> ready -> cutover: docs written during the build are found by
    the NEW model's dense index (dual-write via the async applier); BM25 answers never change;
    rollback restores the old answers exactly."""
    import threading

    q = "short selling borrowing shares"
    before_bm25 = ids(cluster, q, "bm25")
    before_hybrid = ids(cluster, q, "hybrid")
    stop, written = threading.Event(), []

    def writer():
        with httpx.Client(timeout=10) as http:
            i = 0
            while not stop.is_set():
                d = f"amig-{i}"
                http.post(f"{cluster.router_url}/upsert",
                          json={"doc_id": d, "version": 1, "text": f"amigprobe{i} capybara pension"}).raise_for_status()
                written.append(d)
                i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        assert post(cluster, "/admin/migrate", model="bge-small")
        assert ids(cluster, q, "hybrid") == before_hybrid  # still served by minilm during the build
        deadline = time.time() + 60
        while time.time() < deadline:
            shards = cluster.http.get(f"{cluster.router_url}/admin/status").json()["shards"].values()
            if all(s["models"].get("bge-small", {}).get("ready") for s in shards):
                break
            time.sleep(0.1)
        time.sleep(0.3)
    finally:
        stop.set()
        t.join()
    assert post(cluster, "/admin/flush")["drained"]
    assert post(cluster, "/admin/cutover", model="bge-small")
    assert ids(cluster, q, "bm25") == before_bm25  # BM25 does not depend on the embedding model
    for i in range(0, len(written), max(1, len(written) // 10)):  # sample of docs written during the build
        assert written[i] in ids(cluster, f"amigprobe{i} capybara pension", "dense"), written[i]
    post(cluster, "/admin/rollback")
    assert ids(cluster, q, "hybrid") == before_hybrid  # rollback = the old answers, exactly
