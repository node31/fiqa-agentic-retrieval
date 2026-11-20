"""Sync vs async apply under heavy write load: throughput, ack latency, freshness, search latency.

QUESTION (ADR 0013, ch07 §7.4)
    The sync write path (ADR 0007) acks only after a write is applied: freshness = 0, but every write
    pays for its own embedding before the ack. The async path acks after the WAL and applies in
    batches in the background, promising "searchable within 1 s". Under LOTS of writes, which one
    gives more throughput, which one acks faster, does async really keep the 1 s promise, and what
    happens to search latency while all this is going on?

WHAT IT DOES (per mode)
    Starts a real 3-shard cluster on the real FiQA corpus (MiniLM), then for ``--duration`` seconds runs:
      * W writer threads: upsert real FiQA passages as fast as they can
        (80% new docs, 20% updates of a doc the same writer wrote earlier, version + 1);
      * R reader threads: hybrid searches with dev queries, back to back;
      * 1 freshness probe: every 0.25 s upsert a doc with a unique token, then search for that token
        in a tight loop and record the visible lag = time from the ack until the START of the first
        search that finds it (~0 = found by the first search after the ack, i.e. sync's guarantee).
    Then flushes, reads each shard's /admin/status, and stops the cluster.

    python -m scripts.write_load                         # both modes, 30 s each (~4 min total)
    python -m scripts.write_load --mode async --duration 60 --writers 16

OUTPUT
    results/write_load.json  ({"sync": {...}, "async": {...}}) and a side-by-side table on stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import httpx

from src.beir_data import load_corpus, load_split
from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.service.cluster import Cluster
from src.util import percentile


def pct(values: list[float]) -> dict:
    """p50 / p95 / p99 / max of a list of milliseconds (None if empty)."""
    if not values:
        return None
    return {"p50": round(percentile(values, 50), 1), "p95": round(percentile(values, 95), 1),
            "p99": round(percentile(values, 99), 1), "max": round(max(values), 1), "n": len(values)}


def run_mode(mode: str, duration: float, writers: int, readers: int, texts: list[str], queries: list[str],
             router_port: int, base_port: int) -> dict:
    state = ARTIFACTS_DIR / f"writeload_{mode}"
    print(f"\n=== {mode}: starting 3-shard cluster (real corpus, MiniLM) ...", flush=True)
    with Cluster(3, base_port=base_port, router_port=router_port, state_root=state, fresh=True,
                 env={"APPLY_MODE": mode}, timeout_s=2.0) as c:
        url = c.router_url
        stop = threading.Event()
        lock = threading.Lock()
        # Shared result lists (appended under `lock` by the worker threads).
        r = {"ack_ms": [], "write_errors": 0, "writes_ok": 0, "noop": 0, "search_ms": [], "search_errors": 0,
             "partial": 0, "probe_lag_ms": [], "probe_first_hit": 0, "probe_missed": 0, "probe_ack_ms": [],
             "error_samples": []}

        def writer(w: int) -> None:
            rng = random.Random(1000 + w)
            mine: dict[str, int] = {}  # doc_id -> last version this writer wrote
            n = 0
            with httpx.Client(timeout=30) as http:
                while not stop.is_set():
                    if mine and rng.random() < 0.2:  # update one of my earlier docs
                        doc = rng.choice(list(mine))
                        mine[doc] += 1
                    else:
                        doc, n = f"wl-{w}-{n}", n + 1
                        mine[doc] = 1
                    t0 = time.perf_counter()
                    try:
                        resp = http.post(f"{url}/upsert", json={"doc_id": doc, "version": mine[doc],
                                                                "text": rng.choice(texts)})
                        resp.raise_for_status()
                        dt = (time.perf_counter() - t0) * 1e3
                        with lock:
                            r["ack_ms"].append(dt)
                            r["writes_ok" if resp.json()["applied"] else "noop"] += 1
                    except httpx.HTTPError as e:
                        with lock:
                            r["write_errors"] += 1
                            if len(r["error_samples"]) < 5:  # keep a few, so errors can be explained
                                body = getattr(getattr(e, "response", None), "text", "")
                                r["error_samples"].append(f"{doc} v{mine[doc]}: {type(e).__name__}: {e} {body[:200]}")

        def reader(i: int) -> None:
            rng = random.Random(2000 + i)
            with httpx.Client(timeout=30) as http:
                while not stop.is_set():
                    t0 = time.perf_counter()
                    try:
                        resp = http.post(f"{url}/search", json={"query": rng.choice(queries), "strategy": "hybrid"})
                        resp.raise_for_status()
                        with lock:
                            r["search_ms"].append((time.perf_counter() - t0) * 1e3)
                            r["partial"] += bool(resp.json().get("partial"))
                    except httpx.HTTPError:
                        with lock:
                            r["search_errors"] += 1

        def probe() -> None:
            rng = random.Random(3000)
            n = 0
            with httpx.Client(timeout=30) as http:
                while not stop.is_set():
                    token = f"wlprobe{n}x"
                    doc = f"probe-{n}"
                    n += 1
                    t0 = time.perf_counter()
                    try:
                        http.post(f"{url}/upsert", json={"doc_id": doc, "version": 1,
                                                         "text": f"{token} {rng.choice(texts)}"}).raise_for_status()
                    except httpx.HTTPError:
                        continue
                    t_ack = time.perf_counter()
                    first, found, t_hit = True, False, 0.0
                    # Tight polling loop: no sleep before the first search (that is sync's promise).
                    # The lag is measured to the START of the search that finds the doc, so
                    # "found by the very first search after the ack" reads as ~0 ms.
                    while time.perf_counter() - t_ack < 3.0:
                        t_hit = time.perf_counter()
                        hits = http.post(f"{url}/search", json={"query": token, "strategy": "bm25", "k": 10}).json()
                        if any(h["doc_id"] == doc for h in hits["results"]):
                            found = True
                            break
                        first = False
                        time.sleep(0.005)
                    with lock:
                        r["probe_ack_ms"].append((t_ack - t0) * 1e3)
                        if found:
                            r["probe_lag_ms"].append((t_hit - t_ack) * 1e3)
                            r["probe_first_hit"] += first
                        else:
                            r["probe_missed"] += 1
                    stop.wait(0.25)

        threads = ([threading.Thread(target=writer, args=(w,)) for w in range(writers)]
                   + [threading.Thread(target=reader, args=(i,)) for i in range(readers)]
                   + [threading.Thread(target=probe)])
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(duration)
        stop.set()
        for t in threads:
            t.join(60)
        elapsed = time.perf_counter() - t_start
        t0 = time.perf_counter()
        drained = httpx.post(f"{url}/admin/flush", params={"timeout": 30}, timeout=40).json()["drained"]
        flush_s = time.perf_counter() - t0
        status = httpx.get(f"{url}/admin/status", timeout=10).json()

    n_probes = len(r["probe_lag_ms"]) + r["probe_missed"]
    return {
        "mode": mode, "duration_s": round(elapsed, 1), "writers": writers, "readers": readers,
        "writes_ok": r["writes_ok"], "writes_per_s": round(r["writes_ok"] / elapsed, 1),
        "write_errors": r["write_errors"], "write_error_samples": r["error_samples"], "noop": r["noop"],
        "write_ack_ms": pct(r["ack_ms"]),
        "freshness": {"probes": n_probes, "visible_lag_ms": pct(r["probe_lag_ms"]),
                      "found_by_first_search": r["probe_first_hit"], "missed_within_3s": r["probe_missed"],
                      "probe_ack_ms": pct(r["probe_ack_ms"])},
        "search_ms": pct(r["search_ms"]), "searches": len(r["search_ms"]), "search_errors": r["search_errors"],
        "partial_results": r["partial"],
        "final_flush": {"drained": drained, "seconds": round(flush_s, 2)},
        "shards": {i: {k: s.get(k) for k in ("n_docs", "apply_mode", "async")}
                   for i, s in status["shards"].items()},
    }


def table(res: dict) -> str:
    """Side-by-side text table of the headline numbers for every mode that ran."""
    modes = list(res)

    def g(m, path):
        v = res[m]
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        return "-" if v is None else v

    rows = [("writes acked / s", ("writes_per_s",)), ("write errors", ("write_errors",)),
            ("ack ms p50", ("write_ack_ms", "p50")), ("ack ms p95", ("write_ack_ms", "p95")),
            ("ack ms p99", ("write_ack_ms", "p99")),
            ("visible lag ms p50", ("freshness", "visible_lag_ms", "p50")),
            ("visible lag ms p95", ("freshness", "visible_lag_ms", "p95")),
            ("visible lag ms max", ("freshness", "visible_lag_ms", "max")),
            ("probes found by 1st search", ("freshness", "found_by_first_search")),
            ("probes (total)", ("freshness", "probes")),
            ("probes missed (>3 s)", ("freshness", "missed_within_3s")),
            ("search ms p50 (under load)", ("search_ms", "p50")), ("search ms p95", ("search_ms", "p95")),
            ("searches", ("searches",)), ("search errors", ("search_errors",))]
    w = max(len(n) for n, _ in rows)
    out = [f"{'':{w}}  " + "  ".join(f"{m:>10}" for m in modes)]
    out += [f"{n:{w}}  " + "  ".join(f"{str(g(m, p)):>10}" for m in modes) for n, p in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["sync", "async", "both"], default="both")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of load per mode")
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--readers", type=int, default=2)
    ap.add_argument("--router-port", type=int, default=18700)
    ap.add_argument("--base-port", type=int, default=18711)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "write_load.json")
    a = ap.parse_args()
    corpus = load_corpus()
    rng = random.Random(42)
    texts = [t for t in rng.sample(list(corpus.values()), min(2000, len(corpus))) if t.strip()]  # real passages:
    # realistic embedding cost. Empty ones are dropped: FiQA has 38, and the API rejects empty text (422).
    queries = list(load_split("dev")[0].values())
    modes = ["sync", "async"] if a.mode == "both" else [a.mode]
    res = {m: run_mode(m, a.duration, a.writers, a.readers, texts, queries, a.router_port, a.base_port)
           for m in modes}
    a.out.write_text(json.dumps(res, indent=1))
    print("\n" + table(res))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
