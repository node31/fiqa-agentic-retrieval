"""End-to-end B3 demo: migrate -> cutover -> rollback WHILE queries and writes are running.

    python scripts/migration_demo.py [--reencode] [--throttle 0.2]

1. Start a fresh 3-shard cluster serving minilm.
2. Start the query load (4 threads) and a writer that upserts a new passage every 0.5 s.
3. migrate bge-small (dual-write + backfill), wait until ready.
4. cutover -> bge-small serves; keep load running.
5. rollback -> minilm serves again; keep load running.
6. Stop load. Check: zero failed queries, and every passage written DURING the
   migration is found by the dense index of BOTH models (proves dual-write).
Writes results/migration_demo.json.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scripts.query_load import run_load  # noqa: E402
from src.config import ARTIFACTS_DIR, PRIMARY_MODEL, SECONDARY_MODEL  # noqa: E402
from src.service.cluster import Cluster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reencode", action="store_true")
    ap.add_argument("--throttle", type=float, default=0.2)
    ap.add_argument("--phase-s", type=float, default=15, help="seconds of load between steps")
    a = ap.parse_args()
    events, written = [], []
    t0 = time.time()

    def log(msg):
        events.append({"t": round(time.time() - t0, 2), "event": msg})
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    with Cluster(3, state_root=ARTIFACTS_DIR / "migration_demo", models=PRIMARY_MODEL, fresh=True,
                 timeout_s=2.0) as c, httpx.Client(base_url=c.router_url, timeout=60) as http:
        log(f"cluster up, serving {PRIMARY_MODEL}")
        stop = threading.Event()
        load_result = {}
        lt = threading.Thread(target=lambda: load_result.update(run_load(c.router_url, None, 4, None, stop=stop)))
        lt.start()

        import random

        vocab = ("annuity ladder pension escrow dividend bond coupon yield rollover roth hedge margin "
                 "futures warrant spinoff buyback leverage amortize refinance escrow deductible ira "
                 "brokerage custodian fiduciary premium strike expiry collateral liquidity").split()
        rng = random.Random(0)

        def writer():
            i = 0
            while not stop.is_set():
                doc = f"mig-{i}"
                # WHY distinct random words: near-duplicate probe texts would compete with each
                # other in the dense top-k and make the dual-write check flaky.
                text = f"Probe {i}: " + " ".join(rng.sample(vocab, 8))
                r = http.post("/upsert", json={"doc_id": doc, "version": 1, "text": text})
                if r.status_code == 200:
                    written.append((doc, text, time.time() - t0))
                i += 1
                time.sleep(0.5)

        wt = threading.Thread(target=writer)
        wt.start()
        time.sleep(a.phase_s)
        log(f"migrate -> {SECONDARY_MODEL} (reencode={a.reencode}, throttle={a.throttle}s)")
        http.post("/admin/migrate", json={"model": SECONDARY_MODEL, "reencode": a.reencode,
                                          "throttle_s": a.throttle}).raise_for_status()
        while True:
            st = http.get("/admin/status").json()
            if all(s["models"].get(SECONDARY_MODEL, {}).get("ready") for s in st["shards"].values()):
                break
            time.sleep(0.5)
        log("backfill complete on all shards")
        time.sleep(a.phase_s)
        log("cutover"); http.post("/admin/cutover", json={"model": SECONDARY_MODEL}).raise_for_status()
        time.sleep(a.phase_s)
        log("rollback"); http.post("/admin/rollback").raise_for_status()
        time.sleep(a.phase_s)
        stop.set()
        lt.join()
        wt.join()
        log("load stopped")
        # dual-write check 1: per shard, both dense indexes hold exactly the same number of docs
        st = http.get("/admin/status").json()
        index_sizes = {i: {m: v["n"] for m, v in s["models"].items()} | {"docs": s["n_docs"]}
                       for i, s in st["shards"].items()}
        # dual-write check 2: every doc written during the run is retrievable by BOTH models
        missing = {}
        for model in (PRIMARY_MODEL, SECONDARY_MODEL):
            http.post("/admin/cutover", json={"model": model}).raise_for_status()
            miss = [d for d, text, _ in written
                    if d not in [r["doc_id"] for r in http.post("/search", json={
                        "query": text, "strategy": "dense", "k": 10}).json()["results"]]]
            missing[model] = miss
        http.post("/admin/cutover", json={"model": PRIMARY_MODEL})
    report = {"events": events, "load": load_result, "docs_written_during_run": len(written),
              "index_sizes_per_shard": index_sizes,
              "written_docs_missing_from_dense_top10": {m: len(v) for m, v in missing.items()},
              "missing_examples": {m: v[:5] for m, v in missing.items()},
              "zero_failed_queries": load_result.get("failed") == 0}
    (ROOT / "results" / "migration_demo.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "events"} | {"load": {
        k: load_result.get(k) for k in ("total", "failed", "served_by_model", "model_switches", "p50_ms", "p95_ms")}},
        indent=1))


if __name__ == "__main__":
    main()
