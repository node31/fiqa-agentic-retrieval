"""Query load generator: send queries continuously and report every failure.

    python scripts/query_load.py --duration 120 --threads 4            # run for 2 min
    python scripts/query_load.py --until-file stop.flag                  # run until the file appears

A query FAILS if: HTTP error / timeout / connection error, or the router returns
partial results (a shard missed it). Every response records which embedding model
served it, so the report shows the exact moment of cutover and rollback.
Target for B3: zero failed queries across migrate -> cutover -> rollback.
"""
import argparse
import itertools
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from src.beir_data import load_split  # noqa: E402
from src.config import ROUTER_URL  # noqa: E402
from src.util import percentile  # noqa: E402


def run_load(url: str, duration: float | None, threads: int, until_file: Path | None,
             strategy: str = "hybrid", stop: threading.Event | None = None) -> dict:
    queries = list(load_split("dev")[0].values())
    it = itertools.cycle(queries)
    it_lock, rec_lock = threading.Lock(), threading.Lock()
    records: list[dict] = []
    stop = stop or threading.Event()
    t_start = time.time()

    def worker():
        with httpx.Client(base_url=url, timeout=5) as c:
            while not stop.is_set():
                with it_lock:
                    q = next(it)
                t0 = time.perf_counter()
                rec = {"t": round(time.time() - t_start, 3)}
                try:
                    r = c.post("/search", json={"query": q, "strategy": strategy, "k": 10})
                    rec["status"] = r.status_code
                    body = r.json() if r.status_code == 200 else {}
                    rec["model"] = body.get("model")
                    rec["ok"] = r.status_code == 200 and not body.get("partial") and len(body.get("results", [])) > 0
                except Exception as e:  # noqa: BLE001 - every failure must be counted, not raised
                    rec.update(status=-1, ok=False, error=repr(e)[:200])
                rec["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
                with rec_lock:
                    records.append(rec)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    while not stop.is_set():
        time.sleep(0.2)
        if duration and time.time() - t_start >= duration:
            stop.set()
        if until_file and until_file.exists():
            stop.set()
    for t in ts:
        t.join(10)
    records.sort(key=lambda r: r["t"])
    switches, last = [], None
    for r in records:
        if r.get("model") and r["model"] != last:
            switches.append({"t": r["t"], "model": r["model"]})
            last = r["model"]
    ms = [r["ms"] for r in records if r["ok"]]
    return {"total": len(records), "failed": sum(not r["ok"] for r in records),
            "failures": [r for r in records if not r["ok"]][:50],
            "served_by_model": dict(Counter(r.get("model") for r in records)),
            "model_switches": switches, "p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95),
            "duration_s": round(time.time() - t_start, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=ROUTER_URL)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--until-file", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "query_load.json")
    a = ap.parse_args()
    if not a.duration and not a.until_file:
        a.duration = 60
    rep = run_load(a.url, a.duration, a.threads, a.until_file)
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"queries={rep['total']} FAILED={rep['failed']} served_by={rep['served_by_model']} "
          f"p50/p95={rep['p50_ms']}/{rep['p95_ms']}ms switches={rep['model_switches']}")
    sys.exit(1 if rep["failed"] else 0)


if __name__ == "__main__":
    main()
