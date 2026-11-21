"""Zero-downtime embedding-model migration commands (Part B3). Used by the Makefile.

    python scripts/migration.py migrate  [--model bge-small] [--wait] [--reencode] [--throttle 0.2]
    python scripts/migration.py cutover  [--model bge-small]
    python scripts/migration.py rollback
    python scripts/migration.py finalize --model minilm      # stop dual-writing the old model
    python scripts/migration.py status

LIFECYCLE (docs/topics/10_model_migration.md)
    serving=minilm --migrate--> minilm serves, bge-small dual-written + backfilled
                   --cutover--> bge-small serves, minilm still dual-written (instant rollback)
                   --rollback-> minilm serves again (nothing rebuilt)
                   --finalize-> old index dropped (rollback no longer instant)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.config import ROUTER_URL, SECONDARY_MODEL  # noqa: E402


def call(c, method, path, **body):
    r = c.request(method, path, json=body or None)
    if r.status_code >= 400:
        raise SystemExit(f"{path} -> HTTP {r.status_code}: {r.text}")
    return r.json()


def progress(st, model):
    return {i: s["models"].get(model, {}).get("progress", 0.0) for i, s in st["shards"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["migrate", "cutover", "rollback", "finalize", "status"])
    ap.add_argument("--model", default=SECONDARY_MODEL)
    ap.add_argument("--url", default=ROUTER_URL)
    ap.add_argument("--wait", action="store_true", help="migrate: block until every shard is ready")
    ap.add_argument("--reencode", action="store_true", help="ignore the embedding cache (realistic cost)")
    ap.add_argument("--throttle", type=float, default=0.0, help="seconds to sleep between backfill batches")
    a = ap.parse_args()
    with httpx.Client(base_url=a.url, timeout=60) as c:
        if a.cmd == "status":
            st = call(c, "GET", "/admin/status")
            print(json.dumps({"active_model": st["active_model"], "previous_model": st["previous_model"],
                              "shards": {i: s["models"] for i, s in st["shards"].items()}}, indent=1))
        elif a.cmd == "migrate":
            print(json.dumps(call(c, "POST", "/admin/migrate", model=a.model, reencode=a.reencode,
                                  throttle_s=a.throttle)))
            while a.wait:
                st = call(c, "GET", "/admin/status")
                p = progress(st, a.model)
                print(f"  backfill progress {a.model}: {p}", flush=True)
                if all(s["models"].get(a.model, {}).get("ready") for s in st["shards"].values()):
                    print(f"{a.model} ready on all shards; run cutover")
                    break
                time.sleep(1)
        elif a.cmd == "cutover":
            print(json.dumps(call(c, "POST", "/admin/cutover", model=a.model)))
        elif a.cmd == "rollback":
            print(json.dumps(call(c, "POST", "/admin/rollback")))
        elif a.cmd == "finalize":
            print(json.dumps(call(c, "POST", "/admin/finalize", model=a.model)))


if __name__ == "__main__":
    main()
