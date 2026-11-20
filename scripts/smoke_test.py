"""Smoke test for a RUNNING system, over HTTP only: is everything up and behaving correctly?

WHY HTTP ONLY
    So it runs anywhere: on the host (``--url http://127.0.0.1:8000``) or inside the ``tools``
    container of docker-compose (``--url http://router:8000``), which is how scripts/bootstrap.sh
    and bootstrap.ps1 use it, with nothing installed on the host except Docker. Container
    kill/restart checks need the docker CLI, so they live in scripts/docker_check.py (host side).

CHECKS (each prints PASS/FAIL; exit code 1 if any fails)
    basic    3 shards answering, healthy, >= 57,638 docs; a hybrid search returns 10 results, not partial;
             freshness (sync: found by the FIRST search after the ack; async: within 1 s, both detected
             automatically from /admin/status); idempotency (retry + older version are no-ops)
    migrate  migrate -> writes during the build -> all shards ready -> cutover -> docs written during the
             build found by the new model (dual-write) -> BM25 unchanged -> rollback -> answers identical
             to before -> finalize the trial model (leaves the system exactly as it was)

    python -m scripts.smoke_test --url http://127.0.0.1:8000             # basic
    python -m scripts.smoke_test --url http://127.0.0.1:8000 --migrate   # basic + migration
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

RESULTS: list[tuple[str, bool, str]] = []
QUERY = "what is short selling"


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}", flush=True)
    return ok


class Api:
    """Thin client over the router's HTTP API (one keep-alive connection)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=60)

    def get(self, path: str) -> dict:
        r = self.http.get(self.url + path)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, **body) -> httpx.Response:
        return self.http.post(self.url + path, json=body)

    def search(self, query: str, strategy: str = "hybrid") -> dict:
        r = self.post("/search", query=query, strategy=strategy)
        r.raise_for_status()
        return r.json()

    def ids(self, query: str, strategy: str = "hybrid") -> list[str]:
        return [h["doc_id"] for h in self.search(query, strategy)["results"]]

    def upsert(self, doc_id: str, version: int, text: str) -> dict:
        r = self.post("/upsert", doc_id=doc_id, version=version, text=text)
        r.raise_for_status()
        return r.json()

    def visible_lag(self, doc: str, query: str, timeout: float = 3.0) -> float | None:
        """0.0 if the FIRST search after now finds `doc` in BM25 and dense; else seconds until it does."""
        t0 = time.perf_counter()
        first = True
        while time.perf_counter() - t0 < timeout:
            if all(doc in self.ids(query, s) for s in ("bm25", "dense")):
                return 0.0 if first else time.perf_counter() - t0
            first = False
            time.sleep(0.01)
        return None


def wait_ready(api: Api, timeout: float = 600) -> None:
    """Block until the router answers and every shard is healthy (e.g. right after `compose up`)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st = api.get("/admin/status")
            if st["shards"] and not st["missing_shards"] and all(s["healthy"] for s in st["shards"].values()):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{api.url} not ready after {timeout}s")


def basic(api: Api) -> str:
    """Health, search, freshness, idempotency. Returns the apply mode the shards run in."""
    st = api.get("/admin/status")
    shards = st["shards"]
    n_docs = sum(s["n_docs"] for s in shards.values())
    check("3 shards answering and healthy", len(shards) == 3 and not st["missing_shards"]
          and all(s["healthy"] for s in shards.values()), f"active model {st['active_model']}")
    # >= because the WAL keeps docs written by earlier runs (re-running the bootstrap is allowed).
    check("corpus loaded: >= 57,638 docs across the shards", n_docs >= 57638,
          f"{[s['n_docs'] for s in shards.values()]}")
    modes = {s.get("apply_mode", "sync") for s in shards.values()}
    mode = modes.pop() if len(modes) == 1 else "mixed"
    check("all shards run the same apply mode", mode != "mixed", mode)

    out = api.search(QUERY)
    check("hybrid search returns 10 results, not partial", len(out["results"]) == 10 and not out["partial"],
          f"{out['latency_ms']} ms")

    run = str(int(time.time()))  # unique ids per run, so re-runs never collide with old docs
    lags = []
    for i in range(5):
        doc, text = f"smoke-{run}-{i}", f"smokeprobe{run}x{i} okapi annuity"
        assert api.upsert(doc, 1, text)["applied"]
        lags.append(api.visible_lag(doc, text))
    shown = ", ".join("never" if x is None else f"{x:.3f}s" for x in lags)
    if mode == "async":
        check("freshness (async): every write searchable within 1 s",
              all(x is not None and x < 1.0 for x in lags), shown)
    else:
        check("freshness (sync): every write found by the FIRST search after its ack",
              all(x == 0.0 for x in lags), shown)

    doc = f"smoke-idem-{run}"
    a = api.upsert(doc, 3, "smokeidem quokka")
    b = api.upsert(doc, 3, "smokeidem quokka")
    c = api.upsert(doc, 2, "smokeidem STALE")
    check("idempotency: a retry and an older version are no-ops",
          a["applied"] and not b["applied"] and not c["applied"] and c.get("current_version") == 3)
    return mode


def migrate(api: Api, target: str = "bge-small") -> None:
    """Blue/green migration with writes during the build, then rollback + cleanup."""
    st = api.get("/admin/status")
    old = st["active_model"]
    if old == target:
        check(f"migration skipped: {target} is already active", True)
        return
    before = {s: api.ids(QUERY, s) for s in ("bm25", "hybrid")}
    r = api.post("/admin/migrate", model=target)
    check(f"migrate {old} -> {target} started on all shards", r.status_code == 200, r.text[:100])
    check("during the build, answers are unchanged (still the old model)", api.ids(QUERY, "hybrid") == before["hybrid"])
    run, written = str(int(time.time())), []
    for i in range(10):  # writes during the build: must be DUAL-written to both models
        d = f"smoke-mig-{run}-{i}"
        api.upsert(d, 1, f"smokemig{run}x{i} capybara pension")
        written.append((d, f"smokemig{run}x{i} capybara pension"))
    t0, deadline = time.time(), time.time() + 900
    while time.time() < deadline:
        shards = api.get("/admin/status")["shards"].values()
        if all(s["models"].get(target, {}).get("ready") for s in shards):
            break
        time.sleep(2)
    api.post("/admin/flush")  # async mode: make sure the writes above are applied
    r = api.post("/admin/cutover", model=target)
    check(f"backfill ready everywhere in {time.time() - t0:.0f} s; cutover accepted", r.status_code == 200, r.text[:100])
    found = sum(d in api.ids(text, "dense") for d, text in written)
    check("docs written during the build are found by the NEW model (dual-write)", found == len(written),
          f"{found}/{len(written)}")
    check("BM25 answers unchanged by the model switch", api.ids(QUERY, "bm25") == before["bm25"])
    r = api.post("/admin/rollback")
    check(f"rollback to {old}", r.status_code == 200 and api.get("/admin/status")["active_model"] == old)
    check("after rollback, hybrid answers identical to before", api.ids(QUERY, "hybrid") == before["hybrid"])
    r = api.post("/admin/finalize", model=target)  # drop the trial index: back to the original state
    check(f"cleanup: trial {target} index dropped", r.status_code == 200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--migrate", action="store_true", help="also run the blue/green migration check")
    ap.add_argument("--wait", type=float, default=600, help="seconds to wait for the system to be ready")
    a = ap.parse_args()
    api = Api(a.url)
    print(f"smoke test against {a.url}", flush=True)
    wait_ready(api, a.wait)
    basic(api)
    if a.migrate:
        migrate(api)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
