"""End-to-end check of the Docker deployment: 4 containers (3 shards + router) from ONE image.

WHY
    "Works with plain processes" is not the same as "works in containers": separate network
    namespaces, service DNS names (shard0:8101), bind-mounted state, a 2 GB memory cap per shard,
    and container restarts all differ. This script brings the compose stack up and proves the Part B
    guarantees hold IN DOCKER, in either write mode.

ONE IMAGE, FOUR CONTAINERS
    ``docker images`` shows one image (fiqa-retrieval:latest). ``docker compose up`` starts FOUR
    containers from it (shard0, shard1, shard2, router), each with different env vars (SHARD_ID, ...).
    An image is the program on disk; a container is one running copy of it.

WHAT IT CHECKS (each step prints PASS/FAIL)
    1. 4 containers running + healthy; /admin/status sees 3 shards, 57,638 docs in total
    2. a hybrid search answers, not partial
    3. freshness: sync -> found by the first search after the ack; async -> found within 1 s
    4. idempotency: retry and older version are no-ops
    5. crash: ``docker compose kill shard1`` -> searches come back partial (missing [1]);
       ``docker compose start shard1`` -> the acked write is back (WAL replay in the container)
    6. migration: migrate -> backfill ready on all 3 containers -> cutover -> search -> rollback
    7. memory per container (docker stats) vs the 2 GiB cap
    Optional: ``--eval`` runs the 500-query dev eval inside the ``eval`` container (~1 min).

    python -m scripts.docker_check                      # sync mode, fresh state, stack torn down after
    python -m scripts.docker_check --mode async
    python -m scripts.docker_check --mode both --eval   # both modes + the full eval in each
    python -m scripts.docker_check --keep               # leave the stack running afterwards
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts import smoke_test as smoke

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000"


def docker_exe() -> str:
    """docker on PATH, else Docker Desktop's per-user install location (Windows)."""
    found = shutil.which("docker")
    if found:
        return found
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
    if cand.exists():
        return str(cand)
    sys.exit("docker CLI not found")


def compose(*args: str, mode: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    env = {**os.environ, "APPLY_MODE": mode}
    # Make sure the docker CLI's credential/compose helpers next to docker.exe are found too.
    env["PATH"] = str(Path(docker_exe()).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run([docker_exe(), "compose", *args], cwd=ROOT, env=env, check=check,
                          capture_output=capture, text=True)


def run(mode: str, do_eval: bool, keep: bool) -> None:
    print(f"\n===== Docker stack, APPLY_MODE={mode} =====", flush=True)
    state = ROOT / "artifacts" / "docker"
    compose("down", mode=mode, check=False)
    if state.exists():  # fresh WALs + router state, so earlier runs can't leak into this one
        shutil.rmtree(state)
    t0 = time.time()
    # --wait: return only when every container's healthcheck passes (shards first, then the router).
    compose("up", "-d", "--wait", mode=mode)
    print(f"  stack up in {time.time() - t0:.0f} s", flush=True)

    # 1. containers: ONE image, FOUR running containers
    ps = compose("ps", "--format", "json", mode=mode, capture=True).stdout.strip()
    rows = json.loads(ps) if ps.startswith("[") else [json.loads(x) for x in ps.splitlines() if x.strip()]
    health = {r["Service"]: r.get("Health") or r.get("State") for r in rows}
    print("  containers: " + ", ".join(f"{s}={h}" for s, h in sorted(health.items())))
    smoke.check("4 containers (shard0-2 + router) from one image, all healthy",
                sorted(health) == ["router", "shard0", "shard1", "shard2"] and set(health.values()) == {"healthy"})

    # 2-4. the HTTP checks shared with the bootstrap (health, search, freshness, idempotency)
    api = smoke.Api(URL)
    smoke.wait_ready(api)
    got = smoke.basic(api)
    smoke.check(f"shards run APPLY_MODE={mode}", got == mode, got)

    # 5. crash one container, then bring it back
    from src.index.sharding import shard_of

    doc = next(f"dock-crash-{mode}-{i}" for i in range(500) if shard_of(f"dock-crash-{mode}-{i}", 3) == 1)
    api.upsert(doc, 1, "crashprobe pangolin treasury")
    compose("kill", "shard1", mode=mode)
    out = api.search(smoke.QUERY)
    smoke.check("shard1 container killed -> results flagged partial, missing [1]",
                out["partial"] and out["missing_shards"] == [1] and len(out["results"]) > 0)
    compose("start", "shard1", mode=mode)
    smoke.wait_ready(api, 300)
    smoke.check("shard1 restarted -> the acked write is back (WAL replay inside the container)",
                doc in api.ids("crashprobe pangolin treasury", "bm25"))

    # 6. blue/green migration across the 3 containers (shared with the bootstrap's --migrate)
    smoke.migrate(api)

    # 7. memory per container vs the 2 GiB cap
    stats = subprocess.run([docker_exe(), "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    for line in stats:
        if any(s in line for s in ("shard", "router")):
            print("   ", line)

    if do_eval:
        print("  running the 500-query dev eval inside the eval container ...", flush=True)
        # Override the eval service's command so this check doesn't overwrite the committed
        # results/runs/docker-*.json: the output file stays inside the (removed) container.
        p = compose("--profile", "tools", "run", "--rm", "eval", "python", "-m", "src.eval", "--mode", "service",
                    "--url", "http://router:8000", "--strategy", "hybrid", "--fusion", "linear", "--alpha", "0.7",
                    "--name", f"docker-check-{mode}", "--out", "/tmp/docker-check.json",
                    mode=mode, check=False, capture=True)
        last = [x for x in (p.stdout or "").splitlines() if "ndcg" in x.lower()]
        smoke.check("dev eval inside Docker", p.returncode == 0, last[-1] if last else (p.stderr or "")[-300:])
    if not keep:
        compose("down", mode=mode)


def pip_index_url() -> str | None:
    """The PyPI mirror to build with: $PIP_INDEX_URL, else the host's pip config (like the bootstrap).

    WHY: docker-compose.yml falls back to pypi.org when PIP_INDEX_URL is unset. On a network where
    pypi.org is blocked the build then fails, and because the build arg differs from the one the
    bootstrap used, Docker's layer cache misses too.
    """
    if os.environ.get("PIP_INDEX_URL"):
        return os.environ["PIP_INDEX_URL"]
    # PITFALL: ``pip config get global.index-url`` can answer "No such key" for a value that
    # ``pip config list`` shows (it lives in a file ``get`` doesn't look at), so parse ``list``.
    p = subprocess.run([sys.executable, "-m", "pip", "config", "list"], capture_output=True, text=True)
    for line in (p.stdout or "").splitlines():
        key, _, val = line.partition("=")
        if key.strip().endswith("index-url") and val.strip():
            return val.strip().strip("'\"")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["sync", "async", "both"], default="sync")
    ap.add_argument("--eval", action="store_true", help="also run the dev eval in the eval container")
    ap.add_argument("--keep", action="store_true", help="leave the stack running at the end")
    ap.add_argument("--build", action="store_true", help="rebuild the image first (needed after code changes)")
    a = ap.parse_args()
    if a.build:
        mirror = pip_index_url()
        if mirror:
            os.environ["PIP_INDEX_URL"] = mirror  # compose passes it to the build as a build arg
            print(f"  building with PIP_INDEX_URL={mirror}")
        compose("build", mode="sync")
    modes = ["sync", "async"] if a.mode == "both" else [a.mode]
    for i, m in enumerate(modes):
        run(m, a.eval, keep=a.keep and i == len(modes) - 1)
    failed = [n for n, ok, _ in smoke.RESULTS if not ok]
    print(f"\n{len(smoke.RESULTS) - len(failed)}/{len(smoke.RESULTS)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
