"""Launch a local cluster (N shard processes + 1 router) without Docker.

WHAT
    ``Cluster`` starts real OS processes, the same programs docker-compose runs,
    so tests and benchmarks exercise real HTTP, real timeouts and real process
    boundaries. Used by tests/test_integration_*.py, scripts/bench.py and
    scripts/migration_demo.py.

    python -m src.service.cluster --shards 3      # run until Ctrl+C

PITFALLS
    * Port collisions: tests pass ``base_port`` so parallel runs don't clash.
    * Thread oversubscription: each shard gets TORCH_THREADS / OMP_NUM_THREADS so
      N processes do not each grab every core.
    * Child processes must be killed on exit, otherwise they linger and hold ports.

HOW TO READ THIS FILE
    ``Cluster.start`` is the story: (optionally wipe state) -> spawn N shard
    processes -> wait until each answers GET /health -> spawn the router with the
    shard URLs -> wait for the router's /health. ``stop`` kills everything.
    Typical use in a test or benchmark::

        with Cluster(n_shards=3, base_port=9101, router_port=9000) as c:
            httpx.post(f"{c.router_url}/search", json={"query": "etf fees"})
            c.kill("shard1")   # fault injection: next searches come back partial

    Each process writes stdout+stderr to ``<state_root>/<name>.log``
    (default state_root: ARTIFACTS_DIR/cluster<N>/); read those when startup fails.
    Shard WALs live in ``<state_root>/shard<i>/``, the router's model state in
    ``<state_root>/router_state.json``, so a restart (without ``fresh``) keeps writes
    and the cut-over model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from src.config import ARTIFACTS_DIR, PRIMARY_MODEL, ROOT


class Cluster:
    """N shard processes + 1 router process on 127.0.0.1, managed as one unit.

    Args:
        n_shards: number of shard processes; shard i listens on ``base_port + i``.
        router_port: the router's port (the only URL clients normally talk to).
        state_root: directory for logs, WALs and router state (see module docstring).
        models: comma list of embedding models every shard indexes; the FIRST one
            becomes the router's initial ACTIVE_MODEL.
        env: extra env vars for every child (e.g. {"EMBED_BACKEND": "hash"} in tests).
        threads: CPU threads per process (TORCH_THREADS / OMP_NUM_THREADS).
        timeout_s: the router's per-shard timeout (SHARD_TIMEOUT_S), 0.5 s by default.
        fresh: delete ``state_root`` first, i.e. forget all WALs and router state.

    Usable as a context manager: ``with Cluster(...) as c:`` starts it and stops it on exit.
    """

    def __init__(self, n_shards: int = 3, base_port: int = 8101, router_port: int = 8000,
                 state_root: Path | None = None, models: str = PRIMARY_MODEL, env: dict | None = None,
                 threads: int = 4, timeout_s: float = 0.5, fresh: bool = False):
        self.n, self.base_port, self.router_port = n_shards, base_port, router_port
        self.state_root = Path(state_root or ARTIFACTS_DIR / f"cluster{n_shards}")
        self.models, self.extra_env, self.threads, self.timeout_s = models, env or {}, threads, timeout_s
        self.fresh = fresh
        # name ("shard0", ..., "router") -> its child process handle.
        self.procs: dict[str, subprocess.Popen] = {}

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.router_port}"

    def shard_url(self, i: int) -> str:
        return f"http://127.0.0.1:{self.base_port + i}"

    def _env(self, **kw) -> dict:
        """Environment for a child process: ours + cluster-wide settings + per-process ``kw``.

        Later entries win: os.environ < defaults below < ``env=`` from __init__ < ``kw``.
        """
        # PYTHONPATH=ROOT so ``python -m src.service...`` resolves the ``src`` package from any
        # cwd. TOKENIZERS_PARALLELISM=false silences HuggingFace tokenizers' own thread pool
        # (and its fork warning), part of avoiding thread oversubscription.
        env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(self.threads),
               "TORCH_THREADS": str(self.threads), "TOKENIZERS_PARALLELISM": "false", **self.extra_env}
        # Env values must be strings (ports and Paths are passed as int/Path by callers).
        env.update({k: str(v) for k, v in kw.items()})
        return env

    def _spawn(self, name: str, module: str, **env) -> None:
        """Start ``python -m <module>`` as a child process, logging to ``<state_root>/<name>.log``."""
        log = open(self.state_root / f"{name}.log", "w")
        # sys.executable = the current interpreter, so children use the same venv.
        # Popen returns immediately; the process boots in the background (see wait_healthy).
        self.procs[name] = subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=self._env(**env),
                                            stdout=log, stderr=subprocess.STDOUT)

    def start_shard(self, i: int) -> None:
        """(Re)start shard ``i``. Also used by tests to restart a killed shard (WAL replay)."""
        self._spawn(f"shard{i}", "src.service.shard_server", SHARD_ID=i, N_SHARDS=self.n,
                    PORT=self.base_port + i, STATE_DIR=self.state_root / f"shard{i}", SHARD_MODELS=self.models)

    def start(self, wait_s: float = 600) -> "Cluster":
        """Start all shards, wait for them, then start the router. Returns ``self``.

        ``wait_s`` is generous (10 min) because each shard indexes its slice and loads
        embedding models on startup, which can take minutes on a cold cache.
        """
        if self.fresh and self.state_root.exists():
            import shutil

            shutil.rmtree(self.state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)
        # Shards boot in PARALLEL (all spawned, then one combined wait), so startup takes
        # about as long as the slowest shard, not the sum.
        for i in range(self.n):
            self.start_shard(i)
        self.wait_healthy([self.shard_url(i) for i in range(self.n)], wait_s)
        # WHY the router starts last: its startup does not need the shards, but a router
        # that is up before its shards would answer every search with partial/503 results.
        # SHARD_URLS order defines shard ids on the router side, so it must be 0..N-1.
        self._spawn("router", "src.service.router", PORT=self.router_port,
                    SHARD_URLS=",".join(self.shard_url(i) for i in range(self.n)),
                    ROUTER_STATE=self.state_root / "router_state.json", SHARD_TIMEOUT_S=self.timeout_s,
                    ACTIVE_MODEL=self.models.split(",")[0])
        self.wait_healthy([self.router_url], wait_s)
        return self

    def wait_healthy(self, urls: list[str], wait_s: float) -> None:
        """Poll GET <url>/health until every url returns 200, or raise after ``wait_s`` seconds.

        Fails fast with a pointer to the log if any child process has already exited
        (e.g. a crash or "address already in use"), instead of waiting out the deadline.
        """
        deadline = time.time() + wait_s
        pending = set(urls)
        while pending and time.time() < deadline:
            for u in list(pending):
                try:
                    if httpx.get(f"{u}/health", timeout=1.0).status_code == 200:
                        pending.discard(u)
                # Connection refused while the server is still booting: expected, keep polling.
                except httpx.HTTPError:
                    pass
            # poll() is None while a process runs, else its exit code.
            for name, p in self.procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited early; see {self.state_root / (name + '.log')}")
            time.sleep(0.3)
        if pending:
            raise TimeoutError(f"not healthy after {wait_s}s: {pending}")

    def kill(self, name: str) -> None:
        """Hard-kill one process ("shard1", "router"). Tests use it to simulate a crash.

        ``kill()`` is an abrupt stop (no graceful shutdown), like a real crash; durability
        then relies on the shard's WAL. ``wait(10)`` reaps it so its port is freed.
        """
        p = self.procs.pop(name, None)
        if p and p.poll() is None:
            p.kill()
            p.wait(10)

    def stop(self) -> None:
        """Kill every child process this Cluster started."""
        for name in list(self.procs):
            self.kill(name)

    # Context-manager protocol: ``with Cluster(...) as c:`` calls start() on entry and
    # stop() on exit, even if the body raises, so no orphan processes keep holding ports.
    # PITFALL: if start() itself raises (e.g. a shard never becomes healthy), Python does
    # NOT call __exit__, so processes already spawned keep running.
    def __enter__(self) -> "Cluster":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    """CLI: start a cluster on the default ports (router 8000, shards 8101+) and block."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--models", default=PRIMARY_MODEL)
    ap.add_argument("--fresh", action="store_true", help="wipe WAL/state first")
    a = ap.parse_args()
    with Cluster(a.shards, models=a.models, fresh=a.fresh) as c:
        print(f"cluster up: router {c.router_url}, shards {[c.shard_url(i) for i in range(c.n)]}. Ctrl+C to stop.")
        # Keep the parent alive (the children die with the ``with`` block). Catching Ctrl+C
        # lets the block exit normally, so __exit__ -> stop() kills every child process.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
