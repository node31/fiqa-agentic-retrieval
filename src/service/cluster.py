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
    def __init__(self, n_shards: int = 3, base_port: int = 8101, router_port: int = 8000,
                 state_root: Path | None = None, models: str = PRIMARY_MODEL, env: dict | None = None,
                 threads: int = 4, timeout_s: float = 0.5, fresh: bool = False):
        self.n, self.base_port, self.router_port = n_shards, base_port, router_port
        self.state_root = Path(state_root or ARTIFACTS_DIR / f"cluster{n_shards}")
        self.models, self.extra_env, self.threads, self.timeout_s = models, env or {}, threads, timeout_s
        self.fresh = fresh
        self.procs: dict[str, subprocess.Popen] = {}

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.router_port}"

    def shard_url(self, i: int) -> str:
        return f"http://127.0.0.1:{self.base_port + i}"

    def _env(self, **kw) -> dict:
        env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(self.threads),
               "TORCH_THREADS": str(self.threads), "TOKENIZERS_PARALLELISM": "false", **self.extra_env}
        env.update({k: str(v) for k, v in kw.items()})
        return env

    def _spawn(self, name: str, module: str, **env) -> None:
        log = open(self.state_root / f"{name}.log", "w")
        self.procs[name] = subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=self._env(**env),
                                            stdout=log, stderr=subprocess.STDOUT)

    def start_shard(self, i: int) -> None:
        self._spawn(f"shard{i}", "src.service.shard_server", SHARD_ID=i, N_SHARDS=self.n,
                    PORT=self.base_port + i, STATE_DIR=self.state_root / f"shard{i}", SHARD_MODELS=self.models)

    def start(self, wait_s: float = 600) -> "Cluster":
        if self.fresh and self.state_root.exists():
            import shutil

            shutil.rmtree(self.state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)
        for i in range(self.n):
            self.start_shard(i)
        self.wait_healthy([self.shard_url(i) for i in range(self.n)], wait_s)
        self._spawn("router", "src.service.router", PORT=self.router_port,
                    SHARD_URLS=",".join(self.shard_url(i) for i in range(self.n)),
                    ROUTER_STATE=self.state_root / "router_state.json", SHARD_TIMEOUT_S=self.timeout_s,
                    ACTIVE_MODEL=self.models.split(",")[0])
        self.wait_healthy([self.router_url], wait_s)
        return self

    def wait_healthy(self, urls: list[str], wait_s: float) -> None:
        deadline = time.time() + wait_s
        pending = set(urls)
        while pending and time.time() < deadline:
            for u in list(pending):
                try:
                    if httpx.get(f"{u}/health", timeout=1.0).status_code == 200:
                        pending.discard(u)
                except httpx.HTTPError:
                    pass
            for name, p in self.procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited early; see {self.state_root / (name + '.log')}")
            time.sleep(0.3)
        if pending:
            raise TimeoutError(f"not healthy after {wait_s}s: {pending}")

    def kill(self, name: str) -> None:
        p = self.procs.pop(name, None)
        if p and p.poll() is None:
            p.kill()
            p.wait(10)

    def stop(self) -> None:
        for name in list(self.procs):
            self.kill(name)

    def __enter__(self) -> "Cluster":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--models", default=PRIMARY_MODEL)
    ap.add_argument("--fresh", action="store_true", help="wipe WAL/state first")
    a = ap.parse_args()
    with Cluster(a.shards, models=a.models, fresh=a.fresh) as c:
        print(f"cluster up: router {c.router_url}, shards {[c.shard_url(i) for i in range(c.n)]}. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
