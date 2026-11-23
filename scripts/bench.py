"""Run the full benchmark matrix and write results/bench.json.

    python scripts/bench.py                 # everything (about 20-30 min on 32 cores)
    python scripts/bench.py --only local    # single-node configs only
    python scripts/bench.py --only finetune # the fine-tuned model's rows only

WHAT RUNS
    local (in-process single node, each config in a FRESH process, so cold = cold):
        bm25, dense-minilm, dense-bge-small, hybrid-rrf-minilm (alpha .5),
        hybrid-linear-minilm (alpha .7), hybrid-linear-bge-small (alpha .7)
    service (real router + shard processes over HTTP, cluster restarted per config):
        3 shards: bm25 / dense / hybrid-linear  -> the 1%-of-single-node correctness check
        1 shard : hybrid-linear                 -> 1 vs N shard latency
        3 shards, BM25 LOCAL stats              -> what the DFS phase buys us
    finetune (--only finetune, after `make finetune`): dense / hybrid single node and hybrid through 3 shards
        for minilm-ft, at --ft-alpha (0.7, re-swept with scripts/ablations.py --model minilm-ft)
Each run's full output (incl. per-query rankings) lands in results/runs/<name>.json.
bench.json keeps the summary plus machine info (latency is hardware-dependent).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from src.service.cluster import Cluster  # noqa: E402

LOCAL = [
    ("bm25", ["--strategy", "bm25"]),
    ("dense-minilm", ["--strategy", "dense", "--model", "minilm"]),
    ("dense-bge-small", ["--strategy", "dense", "--model", "bge-small"]),
    ("hybrid-rrf-minilm", ["--strategy", "hybrid", "--model", "minilm", "--fusion", "rrf", "--alpha", "0.5"]),
    ("hybrid-linear-minilm", ["--strategy", "hybrid", "--model", "minilm", "--fusion", "linear", "--alpha", "0.7"]),
    ("hybrid-linear-bge-small", ["--strategy", "hybrid", "--model", "bge-small", "--fusion", "linear", "--alpha", "0.7"]),
]
# (name, n_shards, extra router env, eval args)
SERVICE = [
    ("svc3-bm25", 3, {}, ["--strategy", "bm25"]),
    ("svc3-dense-minilm", 3, {}, ["--strategy", "dense"]),
    ("svc3-hybrid-linear-minilm", 3, {}, ["--strategy", "hybrid", "--fusion", "linear", "--alpha", "0.7"]),
    ("svc1-hybrid-linear-minilm", 1, {}, ["--strategy", "hybrid", "--fusion", "linear", "--alpha", "0.7"]),
    ("svc3-bm25-localidf", 3, {"BM25_STATS": "local"}, ["--strategy", "bm25"]),
]


def finetune_configs(alpha: float) -> tuple[list, list]:
    """The fine-tuned model's rows (scripts/finetune.py), at the alpha re-tuned for it on dev.

    Same protocol as the rows above, so they are directly comparable with dense-minilm,
    hybrid-linear-minilm and svc3-hybrid-linear-minilm.
    """
    a = str(alpha)
    local = [
        ("dense-minilm-ft", ["--strategy", "dense", "--model", "minilm-ft"]),
        ("hybrid-linear-minilm-ft", ["--strategy", "hybrid", "--model", "minilm-ft", "--fusion", "linear", "--alpha", a]),
    ]
    service = [("svc3-hybrid-linear-minilm-ft", 3, {}, ["--strategy", "hybrid", "--fusion", "linear", "--alpha", a])]
    return local, service


def machine() -> dict:
    import psutil

    cpu = platform.processor()
    try:
        if os.name == "nt":
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        elif sys.platform == "darwin":  # WHY: macOS has no /proc; sysctl gives e.g. "Apple M2 Pro"
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return {"cpu": cpu, "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "os": platform.platform(),
            "python": platform.python_version()}


def run_eval(name: str, args: list[str]) -> dict:
    out = RESULTS_DIR / "runs" / f"{name}.json"
    cmd = [sys.executable, "-m", "src.eval", "--name", name, "--out", str(out), *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"{name} failed:\n{r.stderr[-3000:]}")
    res = json.loads(out.read_text())
    res.pop("run", None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["local", "service", "finetune"],
                    help="finetune = only the minilm-ft rows (needs `make finetune` first)")
    ap.add_argument("--ft-alpha", type=float, default=0.7, help="alpha for the minilm-ft hybrid rows")
    a = ap.parse_args()
    path = RESULTS_DIR / "bench.json"
    bench = json.loads(path.read_text()) if path.exists() else {}
    bench.update(machine=machine(), protocol={
        "split": "dev (500 queries)", "cold": "first 20 dev queries in a fresh process",
        "warmup": "100 train queries (seed 42)", "warm": "all 500 dev queries after warmup",
        "latency_measured_by": "client, end to end (service: includes HTTP to router)",
        "torch_threads_per_process": 4})
    bench.setdefault("configs", {})
    local, service = LOCAL, SERVICE
    if a.only == "finetune":
        local, service = finetune_configs(a.ft_alpha)
    if a.only in (None, "local", "finetune"):
        for name, args in local:
            bench["configs"][name] = {"mode": "local", **run_eval(name, ["--mode", "local", *args])}
            path.write_text(json.dumps(bench, indent=1))
    if a.only in (None, "service", "finetune"):
        for name, n, env, args in service:
            models = "minilm-ft" if name.endswith("minilm-ft") else "minilm"
            # WHY own ports: 8000 is the Docker router's port; if the stack is up, a cluster on 8000 can't
            # bind it and the eval would silently measure the Docker stack instead.
            with Cluster(n, state_root=ARTIFACTS_DIR / f"bench_{name}", env=env, fresh=True, timeout_s=2.0,
                         models=models, router_port=9000, base_port=9101) as c:
                time.sleep(1)
                res = run_eval(name, ["--mode", "service", "--url", c.router_url, *args])
            bench["configs"][name] = {"mode": "service", "n_shards": n, **res}
            path.write_text(json.dumps(bench, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
