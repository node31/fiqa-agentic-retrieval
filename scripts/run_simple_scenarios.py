"""Run every scenario in scenarios/SimpleScenarios/ on four clusters and record what happened.

    python scripts/run_simple_scenarios.py                                    # all 8 scenarios x 4 clusters
    python scripts/run_simple_scenarios.py --configs SimpleSyncTinyCluster    # one cluster
    python scripts/run_simple_scenarios.py --scenarios 3 8                    # scenarios whose name starts with 3 or 8

THE FOUR CLUSTERS
    Each run starts its OWN fresh 3-shard cluster (``scenario.py --start-cluster``: a router on :19000 and
    shards on :19101+, plain local processes, no Docker), plays the CSV, then stops the cluster. A fresh
    cluster per run means no scenario sees another one's writes or a model left over from a migration.

    SimpleSyncTinyCluster    24-doc tiny corpus, hash "embedders"; sync writes      (~12 s per run)
    SimpleAsyncTinyCluster   the same, async writes (ack after the WAL, searchable within 1 s)
    SimpleSyncFullCluster    the real FiQA corpus, 57,638 docs, MiniLM / bge-small; sync writes  (~1 min per run)
    SimpleAsyncFullCluster   the same, async writes

WHAT IT WRITES  (scenarios/SimpleScenarios/results/)
    <cluster>/<scenario>/console.txt    the console output, paths made repo-relative     (checked in)
    <cluster>/<scenario>/events.jsonl   every recorded event                              (checked in)
    <cluster>/<scenario>/replay.html    the animated replay; open it in a browser         (checked in)
    <cluster>/<scenario>/cluster_logs/  router and shard logs (absolute local paths)      (gitignored)
    <cluster>/<scenario>/run.json       exit code and wall time, for SUMMARY.md            (gitignored)
    SUMMARY.md                          one row per scenario, one column per cluster      (checked in)

    A scenario's folder is emptied before it runs, so a folder always holds exactly one run.
    The cluster logs stay out of git: they hold this machine's absolute paths and say nothing the
    console and events do not.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = ROOT / "scenarios" / "SimpleScenarios"
RESULTS = SIMPLE / "results"

CONFIGS = {
    "SimpleSyncTinyCluster": ["--start-cluster", "--tiny", "--mode", "sync"],
    "SimpleAsyncTinyCluster": ["--start-cluster", "--tiny", "--mode", "async"],
    "SimpleSyncFullCluster": ["--start-cluster", "--real", "--mode", "sync"],
    "SimpleAsyncFullCluster": ["--start-cluster", "--real", "--mode", "async"],
}


def scenario_files() -> list[Path]:
    """The numbered CSVs (1_SimpleRead.csv, ...); the template is for copying, not part of the set."""
    return sorted(p for p in SIMPLE.glob("*.csv") if p.name[0].isdigit())


def run_one(csv: Path, config: str) -> dict:
    """Run one scenario on one cluster; returns a small digest of its events for SUMMARY.md."""
    out = RESULTS / config / csv.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(ROOT / "scripts" / "scenario.py"), str(csv.relative_to(ROOT)), *CONFIGS[config],
           "--out", str(out.relative_to(ROOT)), "--no-color"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    took = time.perf_counter() - t0
    # Repo-relative paths, so the checked-in console.txt does not leak my folder layout.
    text = (p.stdout + p.stderr).replace(str(ROOT) + os.sep, "").replace(str(ROOT), ".")
    cmdline = "python " + " ".join(c if " " not in c else f'"{c}"' for c in
                                   ["scripts/scenario.py", csv.relative_to(ROOT).as_posix(), *CONFIGS[config],
                                    "--out", out.relative_to(ROOT).as_posix(), "--no-color"])
    (out / "console.txt").write_text(f"$ {cmdline}\n\n{text}", encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"exit_code": p.returncode, "took_s": round(took, 1)}), encoding="utf-8")
    return digest(out / "events.jsonl", p.returncode, took)


def digest(events_path: Path, code: int, took: float) -> dict:
    """PASS/FAIL, soft checks, write outcomes and which model answered the searches."""
    if not events_path.exists():
        return {"result": f"no events (exit {code})", "took": took}
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summ = next((e for e in reversed(events) if e.get("type") == "summary"), {})
    ends = [e for e in events if e.get("type") == "op_end"]
    starts = {e["id"]: e for e in events if e.get("type") == "op_start"}   # line + actor live on op_start
    soft = [c for e in ends for c in e.get("checks") or [] if not c.get("hard")]
    misses = [(starts.get(e["id"], {}).get("line"), c["kind"], c["doc_id"], starts.get(e["id"], {}).get("actor"))
              for e in ends for c in e.get("checks") or [] if not c.get("hard") and not c.get("pass")]
    writes = [e for e in ends if e.get("op") in ("upsert", "delete")]
    noops = sum(1 for e in writes if e.get("ok") and not (e.get("resp") or {}).get("applied"))
    models = [(e.get("resp") or {}).get("model") for e in ends if e.get("op") == "search"]
    return {"result": "PASS" if summ.get("passed") and code == 0 else f"FAIL (exit {code})",
            "ops": summ.get("ops"), "errors": summ.get("errors"),
            "soft": f"{sum(1 for c in soft if c.get('pass'))}/{len(soft)}", "misses": misses,
            "writes": len(writes), "noops": noops,
            "models": " → ".join(f"{m}×{n}" for m, n in _runs(models)),
            "lags": [v["lag_ms"] for v in summ.get("visibility_lags", []) if v.get("lag_ms") is not None],
            "took": took}


def _runs(seq: list) -> list[tuple]:
    """['a', 'a', 'b', 'a'] -> [('a', 2), ('b', 1), ('a', 1)]: which model answered, in order."""
    out: list[list] = []
    for x in seq:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return [tuple(r) for r in out]


def write_summary(table: dict[str, dict[str, dict]]) -> Path:
    """SUMMARY.md: a grid (scenario x cluster), then the soft misses one by one."""
    configs = [c for c in CONFIGS if any(c in row for row in table.values())]
    lines = ["# Simple scenarios: recorded runs", "",
             "Written by `python scripts/run_simple_scenarios.py` on "
             f"{time.strftime('%Y-%m-%d')}. Each cell: result · soft checks passed/total · which model answered "
             "the searches, in order. A soft check is a search that expected a doc (found) or expected it gone "
             "(`!doc`); a miss is reported but never fails the run. The console output of every run is in "
             "`<cluster>/<scenario>/console.txt`, and its animated replay in `replay.html` (download it and "
             "open it in a browser).", "",
             "| scenario | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for stem, row in table.items():
        cells = []
        for c in configs:
            d = row.get(c)
            if not d:
                cells.append("")
                continue
            extra = f" · {d['noops']} no-op writes" if d.get("noops") else ""
            cells.append(f"{d['result']} · soft {d.get('soft', '?')} · {d.get('models') or 'no searches'}{extra}")
        lines.append(f"| {stem} | " + " | ".join(cells) + " |")
    lines += ["", "## Soft misses", "",
              "Searches whose expectation did not hold. In async mode a search sent a few ms after a write can "
              "miss it (the write is acked once it is in the WAL and applied within 1 s); in sync mode there "
              "should be none.", ""]
    any_miss = False
    for stem, row in table.items():
        for c in configs:
            for line, kind, doc, actor in (row.get(c) or {}).get("misses", []):
                any_miss = True
                want = "expected" if kind == "expect" else "expected gone"
                lines.append(f"- {c} / {stem}, line {line}: {actor}'s search {want} `{doc}`, "
                             f"{'missed it' if kind == 'expect' else 'still found it'}")
    if not any_miss:
        lines.append("None.")
    times = Counter()
    for row in table.values():
        for c, d in row.items():
            times[c] += d.get("took", 0)
    lines += ["", "## Wall time per cluster (including cluster start)", ""]
    lines += [f"- {c}: {times[c]:.0f} s" for c in configs]
    path = RESULTS / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--scenarios", nargs="+", help="name prefixes, e.g. 3 8 or 4_Read (default: all)")
    a = ap.parse_args(argv)
    files = [f for f in scenario_files() if not a.scenarios or any(f.stem.startswith(s) for s in a.scenarios)]
    # Keep the cells of clusters/scenarios not rerun this time, so a partial rerun does not blank SUMMARY.md.
    table: dict[str, dict[str, dict]] = {}
    for f in scenario_files():
        for c in CONFIGS:
            ev, run = RESULTS / c / f.stem / "events.jsonl", RESULTS / c / f.stem / "run.json"
            if ev.exists() and run.exists():
                r = json.loads(run.read_text(encoding="utf-8"))
                table.setdefault(f.stem, {})[c] = digest(ev, r["exit_code"], r["took_s"])
    failed = 0
    for c in a.configs:
        for f in files:
            print(f"{c:24s} {f.stem:50s} ...", end=" ", flush=True)
            d = run_one(f, c)
            table.setdefault(f.stem, {})[c] = d
            failed += not d["result"].startswith("PASS")
            print(f"{d['result']}  soft {d.get('soft', '?')}  {d.get('models', '')}  ({d['took']:.0f} s)", flush=True)
    table = {f.stem: table[f.stem] for f in scenario_files() if f.stem in table}
    print(f"\nsummary: {write_summary(table).relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
