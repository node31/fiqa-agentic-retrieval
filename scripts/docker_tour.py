"""Docker tour: the whole Docker part of the manual tests (guide section 3) in one command, checked and reported.

WHY
    docs/guides/manual_tests_{windows,mac}.md section 3 is a page of commands to paste, each with the result I got.
    This script runs the same steps against the running Docker stack, checks every result against those numbers,
    leaves an animated ``replay.html`` for every step, and puts the stack back exactly as it found it.

HOW IT IS RUN
    Not directly: ``scripts/docker_tour.ps1`` (Windows) or ``scripts/docker_tour.sh`` (macOS/Linux) call it inside the
    ``tools`` container, between the steps that need the Docker CLI (stopping and starting shard1). The write mode is
    never changed: the tour needs the stack in sync mode (guide 3i, the async mode, stays a manual step).
    Each call is one subcommand:

        preflight   the stack is up; save snapshot_before.json (containers, models, docs per shard, probe answers)
        run STEP..  run steps or phases (search, eval, writes, migration, simple, agents, shard_*)
        finish      delete the docs the tour wrote, restore the models, verify against the snapshot, write the report

    Everything lands in ``toy_runs/docker_tour/<stamp>/`` (gitignored): ``index.html`` (the report), one folder per
    step with ``replay.html``, ``events.jsonl`` and ``console.txt``, and ``results.json`` (one record per step).

THE SHAPE CHECK (finish -> verify)
    Before anything changes, preflight records the stack's shape. At the end the same snapshot is taken again and
    compared field by field: containers and health, apply mode, active/previous model, models per shard, live docs
    per shard, and the exact top-10 answers to six probe queries. Identical answers are a strong check: BM25's
    global statistics count live docs only, so once the tour's docs are deleted every score is what it was.
    Deletes do leave tombstones in each shard's WAL; that is how deletes work, and search never sees them.
"""
from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index.sharding import shard_of  # noqa: E402

URL = os.environ.get("ROUTER_URL", "http://localhost:8000")
# The model the stack serves when the tour starts (saved in snapshot_before.json). The tour's expectations
# are written for "minilm" and follow whatever the stack starts on, e.g. "minilm-ft" after `make finetune`
# and an online migration. Dev numbers per model, for the eval step (DESIGN.md §11):
EVAL_REFERENCE = {"minilm": (0.4190, 0.4914), "minilm-ft": (0.4469, 0.5141), "bge-small": (0.4155, 0.4856)}
MIGRATION_TARGET = "bge-small"   # the model the migration steps build, cut over to, roll back from and drop
TOUR_DIR = ROOT / "scenarios" / "DockerTour"
SIMPLE_DIR = ROOT / "scenarios" / "SimpleScenarios"
PROBES = [(q, s) for q in ("what is a roth ira?", "should I pay off my mortgage early?")
          for s in ("bm25", "dense", "hybrid")]

SECTIONS = [
    ("1a", "Preflight: the stack is up, and its shape is recorded"),
    ("3a", "Search with each strategy"),
    ("3b", "The dev set through the router"),
    ("3c", "Writes: upsert, search, delete"),
    ("3d", "Model migration: migrate, cutover, rollback, finalize"),
    ("3e", "Agents through the router"),
    ("3f", "The eight simple scenarios"),
    ("3g", "Lose a shard"),
    ("end", "Back to the original shape"),
]


# ----------------------------------------------------------------------------- small helpers (pure, unit tested)
def compress(seq: list[str]) -> str:
    """['minilm', 'minilm', 'bge-small'] -> 'minilm×2 → bge-small×1' (the notation of SimpleScenarios/SUMMARY.md)."""
    groups = [(k, len(list(g))) for k, g in itertools.groupby(seq)]
    return " → ".join(f"{k}×{n}" for k, n in groups) or "-"


PIN = re.compile(r"\{on(\d+):([A-Za-z0-9_-]+)\}")


def pinned_id(name: str, shard: int, tag: str, n_shards: int) -> str:
    """The first ``name-tag-i`` that crc32-hashes to ``shard``: a doc id that really lives on that shard."""
    for i in itertools.count():
        doc = f"{name}-{tag}-{i}"
        if shard_of(doc, n_shards) == shard:
            return doc
    raise AssertionError("unreachable")


def render_pins(text: str, tag: str, n_shards: int) -> str:
    """Replace every ``{onN:name}`` with a doc id owned by shard N (the same id everywhere in one tour)."""
    return PIN.sub(lambda m: pinned_id(m.group(2), int(m.group(1)), tag, n_shards), text)


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_of(events: list[dict]) -> dict:
    return next((e for e in reversed(events) if e.get("type") == "summary"), {})


def meta_of(events: list[dict]) -> dict:
    return next((e for e in events if e.get("type") == "meta"), {})


def ops_of(events: list[dict]) -> list[tuple[dict, dict]]:
    """[(op_start, op_end), ...] in the order the ops ended."""
    starts = {e["id"]: e for e in events if e.get("type") == "op_start"}
    return [(starts.get(e["id"], {}), e) for e in events if e.get("type") == "op_end"]


def search_models(events: list[dict], actor: str | None = None) -> list[str]:
    """Which model answered each successful search, in order (optionally only one actor's searches)."""
    out = []
    for start, end in ops_of(events):
        if end.get("op") != "search" or not end.get("ok"):
            continue
        who = (start.get("actor") or start.get("lane") or "").lower()
        if actor and who != actor.lower():
            continue
        out.append((end.get("resp") or {}).get("model", "?"))
    return out


def writes_of(events: list[dict]) -> dict[str, dict]:
    """doc_id -> {"version": highest version seen, "live": True/False/None} from a run's upserts and deletes.

    ``live`` follows the last APPLIED write (upsert = live, delete = gone); None if no write to it was applied.
    The version also counts the server's ``current_version`` from stale replies, so a delete sent with
    version + 1 is always newer than anything the shard has for that doc.
    """
    docs: dict[str, dict] = {}
    for start, end in ops_of(events):
        if end.get("op") not in ("upsert", "delete"):
            continue
        req, resp = start.get("req") or {}, end.get("resp") or {}
        doc = req.get("doc_id")
        if not doc:
            continue
        d = docs.setdefault(doc, {"version": 0, "live": None})
        d["version"] = max(d["version"], int(req.get("version") or 0), int(resp.get("current_version") or 0))
        if end.get("ok") and resp.get("applied"):
            d["live"] = end["op"] == "upsert"
    return docs


def shape_of(status: dict) -> dict:
    """The parts of /admin/status that define the stack's shape (no counters, timings or memory)."""
    shards = {}
    for sid, sh in sorted((status.get("shards") or {}).items()):
        models = sh.get("models") or {}
        shards[str(sid)] = {
            "healthy": bool(sh.get("healthy")), "n_docs": sh.get("n_docs"), "apply_mode": sh.get("apply_mode"),
            "models": sorted(m for m, v in models.items() if (v or {}).get("ready")),
            "building": sorted(m for m, v in models.items() if not (v or {}).get("ready")),
        }
    return {"active_model": status.get("active_model"), "previous_model": status.get("previous_model"),
            "missing_shards": sorted(status.get("missing_shards") or []), "shards": shards}


def parse_compose_ps(text: str) -> dict[str, dict]:
    """``docker compose ps --format json`` (one JSON array, or one object per line) -> {service: {...}}."""
    text = (text or "").strip()
    if not text:
        return {}
    rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    return {r.get("Service"): {"state": r.get("State"), "health": r.get("Health") or "", "image": r.get("Image")}
            for r in rows if r.get("Service")}


def compare(before: dict, after: dict) -> list[dict]:
    """Field-by-field comparison of two snapshots -> [{"name", "before", "after", "ok"}]."""
    checks = []

    def add(name, b, a, ok=None):
        checks.append({"name": name, "before": b, "after": a, "ok": (b == a) if ok is None else ok})

    cb, ca = before.get("compose") or {}, after.get("compose") or {}
    if cb or ca:
        add("containers", ", ".join(sorted(cb)), ", ".join(sorted(ca)))
        for svc in sorted(set(cb) | set(ca)):
            b, a = cb.get(svc, {}), ca.get(svc, {})
            add(f"{svc}: state, health, image",
                f"{b.get('state')}, {b.get('health') or '-'}, {b.get('image')}",
                f"{a.get('state')}, {a.get('health') or '-'}, {a.get('image')}",
                ok=b == a and a.get("state") == "running" and a.get("health") in ("healthy", ""))
    sb, sa = before["shape"], after["shape"]
    add("router: active model", sb["active_model"], sa["active_model"])
    add("router: previous model (kept for rollback)", sb["previous_model"], sa["previous_model"])
    add("missing shards", sb["missing_shards"], sa["missing_shards"])
    for sid in sorted(set(sb["shards"]) | set(sa["shards"])):
        b, a = sb["shards"].get(sid, {}), sa["shards"].get(sid, {})
        add(f"shard {sid}: healthy", b.get("healthy"), a.get("healthy"), ok=a.get("healthy") is True)
        add(f"shard {sid}: apply mode", b.get("apply_mode"), a.get("apply_mode"))
        add(f"shard {sid}: models", b.get("models"), a.get("models"))
        add(f"shard {sid}: models being built", b.get("building"), a.get("building"))
        add(f"shard {sid}: live docs", b.get("n_docs"), a.get("n_docs"))
    pb, pa = before.get("probes") or {}, after.get("probes") or {}
    for key in sorted(set(pb) | set(pa)):
        b, a = pb.get(key, []), pa.get(key, [])
        same = b == a
        add(f"answers: {key}", f"top-10 starts {', '.join(b[:3])}", "identical" if same else
            f"differs: {', '.join(a[:3])}", ok=same)
    return checks


# ----------------------------------------------------------------------------- HTTP
def http():
    import httpx
    return httpx.Client(base_url=URL, timeout=60)


def status(client) -> dict:
    r = client.get("/admin/status")
    r.raise_for_status()
    return r.json()


def probes(client) -> dict[str, list[str]]:
    out = {}
    for q, s in PROBES:
        r = client.post("/search", json={"query": q, "strategy": s, "k": 10})
        r.raise_for_status()
        body = r.json()
        ids = [h["doc_id"] for h in body.get("results", [])]
        out[f"{s} · {q}"] = ids if not body.get("partial") else ["(partial)"] + ids
    return out


def snapshot(client, d: Path) -> dict:
    compose_file = d / "compose_now.json"
    return {"t": time.time(), "url": URL, "shape": shape_of(status(client)), "probes": probes(client),
            "compose": parse_compose_ps(compose_file.read_text(encoding="utf-8-sig")) if compose_file.exists() else {}}


# ----------------------------------------------------------------------------- the step catalogue
def _step(key, section, title, what, expected, **kw):
    return {"key": key, "section": section, "title": title, "what": what, "expected": expected, **kw}


SIMPLE_EXPECTED = {  # model sequences from scenarios/SimpleScenarios/results/SUMMARY.md (every cluster)
    "1_SimpleRead": "minilm×2", "2_SimpleReadAfterWrite": "minilm×3", "3_SimpleReadAfterWriteAndDelete": "minilm×6",
    "4_ReadMigrateRead": "minilm×2 → bge-small×2", "5_ReadWriteMigrateRead": "minilm×3 → bge-small×1",
    "6_ReadWriteMigrateWriteDuringMigrateRead": "minilm×3 → bge-small×2",
    "7_ReadWriteMigrateWriteDuringMigrateRollbackRead": "minilm×3 → bge-small×2 → minilm×2",
    "8_Idempotency": "minilm×7",
}
MIGRATES = {"4_ReadMigrateRead", "5_ReadWriteMigrateRead", "6_ReadWriteMigrateWriteDuringMigrateRead",
            "7_ReadWriteMigrateWriteDuringMigrateRollbackRead"}

STEPS: list[dict] = [
    _step("search", "3a", "One question, three strategies (scenario)",
          "bm25, dense and hybrid for 'what is a roth ira?' through the router and all 3 shard containers",
          "every search complete (partial false); hybrid finds 553031, 488556 and 214174",
          kind="scenario", file=TOUR_DIR / "1_search_strategies.jsonl"),
    _step("cli", "3a", "The same through the tools: ask.py and src/search.py",
          "ask.py 'what is a roth ira?' --strategy hybrid and search.py 'what is short selling?' against the router",
          "both exit 0; ask.py lists 553031", kind="cli"),
    _step("eval", "3b", "Dev set (500 questions) through the router",
          "src.eval --mode service, hybrid linear α 0.7, on the model the stack serves",
          "the single-process numbers for that model, e.g. nDCG@10 0.4190 and R@10 0.4914 for minilm, 0.4469 and "
          "0.5141 for minilm-ft (pass within 1%: your own docs in the stack can shift them slightly)", kind="eval"),
    _step("writes", "3c", "Upsert, search, delete, stale write",
          "owl writes a doc, fox searches, owl deletes it, then a late version 1 arrives",
          "found at #1 right after the upsert (sync); gone after the delete; the late v1 is a no-op "
          "(stale_or_duplicate)", kind="scenario", file=TOUR_DIR / "2_writes.jsonl", mode="sync"),
    _step("migration", "3d", "Migrate, cut over, roll back, finalize under live queries",
          "cutover refused first; bge-small built next to the starting model, cut over, rolled back, dropped; "
          "probe-bot searches throughout", "cutover refused (409); elephant's searches answered by the starting model → "
          "bge-small → the starting model; probe-bot: 0 failed queries; ends on the starting model", kind="scenario", file=TOUR_DIR / "3_migration.jsonl",
          migrates=True),
    _step("agents", "3e", "Single agent and multi-agent through the router (Azure)",
          "ask.py --strategy single on a compound question; agent_eval.py single,multi on 10 compound questions",
          "both finish; the agents' searches go to the cluster", kind="agents"),
    *[_step(f"simple{name[0]}", "3f", name, f"scenarios/SimpleScenarios/{name}.csv against the Docker stack",
            f"PASS; searches answered by {SIMPLE_EXPECTED[name]} (minilm = the starting model)", kind="scenario",
            file=SIMPLE_DIR / f"{name}.csv", models=SIMPLE_EXPECTED[name], migrates=name in MIGRATES)
      for name in SIMPLE_EXPECTED],
    _step("shard_before", "3g", "Before the outage: a doc on shard 1 and one on shard 0",
          "fox writes two docs whose ids hash to shard 1 and shard 0", "both acked and searchable",
          kind="scenario", file=TOUR_DIR / "4_shard1_before.jsonl", mode="sync"),
    _step("shard_down", "3g", "Shard 1 stopped: partial answers, failing writes",
          "searches and writes while the shard1 container is stopped",
          "searches answer with partial true and missing_shards [1]; the write owned by shard 1 fails (503); the "
          "write to shard 0 works", kind="scenario", file=TOUR_DIR / "5_shard1_down.jsonl"),
    _step("shard_back", "3g", "Shard 1 back: WAL replay",
          "searches after `docker compose start shard1`, then fox retries the failed write",
          "complete answers again; the doc acked before the outage is back; the retried write is applied",
          kind="scenario", file=TOUR_DIR / "6_shard1_back.jsonl", mode="sync"),
]
BY_KEY = {s["key"]: s for s in STEPS}
PHASES = {"search": ["search", "cli"], "eval": ["eval"], "writes": ["writes"], "migration": ["migration"],
          "agents": ["agents"], "simple": [s["key"] for s in STEPS if s["section"] == "3f"],
          "shard": ["shard_before", "shard_down", "shard_back"]}


# ----------------------------------------------------------------------------- results.json
def record(d: Path, rec: dict) -> dict:
    path = d / "results.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    data = [r for r in data if r.get("key") != rec["key"]] + [rec]
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}[rec["status"]]
    print(f"  {mark:4}  {rec['section']:>3}  {rec['title']}  ({rec.get('seconds', 0):.1f} s)", flush=True)
    if rec.get("got"):
        print(f"              {rec['got']}", flush=True)
    for name, ok, text in rec.get("details", []):
        if ok is False:
            print(f"              ✗ {name}: {text}", flush=True)
    return rec


def base_record(step: dict, t0: float) -> dict:
    return {k: step[k] for k in ("key", "section", "title", "what", "expected")} | {"t": t0, "links": {},
                                                                                       "details": []}


# ----------------------------------------------------------------------------- running one step
def _env() -> dict:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_cmd(cmd: list[str], log: Path, timeout: int = 3600) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, env=_env())
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(out, encoding="utf-8")
    return p.returncode, out


def for_model(expected: str, primary: str) -> str:
    """SimpleScenarios' expected sequences are written for minilm; rename it to the model the stack started on.

    >>> for_model("minilm×3 → bge-small×2 → minilm×2", "minilm-ft")
    'minilm-ft×3 → bge-small×2 → minilm-ft×2'
    """
    return re.sub(r"\bminilm(?=×)", primary, expected)


def extra_checks(step: dict, events: list[dict], primary: str = "minilm") -> list[tuple[str, bool | None, str]]:
    """Checks the scenario format can't express, read from the recorded events. ok=None is information only."""
    out = []
    ops = ops_of(events)
    run_id = meta_of(events).get("run_id", "")
    searches = [(s, e) for s, e in ops if e.get("op") == "search"]
    if step["key"] == "search":
        hyb = [e for s, e in searches if (s.get("req") or {}).get("strategy") == "hybrid"]
        if hyb:
            top = [h["doc_id"] for h in (hyb[0].get("resp") or {}).get("results", [])[:3]]
            out.append(("hybrid top 3", None, ", ".join(top)))
    if step["key"] == "writes" and searches:
        top = ((searches[0][1].get("resp") or {}).get("results") or [{}])[0].get("doc_id")
        out.append(("the new doc is #1 right after the upsert", top == f"tour-zebra-{run_id}", f"#1 is {top}"))
        last_upsert = [e for s, e in ops if e.get("op") == "upsert"][-1]
        reason = (last_upsert.get("resp") or {}).get("reason")
        out.append(("the late v1 is rejected as stale", reason == "stale_or_duplicate", f"reason {reason}"))
    if step["key"] == "migration":
        seq = search_models(events, "elephant")
        out.append(("elephant's searches answered by", seq == [primary, MIGRATION_TARGET, primary], " → ".join(seq)))
        bot = [e for s, e in ops if e.get("op") == "search" and (s.get("lane") or "").lower() == "probe-bot"]
        failed = sum(1 for e in bot if not e.get("ok"))
        out.append(("probe-bot queries during the migration", failed == 0 and bool(bot),
                    f"{len(bot)} queries, {failed} failed"))
    if "models" in step:
        seq = compress(search_models(events))
        out.append(("searches answered by", seq == for_model(step["models"], primary), seq))
    if step["key"] == "shard_down":
        lat = [e.get("latency_ms", 0) for s, e in searches]
        out.append(("partial search latency", None, ", ".join(f"{x:.0f} ms" for x in lat)))
    return out


def run_scenario_step(step: dict, d: Path, tag: str, n_shards: int, primary: str = "minilm") -> dict:
    t0 = time.time()
    rec = base_record(step, t0)
    out = d / step["key"]
    out.mkdir(parents=True, exist_ok=True)
    src = step["file"]
    text = src.read_text(encoding="utf-8")
    if PIN.search(text):                      # pin doc ids to shards; keep the file name (it names the replay)
        src = out / step["file"].name
        src.write_text(render_pins(text, tag, n_shards), encoding="utf-8")
    rc, _ = run_cmd([sys.executable, "scripts/scenario.py", str(src), "--url", URL, "--out", str(out),
                     "--no-color"], out / "console.txt")
    events = read_events(out / "events.jsonl")
    s = summary_of(events)
    details = extra_checks(step, events, primary)
    ok = rc == 0 and bool(s.get("passed")) and all(x[1] is not False for x in details)
    checks_total = s.get("checks_passed", 0) + s.get("checks_failed", 0)
    soft_total = s.get("soft_found", 0) + s.get("soft_missed", 0)
    got = f"{s.get('ops', 0)} ops, {s.get('errors', 0)} errors, checks {s.get('checks_passed', 0)}/{checks_total}"
    if soft_total:
        got += f", soft {s.get('soft_found', 0)}/{soft_total}"
    if not events:
        got = f"scenario did not run (exit {rc}); see console.txt"
    rel = lambda p: p.relative_to(d).as_posix()  # noqa: E731
    links = {"replay": rel(out / "replay.html"), "console": rel(out / "console.txt"),
             "events": rel(out / "events.jsonl")}
    rec |= {"status": "pass" if ok else "fail", "got": got, "seconds": time.time() - t0, "details": details,
            "links": {k: v for k, v in links.items() if (d / v).exists()}}
    return record(d, rec)


def run_cli_step(step: dict, d: Path) -> dict:
    t0 = time.time()
    out = d / "cli"
    rc1, o1 = run_cmd([sys.executable, "scripts/ask.py", "what is a roth ira?", "--strategy", "hybrid", "--url", URL],
                      out / "ask.txt")
    rc2, o2 = run_cmd([sys.executable, "src/search.py", "--url", URL, "--query", "what is short selling?",
                       "--top-k", "10", "--strategy", "hybrid"], out / "search.txt")
    details = [("ask.py exit code", rc1 == 0, str(rc1)), ("ask.py lists 553031", "553031" in o1, ""),
               ("search.py exit code", rc2 == 0, str(rc2))]
    ok = all(x[1] for x in details)
    return record(d, base_record(step, t0) | {
        "status": "pass" if ok else "fail", "seconds": time.time() - t0, "details": details,
        "got": "both answered" if ok else "see the logs", "links": {"ask.py": "cli/ask.txt", "search.py": "cli/search.txt"}})


def run_eval_step(step: dict, d: Path, primary: str = "minilm") -> dict:
    t0 = time.time()
    out = d / "eval"
    res_file = out / "dev_hybrid.json"
    rc, _ = run_cmd([sys.executable, "-m", "src.eval", "--mode", "service", "--url", URL, "--strategy", "hybrid",
                     "--fusion", "linear", "--alpha", "0.7", "--name", "docker-tour-hybrid", "--out", str(res_file)],
                    out / "console.txt")
    rec = base_record(step, t0) | {"links": {"console": "eval/console.txt"}}
    if rc != 0 or not res_file.exists():
        return record(d, rec | {"status": "fail", "seconds": time.time() - t0, "got": f"eval failed (exit {rc})"})
    res = json.loads(res_file.read_text(encoding="utf-8"))
    o, warm = res["overall"], res["latency"]["warm"]
    ndcg, r10 = o.get("ndcg@10"), o.get("recall@10")
    close = lambda got, want: got is not None and abs(got - want) / want <= 0.01  # noqa: E731
    if primary not in EVAL_REFERENCE:
        details = [(f"no reference numbers for {primary}", None, f"nDCG@10 {ndcg}, R@10 {r10}")]
    else:
        want_n, want_r = EVAL_REFERENCE[primary]
        details = [(f"nDCG@10 vs {want_n:.4f} ({primary})", close(ndcg, want_n), "exact" if ndcg == want_n else f"{ndcg}"),
                   (f"R@10 vs {want_r:.4f} ({primary})", close(r10, want_r), "exact" if r10 == want_r else f"{r10}")]
    return record(d, rec | {
        "status": "pass" if all(x[1] for x in details) else "fail", "seconds": time.time() - t0,
        "details": details, "links": rec["links"] | {"results": "eval/dev_hybrid.json"},
        "got": f"nDCG@10 {ndcg}, R@10 {r10}, MRR@10 {o.get('mrr@10')}, warm p50/p95 {warm['p50_ms']}/{warm['p95_ms']} ms"})


def run_agents_step(step: dict, d: Path, azure: bool) -> dict:
    t0 = time.time()
    rec = base_record(step, t0)
    if not azure:
        return record(d, rec | {"status": "skip", "seconds": 0,
                                "got": "skipped: costs money; run the tour with -Azure (Windows) or --azure"})
    env_text = (ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else ""
    if not re.search(r"^AZURE_OPENAI_API_KEY=\S+", env_text, re.M):
        return record(d, rec | {"status": "fail", "seconds": 0, "got": "AZURE_OPENAI_API_KEY is empty in .env"})
    out = d / "agents"
    rc1, _ = run_cmd([sys.executable, "scripts/ask.py", "Can stock brokerage firms fail, and can I print my own checks?",
                      "--strategy", "single", "--backend", "azure", "--confirm-spend", "--url", URL], out / "ask.txt")
    rc2, _ = run_cmd([sys.executable, "scripts/agent_eval.py", "--backend", "azure", "--confirm-spend", "--systems",
                      "single,multi", "--sets", "compound", "--limit", "10", "--search", "service", "--url", URL,
                      "--out", str(out / "agents.json")], out / "agent_eval.txt")
    details = [("ask.py --strategy single", rc1 == 0, f"exit {rc1}"), ("agent_eval.py", rc2 == 0, f"exit {rc2}")]
    return record(d, rec | {"status": "pass" if rc1 == rc2 == 0 else "fail", "seconds": time.time() - t0,
                            "details": details, "got": "see agents/agent_eval.txt",
                            "links": {"ask.py": "agents/ask.txt", "agent_eval": "agents/agent_eval.txt"}})


def run_step(key: str, d: Path, azure: bool) -> dict:
    step = BY_KEY[key]
    snap = json.loads((d / "snapshot_before.json").read_text(encoding="utf-8"))
    t0 = time.time()
    if step.get("migrates") and not snap.get("migration_ok"):
        return record(d, base_record(step, t0) | {"status": "skip", "seconds": 0, "got":
                      "skipped: before the tour the stack had more than one model built or active (or served "
                      f"{MIGRATION_TARGET}), and this step would change that"})
    if step.get("mode"):
        with http() as c:
            modes = {sh.get("apply_mode") for sh in status(c).get("shards", {}).values()}
        if modes != {step["mode"]}:
            return record(d, base_record(step, t0) | {"status": "fail", "seconds": 0,
                          "got": f"needs the shards in {step['mode']} mode, found {sorted(map(str, modes))}"})
    if step["kind"] == "scenario":
        rec = run_scenario_step(step, d, d.name, len(snap["shape"]["shards"]) or 3, primary_of(snap))
    elif step["kind"] == "cli":
        rec = run_cli_step(step, d)
    elif step["kind"] == "eval":
        rec = run_eval_step(step, d, primary_of(snap))
    else:
        rec = run_agents_step(step, d, azure)
    if step.get("migrates"):                  # every migration step ends back on the starting models
        restore_models(d, quiet=True)
    return rec


# ----------------------------------------------------------------------------- preflight / restore / verify
def primary_of(snap: dict) -> str:
    """The model the stack served when the tour started."""
    return snap.get("shape", {}).get("active_model") or "minilm"


def preflight(d: Path) -> int:
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {"key": "preflight", "section": "1a", "title": "The stack is up; its shape is saved",
           "what": "GET /admin/status, the containers, and six probe searches whose answers must be identical at the end",
           "expected": "3 healthy shards in sync mode, about 19.2K docs each, no missing shard", "t": t0, "links": {}}
    try:
        with http() as c:
            snap = snapshot(c, d)
    except Exception as e:  # noqa: BLE001
        print(f"\nThe router at {URL} does not answer ({e}).\nStart the stack first: .\\scripts\\bootstrap.ps1 "
              "(Windows) or ./scripts/bootstrap.sh (macOS/Linux).", flush=True)
        return 2
    shape = snap["shape"]
    shards = shape["shards"]
    modes = sorted({str(s["apply_mode"]) for s in shards.values()})
    snap["apply_mode"] = modes[0] if len(modes) == 1 else "mixed"
    primary = shape["active_model"]
    snap["migration_ok"] = (bool(primary) and primary != MIGRATION_TARGET and not shape["previous_model"]
                            and all(s["models"] == [primary] and not s["building"] for s in shards.values()))
    (d / "snapshot_before.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    (d / "original_mode.txt").write_text(snap["apply_mode"], encoding="utf-8")
    healthy = len(shards) == 3 and all(s["healthy"] for s in shards.values()) and not shape["missing_shards"]
    details = [(f"shard {i}", s["healthy"], f"{s['n_docs']:,} docs, {s['apply_mode']}, models {', '.join(s['models'])}")
               for i, s in shards.items()]
    details.append(("router", None, f"active model {shape['active_model']}, previous {shape['previous_model']}"))
    if not snap["migration_ok"]:
        details.append(("migration steps", None, "will be skipped: another model is built or active"))
    record(d, rec | {"status": "pass" if healthy and snap["apply_mode"] == "sync" else "fail",
                     "seconds": time.time() - t0, "details": details,
                     "got": f"{len(shards)} shards, {sum(s['n_docs'] or 0 for s in shards.values()):,} docs, "
                            f"{snap['apply_mode']} mode, queries on {shape['active_model']}",
                     "links": {"snapshot": "snapshot_before.json"}})
    if snap["apply_mode"] != "sync":
        print(f"The shards are in {snap['apply_mode']} write mode. The tour checks that a write is searchable right "
              "after its ack, which only sync mode promises, and it never changes the write mode itself.\nRestart "
              "the stack in sync mode first: .\\scripts\\bootstrap.ps1 (Windows) or ./scripts/bootstrap.sh.", flush=True)
        return 2
    return 0 if healthy else 2


def restore_models(d: Path, quiet: bool = False) -> list[str]:
    """Put the router and shards back on the one model they had before the tour (only when there was one)."""
    snap = json.loads((d / "snapshot_before.json").read_text(encoding="utf-8"))
    if not snap.get("migration_ok"):
        return []
    primary = primary_of(snap)
    actions = []
    with http() as c:
        st = shape_of(status(c))
        if st["active_model"] != primary:
            if st["previous_model"] == primary:
                c.post("/admin/rollback", json={}).raise_for_status()
                actions.append(f"rollback to {primary}")
            else:
                c.post("/admin/cutover", json={"model": primary}).raise_for_status()
                actions.append(f"cutover to {primary}")
        st = shape_of(status(c))
        extra = sorted({m for s in st["shards"].values() for m in s["models"] + s["building"]} - {primary})
        for m in extra:
            c.post("/admin/finalize", json={"model": m}).raise_for_status()
            actions.append(f"finalize {m}")
    if actions and not quiet:
        print(f"    models restored: {', '.join(actions)}", flush=True)
    return actions


def cleanup_docs(d: Path) -> tuple[int, int]:
    """Delete every doc the tour wrote that is still live (version = the highest seen + 1). -> (deleted, total)."""
    docs: dict[str, dict] = {}
    for ev in sorted(d.rglob("events.jsonl")):
        for doc, w in writes_of(read_events(ev)).items():
            cur = docs.setdefault(doc, {"version": 0, "live": None})
            cur["version"] = max(cur["version"], w["version"])
            if w["live"] is not None:
                cur["live"] = w["live"]
    live = {doc: w for doc, w in docs.items() if w["live"]}
    with http() as c:
        for doc, w in sorted(live.items()):
            r = c.post("/delete", json={"doc_id": doc, "version": w["version"] + 1})
            r.raise_for_status()
        if live:
            c.post("/admin/flush", json={}, timeout=120)
    return len(live), len(docs)


def finish(d: Path) -> int:
    t0 = time.time()
    deleted, total = cleanup_docs(d)
    actions = restore_models(d, quiet=True)
    record(d, {"key": "cleanup", "section": "end", "title": "Delete the tour's docs, restore the models",
               "what": "every doc the tour wrote is found in the recorded events and deleted with a newer version",
               "expected": "no tour doc left; only the starting model", "t": t0, "status": "pass", "seconds": time.time() - t0,
               "got": f"{total} docs written, {deleted} still live and now deleted; models: "
                      f"{', '.join(actions) or 'nothing to restore'}", "links": {}, "details": []})
    t1 = time.time()
    before = json.loads((d / "snapshot_before.json").read_text(encoding="utf-8"))
    with http() as c:
        after = snapshot(c, d)
    (d / "snapshot_after.json").write_text(json.dumps(after, indent=1), encoding="utf-8")
    checks = compare(before, after)
    (d / "shape_checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = [c for c in checks if not c["ok"]]
    record(d, {"key": "verify", "section": "end", "title": "Verify: the stack is back in its original shape",
               "what": "a fresh snapshot compared with the one taken before the tour, field by field",
               "expected": "identical containers, modes, models, live docs per shard and probe answers",
               "t": t1, "status": "fail" if bad else "pass", "seconds": time.time() - t1,
               "got": f"{len(checks) - len(bad)}/{len(checks)} fields identical",
               "details": [(c["name"], c["ok"], f"before {c['before']} / after {c['after']}") for c in bad],
               "links": {"before": "snapshot_before.json", "after": "snapshot_after.json"}})
    write_report(d)
    return 0 if all(r["status"] != "fail" for r in load_records(d)) else 1


# ----------------------------------------------------------------------------- the HTML report
def load_records(d: Path) -> list[dict]:
    recs = json.loads((d / "results.json").read_text(encoding="utf-8")) if (d / "results.json").exists() else []
    wrapper = d / "wrapper.jsonl"
    if wrapper.exists():
        for line in wrapper.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                recs.append({"what": "", "expected": "", "links": {}, "details": []} | r)
    return recs


CSS = """
body{font-family:Segoe UI,system-ui,-apple-system,sans-serif;margin:0;background:#f6f7fb;color:#1f2937}
main{max-width:1180px;margin:0 auto;padding:28px 24px 60px}
h1{font-family:Georgia,serif;color:#1e2761;margin:0 0 4px;font-size:30px}
h2{font-family:Georgia,serif;color:#1e2761;margin:34px 0 10px;font-size:21px}
.sub{color:#64748b;margin:0 0 18px}
.verdict{padding:16px 20px;border-radius:12px;font-size:19px;font-weight:600;margin:18px 0}
.verdict.ok{background:#dcf3e6;color:#1e7f4f}.verdict.bad{background:#fde2e1;color:#b42318}
.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{background:#fff;border-radius:12px;padding:14px 18px;min-width:150px;box-shadow:0 1px 3px #0001}
.card b{display:block;font-size:26px;color:#1e2761;font-family:Georgia,serif}.card span{color:#64748b;font-size:13px}
table{border-collapse:collapse;width:100%;table-layout:fixed;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0001}
th{background:#1e2761;color:#fff;text-align:left;font-weight:600;padding:9px 10px;font-size:13px}
td{padding:9px 10px;border-top:1px solid #e5e9f2;vertical-align:top;font-size:13.5px;overflow-wrap:anywhere}
td.what{color:#475569}.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:700;font-size:12px}
.pass{background:#dcf3e6;color:#1e7f4f}.fail{background:#fde2e1;color:#b42318}.skip{background:#eef2fd;color:#475569}
.det{margin:6px 0 0;padding:0;list-style:none;font-size:12.5px;color:#475569}.det li.bad{color:#b42318;font-weight:600}
a{color:#2b3a80}.links a{margin-right:8px;white-space:nowrap}.muted{color:#64748b;font-size:13px}
"""


def _cell_details(r: dict) -> str:
    items = []
    for name, ok, text in r.get("details") or []:
        mark = "✓" if ok else ("✗" if ok is False else "·")
        items.append(f'<li class="{"bad" if ok is False else ""}">{mark} {html.escape(str(name))}'
                     f'{": " + html.escape(str(text)) if text else ""}</li>')
    return f'<ul class="det">{"".join(items)}</ul>' if items else ""


def write_report(d: Path) -> Path:
    recs = sorted(load_records(d), key=lambda r: r.get("t", 0))
    n = {k: sum(1 for r in recs if r["status"] == k) for k in ("pass", "fail", "skip")}
    duration = (max(r.get("t", 0) + r.get("seconds", 0) for r in recs) - min(r.get("t", 0) for r in recs)) if recs else 0
    before = json.loads((d / "snapshot_before.json").read_text(encoding="utf-8")) if (d / "snapshot_before.json").exists() else {}
    shape_ok = next((r["status"] == "pass" for r in recs if r["key"] == "verify"), None)
    ok = n["fail"] == 0
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Docker tour {html.escape(d.name)}</title>"
             f"<style>{CSS}</style></head><body><main>",
             "<h1>Docker tour: the manual tests, run and checked</h1>",
             f"<p class='sub'>Run {html.escape(d.name)} against {html.escape(before.get('url', URL))} · every step of "
             "guide section 3, with what I expected and what this run got. Each scenario step links to its animated "
             "replay.</p>",
             f"<div class='verdict {'ok' if ok else 'bad'}'>{'All steps passed' if ok else 'Some steps failed'}"
             f"{'' if shape_ok is None else (' · the stack is back in its original shape' if shape_ok else ' · the stack is NOT back in its original shape')}</div>",
             "<div class='cards'>"
             f"<div class='card'><b>{n['pass']}</b><span>passed</span></div>"
             f"<div class='card'><b>{n['fail']}</b><span>failed</span></div>"
             f"<div class='card'><b>{n['skip']}</b><span>skipped</span></div>"
             f"<div class='card'><b>{duration / 60:.1f} min</b><span>wall time</span></div>"
             f"<div class='card'><b>{html.escape(str(before.get('apply_mode', '?')))}</b><span>write mode before and after</span></div>"
             "</div>"]
    for sec, name in SECTIONS:
        rows = [r for r in recs if r["section"] == sec]
        if not rows:
            continue
        parts.append(f"<h2>{html.escape(sec if sec != 'end' else '')} {html.escape(name)}</h2>")
        parts.append("<table><tr><th style='width:22%'>step</th><th style='width:27%'>expected</th>"
                     "<th>got</th><th style='width:70px'>result</th><th style='width:60px'>time</th>"
                     "<th style='width:120px'>open</th></tr>")
        for r in rows:
            links = " ".join(f"<a href='{html.escape(v)}'>{html.escape(k)}</a>" for k, v in (r.get("links") or {}).items())
            parts.append(
                f"<tr><td><b>{html.escape(r['title'])}</b><div class='muted'>{html.escape(r.get('what') or '')}</div></td>"
                f"<td class='what'>{html.escape(r.get('expected') or '')}</td>"
                f"<td>{html.escape(r.get('got') or '')}{_cell_details(r)}</td>"
                f"<td><span class='badge {r['status']}'>{r['status'].upper()}</span></td>"
                f"<td>{r.get('seconds', 0):.1f} s</td><td class='links'>{links}</td></tr>")
        parts.append("</table>")
        if sec == "end" and (d / "shape_checks.json").exists():
            checks = json.loads((d / "shape_checks.json").read_text(encoding="utf-8"))
            parts.append("<h2>Before and after, field by field</h2><table><tr><th>field</th><th>before the tour</th>"
                         "<th>after the tour</th><th style='width:70px'>same?</th></tr>")
            for c in checks:
                parts.append(f"<tr><td>{html.escape(c['name'])}</td><td>{html.escape(str(c['before']))}</td>"
                             f"<td>{html.escape(str(c['after']))}</td><td><span class='badge {'pass' if c['ok'] else 'fail'}'>"
                             f"{'yes' if c['ok'] else 'NO'}</span></td></tr>")
            parts.append("</table><p class='muted'>Deleted docs leave a tombstone entry in their shard's WAL: that is "
                         "how a delete stays newer than any late write. Search, the live-doc counts and the answers "
                         "above never see it.</p>")
    parts.append("</main></body></html>")
    out = d / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\n  {n['pass']} passed, {n['fail']} failed, {n['skip']} skipped. Report: {shown}", flush=True)
    return out


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preflight", "run", "finish", "report", "list"])
    ap.add_argument("steps", nargs="*", help="for run: step keys or phase names (" + ", ".join(PHASES) + ")")
    ap.add_argument("--dir", type=Path, help="the tour folder, e.g. toy_runs/docker_tour/20261121-170000")
    ap.add_argument("--azure", action="store_true", help="run the agent step on Azure (costs money)")
    a = ap.parse_args(argv)
    if a.command == "list":
        for p, keys in PHASES.items():
            print(f"{p:10} {', '.join(keys)}")
        return 0
    if not a.dir:
        ap.error("--dir is required")
    d = a.dir if a.dir.is_absolute() else ROOT / a.dir
    if a.command == "preflight":
        return preflight(d)
    if a.command == "run":
        skipped = [k for name in a.steps if name.startswith("skip:") for k in PHASES.get(name[5:], [name[5:]])]
        keys = [k for name in a.steps if not name.startswith("skip:") for k in PHASES.get(name, [name])]
        if unknown := [k for k in keys + skipped if k not in BY_KEY]:
            ap.error(f"unknown step(s): {', '.join(unknown)}")
        for k in skipped:
            record(d, base_record(BY_KEY[k], time.time()) | {"status": "skip", "seconds": 0,
                                                             "got": "skipped on request (-Skip / -Only)"})
        results = [run_step(k, d, a.azure) for k in keys]
        return 0 if all(r["status"] != "fail" for r in results) else 1
    if a.command == "finish":
        return finish(d)
    write_report(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
