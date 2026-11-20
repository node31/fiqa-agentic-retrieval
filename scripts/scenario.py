"""Scenario runner: play a file of writes, searches and admin actions against the live service.

    python scripts/scenario.py scenarios/freshness_demo.jsonl --start-cluster --tiny --mode async
    python scripts/scenario.py scenarios/04_multi_node_read_migrate.jsonl --start-cluster --tiny
    python scripts/scenario.py scenarios/freshness_demo.jsonl --url http://127.0.0.1:8000
    python scripts/scenario.py --replay-only toy_runs/scenarios/<run>/events.jsonl

WHAT
    You write a small "script" of operations in a JSON Lines file: writes (upsert, delete),
    searches with expectations, waits, admin actions (migrate, cutover, rollback, finalize) and
    cluster faults (kill_shard, restart_shard). This tool sends them to the router over real HTTP,
    prints one line per operation (did it work? did the search return what we expected?), and
    records every event to ``events.jsonl`` plus a self-contained animated ``replay.html``
    (scripts/replay_html.py). Full guide with the file format: docs/scenarios.md.
    Every op is performed by an ACTOR (user, bot, admin, ops; see resolve_actors): clients search
    and write, admins/ops managers run admin and cluster ops, never the other way round. Lanes
    (threads) run in parallel and coordinate with ``signal`` / ``wait_signal``.

WHY
    The service makes promises that are about TIME and ORDER, which unit tests check but do not
    show: a write is searchable within 1 s of its ack; a model migration never fails a query and
    answers stay identical until the cutover; a crashed shard comes back with every acked write.
    A scenario turns each promise into a few readable lines you can run in sync and async mode,
    and the replay lets you watch it happen at 0.25x speed.

HOW (the story of one run)
    parse_scenario()  text -> validated list of op dicts (errors name the line number)
    run_scenario()    read the optional "config" header -> optionally start a Cluster ->
                      start the StatusSampler thread -> Runner.run_lane("main", ops) ->
                      golden save/check -> summarize() -> write events + replay.html
    Runner            executes ops. ``repeat`` and ``parallel`` are CONTROL ops that run their
                      children; every other op is a LEAF that emits an ``op_start`` event, does
                      one HTTP call (or a sleep / a process kill), then emits an ``op_end`` event
                      with the response summary, latency and expectation ``checks``.
    One lane = one thread = one keep-alive httpx.Client (a fresh client per request would add
    connection setup to every measured latency).

EVENTS (one JSON object per line in events.jsonl; ``t`` = seconds since the run started)
    meta      scenario name, mode, url, n_shards, corpus, models, run id, config, start date
    op_start  id, t, lane, op, line, req (what we send), shard (owner shard for writes)
    op_end    id, t, lane, op, ok, latency_ms, resp (summary), checks [...], error
    status    t, active_model, previous_model, shards {id: n_docs, pending, models, ...}
    summary   totals (see summarize())

CHECKS (inside op_end; ``hard`` checks decide pass/fail, soft ones are only recorded)
    expect / expect_absent      a doc id is (not) in the results; with ``since_ack_ms``
    ok / partial / missing_shards   result health (expect_ok, expect_partial, expect_missing_shards)
    same_as / overlap           compare with an answer saved earlier via ``save_as``
    expect_applied / expect_error   write outcome / "this call must fail" (e.g. a refused cutover)
    visible / ready / drained   wait_visible, wait_ready, flush
    golden                      compare with a golden file from another run (--golden-check)

PITFALLS
    * Versions must increase per doc_id. Omit ``version`` and the runner auto-increments it per
      doc. Against a long-lived cluster use ``{run}`` in doc ids so every run writes NEW docs
      (a re-run with the same id + version 1 would be a no-op on the server).
    * "Found right after the write" is guaranteed only in SYNC mode. In async mode use
      ``"soft": true`` on immediate checks (recorded, never fails) and hard checks after 1 s.
    * ``expect_same_as`` compares exact ranked lists. Only use it when nothing that can change the
      ranking happened in between: new docs (even unrelated ones change BM25's global statistics)
      or a different embedding model (after cutover). Use ``expect_overlap_with`` there instead.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import itertools
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scripts.replay_html import write_replay  # noqa: E402
from src.index.sharding import shard_of  # noqa: E402
from src.util import percentile  # noqa: E402

TINY_CORPUS = ROOT / "tests" / "fixtures" / "tinyfiqa"
DEFAULT_OUT = ROOT / "toy_runs" / "scenarios"
STATE_ROOT = ROOT / "artifacts" / "scenario_runs"
DEFAULT_MODEL = "minilm"  # = src.config.PRIMARY_MODEL's default; not imported to keep startup light

# ----------------------------------------------------------------------------- the file format
# op -> (required fields, optional fields). Unknown fields are an ERROR, so a typo such as
# "expected" instead of "expect" fails loudly instead of silently checking nothing.
_WRITE_OPT = {"version", "expect_applied", "expect_error", "soft"}
OPS: dict[str, tuple[set[str], set[str]]] = {
    "config": (set(), {"shards", "mode", "models", "tiny", "description", "actors"}),
    "upsert": ({"doc_id", "text"}, _WRITE_OPT),
    "delete": ({"doc_id"}, _WRITE_OPT),
    "search": ({"query"}, {"strategy", "k", "fusion", "alpha", "soft", "expect", "expect_absent", "save_as",
                           "expect_same_as", "expect_overlap_with", "expect_ok", "expect_partial",
                           "expect_missing_shards", "expect_error"}),
    "sleep": ({"seconds"}, set()),
    "wait_visible": ({"query", "doc_id"}, {"timeout", "strategy", "k", "absent", "interval"}),
    "flush": (set(), {"timeout"}),
    "status": (set(), set()),
    "label": ({"text"}, set()),
    # admin (model migration, blue/green): see src/service/router.py "MODEL MIGRATION"
    "migrate": ({"model"}, {"throttle_s", "reencode", "expect_error"}),
    "wait_ready": ({"model"}, {"timeout", "interval"}),
    "cutover": ({"model"}, {"expect_error"}),
    "rollback": (set(), {"expect_error"}),
    "finalize": ({"model"}, {"expect_error"}),
    # cluster faults: only with --start-cluster (the runner owns the processes)
    "kill_shard": ({"shard"}, set()),
    "restart_shard": ({"shard"}, {"timeout"}),
    # control flow
    "repeat": ({"ops"}, {"times", "each", "until_done", "until_signal", "var", "start"}),
    "parallel": ({"lanes"}, set()),
    # coordination between lanes: one lane "signal"s a named event, others wait for it
    "signal": ({"name"}, set()),
    "wait_signal": ({"name"}, {"timeout"}),
}
ADMIN_OPS = {"migrate", "wait_ready", "cutover", "rollback", "finalize", "kill_shard", "restart_shard"}


class ScenarioError(ValueError):
    """A problem in the scenario file (message includes the line number)."""


def parse_scenario(text: str) -> list[dict]:
    """Scenario text -> list of validated op dicts (each gets a ``_line`` key).

    Format: JSON Lines. Blank lines and lines whose first non-space char is ``#`` are comments.
    An op MAY span several lines (handy for ``parallel``): we read consecutive JSON values,
    not strictly one per line. An optional ``config`` op must come first.

    >>> [o["op"] for o in parse_scenario('# hi\\n{"op": "sleep", "seconds": 0.1}\\n{"op": "flush"}')]
    ['sleep', 'flush']
    """
    # WHY blank out comments instead of deleting them: line numbers in error messages stay right.
    # A JSON string cannot contain a raw newline, so a line starting with '#' is never inside one.
    lines = ["" if ln.lstrip().startswith("#") else ln for ln in text.splitlines()]
    body = "\n".join(lines)
    dec, pos, ops = _PosDecoder(), 0, []
    while True:
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos >= len(body):
            break
        line = body.count("\n", 0, pos) + 1
        dec.starts.clear()
        try:
            obj, pos = dec.raw_decode(body, pos)
        except json.JSONDecodeError as e:
            raise ScenarioError(f"line {body.count(chr(10), 0, e.pos) + 1}: invalid JSON ({e.msg})") from None
        # Every op gets the line of its OWN opening brace, also inside a multi-line parallel/repeat.
        line_of = {k: body.count("\n", 0, p) + 1 for k, p in dec.starts.items()}
        ops.append(_validate(obj, line, top_level=not ops, line_of=line_of))
    # "repeat until_done" must name a lane that exists, else it would loop forever.
    lanes = set(_lane_names(ops, "main"))
    signals = {op["name"] for op in _walk(ops) if op["op"] == "signal"}
    for op in _walk(ops):
        if op["op"] == "repeat" and op.get("until_done") and op["until_done"] not in lanes:
            raise ScenarioError(f"line {op['_line']}: until_done lane {op['until_done']!r} does not exist "
                                f"(lanes: {sorted(lanes)})")
        # Same for signals: waiting for a signal nobody sends would hang until the timeout.
        wanted = op.get("until_signal") if op["op"] == "repeat" else op.get("name") if op["op"] == "wait_signal" else None
        if wanted is not None and wanted not in signals:
            raise ScenarioError(f"line {op['_line']}: no op signals {wanted!r} (signals: {sorted(signals)})")
    resolve_actors(ops)  # who does what; raises when an actor runs an op its role may not run
    return ops


class _PosDecoder(json.JSONDecoder):
    """A JSON decoder that remembers where every object starts (``starts``: id(obj) -> offset).

    WHY: an op inside a multi-line ``parallel`` block should point to its own line, not to the
    line where the block starts. The C scanner cannot be hooked, so we use the pure-Python one
    (py_make_scanner); scenario files are small, speed does not matter.
    """

    def __init__(self):
        super().__init__()
        self.starts: dict[int, int] = {}

        def parse_object(s_and_end, *args, **kw):
            obj, end = json.decoder.JSONObject(s_and_end, *args, **kw)
            self.starts[id(obj)] = s_and_end[1] - 1   # s_and_end[1] = index just after the '{'
            return obj, end

        self.parse_object = parse_object
        self.scan_once = json.scanner.py_make_scanner(self)


def _validate(op, line: int, top_level: bool = False, line_of: dict | None = None) -> dict:
    """Check one op (and, recursively, its children). Returns it with ``_line`` set."""
    line_of = line_of or {}
    line = line_of.get(id(op), line)
    where = f"line {line}"
    if not isinstance(op, dict) or "op" not in op:
        raise ScenarioError(f"{where}: every op must be an object with an \"op\" field, got {op!r}")
    kind = op["op"]
    if kind not in OPS:
        raise ScenarioError(f"{where}: unknown op {kind!r} (known: {', '.join(OPS)})")
    if kind == "config" and not top_level:
        raise ScenarioError(f"{where}: config must be the first op of the file")
    required, optional = OPS[kind]
    if kind not in ("config", "repeat", "parallel"):
        optional = optional | {"actor"}  # who performs it (see resolve_actors)
    fields = set(op) - {"op", "_line"}
    if missing := required - fields:
        raise ScenarioError(f"{where}: {kind} needs {sorted(missing)}")
    if unknown := fields - required - optional:
        raise ScenarioError(f"{where}: {kind} has unknown field(s) {sorted(unknown)}; allowed: "
                            f"{sorted(required | optional)}")
    op = {**op, "_line": line}
    if kind == "repeat":
        if not ({"times", "each", "until_done", "until_signal"} & fields):
            raise ScenarioError(f"{where}: repeat needs one of times / each / until_done / until_signal")
        if "each" in op:
            _as_list(op["each"], where, "repeat.each")
        op["ops"] = [_validate(o, line, line_of=line_of) for o in _as_list(op["ops"], where, "repeat.ops")]
    elif kind == "parallel":
        lanes = []
        for k, lane in enumerate(_as_list(op["lanes"], where, "parallel.lanes")):
            # A lane is either a bare list of ops or
            # {"name": "...", "actor": "...", "description": "...", "ops": [...]}.
            if isinstance(lane, dict):
                if unknown := set(lane) - {"name", "actor", "description", "ops"}:
                    raise ScenarioError(f"{where}: a lane has unknown field(s) {sorted(unknown)}; "
                                        f"allowed: name, actor, description, ops")
                name, ops, desc = lane.get("name", lane.get("actor", f"lane{k}")), lane.get("ops"), lane.get("description")
            else:
                name, ops, desc = f"lane{k}", lane, None
            lanes.append({"name": str(name), "actor": lane.get("actor") if isinstance(lane, dict) else None,
                          "description": desc,
                          "ops": [_validate(o, line, line_of=line_of) for o in _as_list(ops, where, "lane ops")]})
        op["lanes"] = lanes
    for key in ("expect", "expect_absent", "expect_missing_shards"):
        if key in op and not isinstance(op[key], list):
            raise ScenarioError(f"{where}: {key} must be a list")
    if "expect_overlap_with" in op and not (isinstance(op["expect_overlap_with"], dict) and "ref" in op["expect_overlap_with"]):
        raise ScenarioError(f"{where}: expect_overlap_with must look like {{\"ref\": \"q1\", \"min\": 0.6}}")
    if kind == "config" and op.get("mode", "sync") not in ("sync", "async"):
        raise ScenarioError(f"{where}: config.mode must be \"sync\" or \"async\"")
    if kind == "config" and "actors" in op:
        if not isinstance(op["actors"], dict):
            raise ScenarioError(f"{where}: config.actors must look like {{\"fox\": {{\"role\": \"user\"}}}}")
        for name, a in op["actors"].items():
            if not isinstance(a, dict) or a.get("role") not in ROLES:
                raise ScenarioError(f"{where}: actor {name!r} needs a role, one of {', '.join(ROLES)}")
            if unknown := set(a) - {"role", "emoji", "description"}:
                raise ScenarioError(f"{where}: actor {name!r} has unknown field(s) {sorted(unknown)}")
    return op


def _as_list(v, where: str, what: str) -> list:
    if not isinstance(v, list):
        raise ScenarioError(f"{where}: {what} must be a list")
    return v


def _walk(ops: list[dict]):
    """Every op, depth-first (children of repeat/parallel included)."""
    for op in ops:
        yield op
        if op["op"] == "repeat":
            yield from _walk(op["ops"])
        elif op["op"] == "parallel":
            for lane in op["lanes"]:
                yield from _walk(lane["ops"])


def _lane_names(ops: list[dict], lane: str):
    """All lane names a scenario creates (same naming rule as Runner._parallel)."""
    yield lane
    for op in ops:
        if op["op"] == "repeat":
            yield from _lane_names(op["ops"], lane)
        elif op["op"] == "parallel":
            for ln in op["lanes"]:
                name = ln["name"] if lane == "main" else f"{lane}.{ln['name']}"
                yield from _lane_names(ln["ops"], name)


def describe_plan(ops: list[dict]) -> str:
    """A one-line summary of what a list of ops does, for the replay's lane legend.

    >>> describe_plan([{"op": "repeat", "times": 3, "ops": [{"op": "upsert"}, {"op": "sleep"}]},
    ...                {"op": "search"}, {"op": "search"}])
    '3x [upsert, sleep], search(hybrid) x2'
    """
    parts: list[str] = []
    for op in ops:
        k = op["op"]
        if k in ("label", "config"):
            continue
        if k == "repeat":
            if "each" in op:
                head = f"for each of {len(op['each'])} {op.get('var', 'i')}"
            elif op.get("until_done"):
                head = f"until lane '{op['until_done']}' is done"
            elif op.get("until_signal"):
                head = f"until signal '{op['until_signal']}'"
            else:
                head = f"{op['times']}x"
            parts.append(f"{head} [{describe_plan(op['ops'])}]")
        elif k == "parallel":
            parts.append("fork -> " + ", ".join(ln["name"] for ln in op["lanes"]))
        elif k == "search":
            parts.append(f"search({op.get('strategy', 'hybrid')})")
        elif k in ("migrate", "cutover", "finalize", "wait_ready"):
            parts.append(f"{k} {op.get('model', '')}".strip())
        elif k in ("kill_shard", "restart_shard"):
            parts.append(f"{k} {op.get('shard')}")
        else:
            parts.append(k)
    # Collapse runs of the same step: "search(hybrid), search(hybrid)" -> "search(hybrid) x2".
    out: list[str] = []
    for key, grp in itertools.groupby(parts):
        n = len(list(grp))
        out.append(key if n == 1 else f"{key} x{n}")
    text = ", ".join(out)
    return text if len(text) <= 220 else text[:219] + "…"


def lane_table(ops: list[dict]) -> list[dict]:
    """Every lane the scenario creates: name, parent lane, file line, description, plan.

    Stored in the run's meta event so the replay can explain each lane ("actor") without any
    per-scenario code.
    """
    rows = [{"name": "main", "parent": None, "line": 1, "plan": describe_plan(ops), "actor": None,
             "description": "the top-level ops of the file, run one after another"}]

    def walk(ops, lane):
        for op in ops:
            if op["op"] == "repeat":
                walk(op["ops"], lane)
            elif op["op"] == "parallel":
                for ln in op["lanes"]:
                    name = ln["name"] if lane == "main" else f"{lane}.{ln['name']}"
                    rows.append({"name": name, "parent": lane, "line": op["_line"], "plan": describe_plan(ln["ops"]),
                                 "actor": ln.get("actor") or ln["name"],
                                 "description": ln.get("description")})
                    walk(ln["ops"], name)

    walk(ops, "main")
    return rows


# ----------------------------------------------------------------------------- actors (who does what)
# Every op is performed by an ACTOR with a ROLE. The replay puts clients (user, bot) on the LEFT and
# privileged operators (admin, ops) on the RIGHT, so a reader sees who did what. The split is strict
# both ways: client ops only by clients, admin/ops ops only by admins/ops managers.
ROLES = ("user", "bot", "admin", "ops")
CLIENT_ROLES = {"user", "bot"}
CLIENT_OPS = {"upsert", "delete", "search", "wait_visible"}                          # what clients do
ADMIN_TYPE_OPS = {"migrate", "wait_ready", "cutover", "rollback", "finalize", "flush"}   # the admin's job
OPS_TYPE_OPS = {"kill_shard", "restart_shard", "status"}                                # the ops manager's job
PRIVILEGED_OPS = ADMIN_TYPE_OPS | OPS_TYPE_OPS
NEUTRAL_OPS = {"sleep", "label", "signal", "wait_signal"}   # anyone may pause, narrate or coordinate
# Human users are shown as animal personas; assigned in order of first appearance.
ANIMALS = [("fox", "🦊"), ("elephant", "🐘"), ("dog", "🐶"), ("owl", "🦉"), ("panda", "🐼"), ("koala", "🐨"),
           ("tiger", "🐯"), ("frog", "🐸"), ("penguin", "🐧"), ("lion", "🦁"), ("rabbit", "🐰"), ("bear", "🐻")]
ROLE_EMOJI = {"bot": "🤖", "admin": "🛠️", "ops": "🧑‍🔧"}
_GENERIC_USER = re.compile(r"^(user|users|client|clients|main|lane\d+)$", re.I)


def role_from_name(name: str) -> str | None:
    """Role suggested by an actor/lane name, or None.

    A synthetic probe/watchdog SEARCHES, so it is a bot (a client), not an ops manager.

    >>> [role_from_name(n) for n in ("admin-2", "ops", "chaos", "watchdog", "probe-bot", "bot-1", "writer-a",
    ...                              "Read Bot #2", "fox", "zebra")]
    ['admin', 'ops', 'ops', 'bot', 'bot', 'bot', 'bot', 'bot', 'user', None]
    """
    n = name.lower()
    if re.match(r"(admin|build|migrat|operator)", n):
        return "admin"
    if re.match(r"(ops|chaos|sre|oncall)", n):
        return "ops"
    if re.match(r"(bot|load|burst|writer|reader|checker|probe|canary|watchdog|monitor)", n) or re.search(r"\bbot\b", n):
        return "bot"
    if n in {a for a, _ in ANIMALS}:
        return "user"
    return None


def resolve_actors(ops: list[dict]) -> list[dict]:
    """Decide the actor of every op (sets ``op["_actor"]``) and return the actor table.

    Rules (docs/scenarios.md "Actors"):
      * the actor of an op = its ``"actor"`` field, else its lane's actor (the lane's ``"actor"``
        field, else the lane's own name). Top-level ops (lane "main") without ``"actor"`` go to a
        default actor by op type: client ops -> the first declared user (else "user"), admin ops ->
        the first declared admin (else "admin"), ops-manager ops -> first declared ops (else "ops").
        A top-level sleep/label has no actor (it is part of the story, shown in the middle).
      * the role of an actor = ``config.actors[name].role``, else what its name suggests
        (role_from_name), else what it does: any admin op -> admin, any ops-manager op -> ops,
        otherwise -> user. Neutral ops (sleep, label, signal, wait_signal) do not count.
      * VALIDATION, both ways (ScenarioError, never a silent fix): a client actor (user, bot) must
        not run an admin/ops op, and an admin/ops actor must not run a client op (search, upsert,
        delete, wait_visible). Neutral ops are allowed for everybody.
    """
    cfg = ops[0] if ops and ops[0]["op"] == "config" else {}
    declared: dict[str, dict] = dict(cfg.get("actors") or {})

    def first(roles, fallback):
        return next((n for n, a in declared.items() if a["role"] in roles), fallback)

    default = {"client": first(("user",), None) or first(("bot",), "user"),
               "admin": first(("admin",), "admin"), "ops": first(("ops",), "ops")}
    order: list[str] = []
    ops_of: dict[str, list[dict]] = {}

    def assign(op, lane_actor):
        k = op["op"]
        if k in ("config", "label"):
            op["_actor"] = None
            return
        actor = op.get("actor") or lane_actor
        if actor is None:   # top-level op without an actor
            if k in CLIENT_OPS:
                actor = default["client"]
            elif k in ADMIN_TYPE_OPS:
                actor = default["admin"]
            elif k in OPS_TYPE_OPS:
                actor = default["ops"]
        op["_actor"] = actor
        if actor is not None:
            if actor not in ops_of:
                order.append(actor)
                ops_of[actor] = []
            ops_of[actor].append(op)

    def walk(ops, lane_actor):
        for op in ops:
            if op["op"] == "repeat":
                op["_actor"] = None
                walk(op["ops"], lane_actor)
            elif op["op"] == "parallel":
                op["_actor"] = None
                for ln in op["lanes"]:
                    walk(ln["ops"], ln.get("actor") or ln["name"])
            else:
                assign(op, lane_actor)

    walk(ops, None)
    for name in declared:   # declared actors show up even if they never act
        if name not in ops_of:
            order.append(name)
            ops_of[name] = []

    table, animals_used = [], set()
    free_animals = iter([a for a in ANIMALS if a[0] not in {n.lower() for n in order}])
    for name in order:
        kinds = {o["op"] for o in ops_of[name]}
        if name in declared:
            role, how = declared[name]["role"], "declared"
        elif role_from_name(name):
            role, how = role_from_name(name), "from its name"
        elif kinds & ADMIN_TYPE_OPS:
            role, how = "admin", "from its ops"
        elif kinds & OPS_TYPE_OPS:
            role, how = "ops", "from its ops"
        else:
            role, how = "user", "from its ops"
        if role in CLIENT_ROLES:
            for o in ops_of[name]:
                if o["op"] in PRIVILEGED_OPS:
                    raise ScenarioError(
                        f"line {o['_line']}: {o['op']!r} by actor {name!r} (role {role}, {how}): admin ops must run "
                        f"in an admin/ops actor. Add \"actor\": \"admin\" (or \"ops\") to the op, or run it in a lane "
                        f"whose actor has role admin/ops in config.actors")
        else:
            for o in ops_of[name]:
                if o["op"] in CLIENT_OPS:
                    raise ScenarioError(
                        f"line {o['_line']}: {o['op']!r} by actor {name!r} (role {role}, {how}): client ops "
                        f"(search/upsert/delete/wait_visible) must run in a user/bot actor. Add \"actor\": "
                        f"\"<a user or bot>\" to the op (e.g. a probe bot for health searches), or declare the "
                        f"lane's actor with role user/bot in config.actors")
        emoji = (declared.get(name) or {}).get("emoji")
        display = name
        if role == "user":
            animal = next((a for a in ANIMALS if a[0] == name.lower()), None)
            if animal is None:
                animal = next(free_animals, ("user", "🙂"))
                if _GENERIC_USER.match(name):
                    display = animal[0]   # "user" / "main" -> "fox": a persona reads better
            animals_used.add(animal[0])
            emoji = emoji or animal[1]
        emoji = emoji or ROLE_EMOJI.get(role, "🙂")
        counts: dict[str, int] = {}
        for o in ops_of[name]:
            counts[o["op"]] = counts.get(o["op"], 0) + 1
        table.append({"name": name, "display": display, "role": role, "emoji": emoji, "how": how,
                      "description": (declared.get(name) or {}).get("description"), "op_counts": counts})
    return table


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render(value, variables: dict):
    """Replace ``{name}`` placeholders (e.g. ``{i}``, ``{run}``) in every string of ``value``.

    Unknown placeholders are left as they are.

    >>> render({"doc_id": "load-{i}", "k": 3, "x": "{nope}"}, {"i": 7})
    {'doc_id': 'load-7', 'k': 3, 'x': '{nope}'}
    """
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: str(variables[m[1]]) if m[1] in variables else m[0], value)
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: render(v, variables) for k, v in value.items()}
    return value


def fmt_ms(ms: float) -> str:
    """Human duration: ``fmt_ms(12.34) == '12 ms'``, ``fmt_ms(2070) == '2.07 s'``."""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def answer_diff(ref: list[str], got: list[str]) -> dict:
    """How two ranked id lists differ: first differing rank, ids lost, ids gained.

    >>> answer_diff(["a", "b", "c"], ["a", "c", "d"])
    {'first_diff_rank': 2, 'missing': ['b'], 'extra': ['d']}
    """
    first = next((i + 1 for i, (a, b) in enumerate(zip(ref, got)) if a != b),
                 None if len(ref) == len(got) else min(len(ref), len(got)) + 1)
    return {"first_diff_rank": first, "missing": [d for d in ref if d not in got],
            "extra": [d for d in got if d not in ref]}


# ----------------------------------------------------------------------------- console output
class Console:
    """Prints one line per event; ANSI colors when the terminal supports them."""

    CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "dim": "2", "bold": "1", "magenta": "35"}

    def __init__(self, color: bool | None = None, quiet: bool = False, print_fn=print):
        self.color = _supports_color() if color is None else color
        self.quiet, self.print_fn = quiet, print_fn
        self.lock = threading.Lock()  # lanes print from several threads; keep lines whole
        enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
        # PITFALL: a cp1252 console cannot print check marks; fall back to ASCII symbols.
        self.ok_sym, self.bad_sym = ("✓", "✗") if "utf" in enc else ("OK", "X")

    def c(self, text: str, color: str) -> str:
        return f"\033[{self.CODES[color]}m{text}\033[0m" if self.color else text

    def line(self, text: str, important: bool = False) -> None:
        if self.quiet and not important:
            return
        with self.lock:
            self.print_fn(text)


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # WHY: classic Windows consoles show raw escape codes unless "virtual terminal
        # processing" is switched on for the output handle (Windows Terminal has it on already).
        try:
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not k32.GetConsoleMode(h, ctypes.byref(mode)):
                return False
            return bool(k32.SetConsoleMode(h, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:  # noqa: BLE001 - any failure just means "no colors"
            return False
    return True


# ----------------------------------------------------------------------------- recording
class Recorder:
    """Thread-safe event log: keeps events in memory and appends each one to events.jsonl."""

    def __init__(self, path: Path):
        self.t0 = time.perf_counter()
        self.path = path
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self._f = open(path, "w", encoding="utf-8")

    def now(self) -> float:
        """Seconds since the run started (perf_counter: monotonic, sub-microsecond)."""
        return time.perf_counter() - self.t0

    def emit(self, ev: dict) -> dict:
        with self.lock:
            self.events.append(ev)
            # flush() per event: if the run crashes, the log up to the crash is still on disk.
            self._f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._f.flush()
        return ev

    def close(self) -> None:
        self._f.close()


def compact_status(body: dict) -> dict:
    """GET /admin/status body -> the small dict a ``status`` event stores.

    {"active_model", "previous_model", "shards": {shard_id: {n_docs, healthy, mode, pending,
    oldest_ms, lag_p50/p95/max, accepted, applied, batches, bp_waits, noop, replayed,
    models: {model: {ready, progress, n}}}}}. A shard the router could not reach is
    {"missing": True, "healthy": False}.
    """
    shards = {}
    for sid, s in (body.get("shards") or {}).items():
        a = s.get("async") or {}
        lag = a.get("lag_ms") or {}
        counters = s.get("counters") or {}
        shards[sid] = {"n_docs": s.get("n_docs"), "healthy": s.get("healthy"), "mode": s.get("apply_mode", "sync"),
                       "pending": a.get("pending", 0), "oldest_ms": a.get("oldest_pending_age_ms", 0.0),
                       "lag_p50": lag.get("p50"), "lag_p95": lag.get("p95"), "lag_max": lag.get("max"),
                       "accepted": a.get("accepted"), "applied": a.get("applied"), "batches": a.get("batches"),
                       "bp_waits": a.get("backpressure_waits"), "noop": counters.get("noop"),
                       "replayed": counters.get("replayed"),
                       "models": {m: {"ready": v.get("ready"), "progress": v.get("progress"), "n": v.get("n")}
                                  for m, v in (s.get("models") or {}).items()}}
    for sid in body.get("missing_shards") or []:
        shards[str(sid)] = {"n_docs": None, "healthy": False, "missing": True}
    return {"active_model": body.get("active_model"), "previous_model": body.get("previous_model"),
            "shards": dict(sorted(shards.items(), key=lambda kv: int(kv[0])))}


class StatusSampler(threading.Thread):
    """Background thread: GET /admin/status every ``interval`` s -> ``status`` events.

    WHY: the replay shows each shard's doc count, async queue and model-build progress OVER
    TIME, so you can watch a queue fill and drain, or a backfill creep to 100%. Cheap: one
    small fan-out per sample.
    """

    def __init__(self, url: str, rec: Recorder, interval: float):
        super().__init__(name="status-sampler", daemon=True)
        self.url, self.rec, self.interval = url, rec, interval
        self.stop_event = threading.Event()

    def run(self) -> None:
        with make_client(self.url, 10.0) as http:
            while not self.stop_event.is_set():
                try:
                    body = http.get("/admin/status").raise_for_status().json()
                    self.rec.emit({"type": "status", "t": round(self.rec.now(), 4), "src": "sampler",
                                   **compact_status(body)})
                except httpx.HTTPError:
                    pass  # a sample is best-effort; the ops themselves report real errors
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(15)


class CallError(RuntimeError):
    """An HTTP call failed; keeps the status code (None = no response, e.g. connection refused)."""

    def __init__(self, msg: str, status: int | None):
        super().__init__(msg)
        self.status = status


_SSL_CTX = None


def make_client(base_url: str, timeout: float) -> httpx.Client:
    """A keep-alive httpx.Client that reuses ONE process-wide SSL context.

    PITFALL (measured on Windows): ``httpx.Client()`` builds a fresh SSL context every time, which
    loads the CA bundle and takes ~0.5 s, even for plain http:// URLs. With one client per lane
    (and nested parallel blocks) that delay would show up as fake gaps in the replay.
    """
    global _SSL_CTX
    if _SSL_CTX is None:
        import ssl

        _SSL_CTX = ssl.create_default_context()
    return httpx.Client(base_url=base_url, timeout=timeout, verify=_SSL_CTX)


def _call(http: httpx.Client, method: str, path: str, **kw) -> dict:
    """One HTTP call -> JSON body; any failure becomes a CallError with a readable message."""
    try:
        r = http.request(method, path, **kw)
    except httpx.HTTPError as e:
        raise CallError(f"{type(e).__name__}: {e}", None) from None
    if r.status_code >= 400:
        # The router puts the reason in "detail" (e.g. "refusing cutover: ... not ready").
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise CallError(f"HTTP {r.status_code}: {str(detail)[:240]}", r.status_code)
    return r.json()


# ----------------------------------------------------------------------------- the runner
class Runner:
    """Executes ops, emitting ``op_start``/``op_end`` events and console lines."""

    def __init__(self, url: str, rec: Recorder, console: Console, n_shards: int, run_id: str,
                 cluster=None, active_model: str | None = None, http_timeout: float = 60.0):
        self.url, self.rec, self.console, self.n_shards = url, rec, console, n_shards
        self.run_id, self.cluster, self.http_timeout = run_id, cluster, http_timeout
        self.active_model = active_model
        self.display: dict[str, str] = {}   # actor name -> persona shown on the console (set by run_scenario)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        # Last version this run used per doc (auto-increment when a write omits "version").
        self.versions: dict[str, int] = {}
        # doc_id -> recorder time of its last successful write ack. Used for "time since ack"
        # in search checks and as the start of the wait_visible lag.
        self.last_ack: dict[str, float] = {}
        # save_as name -> {"ids": ranked doc ids, "query", "strategy", "k", "model"}.
        self.saved: dict[str, dict] = {}
        # lane name -> Event set when the lane finished (for "repeat": {"until_done": lane}).
        self.lane_done: dict[str, threading.Event] = {}
        # signal name -> Event ("signal" sets it; "wait_signal" / repeat "until_signal" read it).
        self.signals: dict[str, threading.Event] = {}

    def _signal(self, name: str) -> threading.Event:
        with self._lock:   # lanes may ask for the same signal at the same moment
            return self.signals.setdefault(name, threading.Event())

    # ---- control flow
    def run_lane(self, lane: str, ops: list[dict], variables: dict) -> None:
        """Run ``ops`` in order on THIS thread with one keep-alive client."""
        with make_client(self.url, self.http_timeout) as http:
            self._run_ops(ops, lane, variables, http)

    def _run_ops(self, ops: list[dict], lane: str, variables: dict, http: httpx.Client) -> None:
        for op in ops:
            if op["op"] == "config":
                continue  # already applied by run_scenario before the cluster started
            if op["op"] == "repeat":
                self._repeat(op, lane, variables, http)
            elif op["op"] == "parallel":
                self._parallel(op, lane, variables)
            else:
                self._leaf(op, lane, variables, http)

    def _repeat(self, op: dict, lane: str, variables: dict, http: httpx.Client) -> None:
        """times: i = start..start+times-1 | each: i = every list item | until_done: loop until a lane ends
        | until_signal: loop until another lane sends that signal."""
        var, start = op.get("var", "i"), int(op.get("start", 0))
        if "each" in op:
            values = render(op["each"], {"run": self.run_id, **variables})
            for v in values:
                self._run_ops(op["ops"], lane, {**variables, var: v}, http)
            return
        done = (self.lane_done.get(op["until_done"]) if op.get("until_done")
                else self._signal(op["until_signal"]) if op.get("until_signal") else None)
        cap = int(op["times"]) if "times" in op else None
        for i in itertools.count(start):
            if cap is not None and i >= start + cap:
                break
            # do-while: at least one round even if the other lane finished very fast (tiny corpus).
            if done is not None and done.is_set() and i > start:
                break
            self._run_ops(op["ops"], lane, {**variables, var: i}, http)

    def _parallel(self, op: dict, lane: str, variables: dict) -> None:
        """Run each lane in its own thread; return when all are done (fork/join)."""
        # Nested parallel blocks get prefixed names so lanes stay unique ("main" has no prefix).
        names = [ln["name"] if lane == "main" else f"{lane}.{ln['name']}" for ln in op["lanes"]]
        for n in names:  # register BEFORE starting threads: a reader may wait on a writer lane
            self.lane_done[n] = threading.Event()
        oid, t0 = self._begin(lane, "parallel", op["_line"], {"lanes": names})
        errors: list[str] = []

        def target(name, ops):
            try:
                self.run_lane(name, ops, variables)
            except Exception as e:  # noqa: BLE001 - a bug in one lane must not hang the join
                errors.append(f"{name}: {type(e).__name__}: {e}")
            finally:
                self.lane_done[name].set()

        threads = [threading.Thread(target=target, args=(n, ln["ops"]), name=n) for n, ln in zip(names, op["lanes"])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._end(oid, lane, "parallel", t0, {"lanes": names},
                  {"ok": not errors, "error": "; ".join(errors) or None, "checks": []})

    # ---- leaves
    def _leaf(self, op: dict, lane: str, variables: dict, http: httpx.Client) -> None:
        kind = op["op"]
        op = render(op, {"run": self.run_id, "lane": lane, **variables})
        req, extra = self._prepare(kind, op)
        oid, t0 = self._begin(lane, kind, op["_line"], req, actor=op.get("_actor"), **extra)
        hard = not op.get("soft", False)
        try:
            end = getattr(self, f"_do_{kind}")(op, req, http, t0)
            if op.get("expect_error"):
                end.setdefault("checks", []).append({"kind": "expect_error", "pass": False, "hard": hard,
                                                     "detail": "the call succeeded but an error was expected"})
        except CallError as e:
            if op.get("expect_error"):
                # The failure IS the expected outcome (e.g. cutover to an unbuilt model -> 409).
                end = {"resp": {"error": str(e), "status": e.status},
                       "checks": [{"kind": "expect_error", "pass": True, "hard": hard, "status": e.status,
                                   "detail": str(e)}]}
            else:
                end = {"error": str(e), "status": e.status, "checks": []}
        except Exception as e:  # noqa: BLE001 - every failure is recorded as a failed op, not raised
            end = {"error": f"{type(e).__name__}: {e}"[:300], "checks": []}
        self._end(oid, lane, kind, t0, req, end, who=op.get("_actor"))

    def _prepare(self, kind: str, op: dict) -> tuple[dict, dict]:
        """Build the request we will send (logged in op_start) + extra event fields."""
        if kind in ("upsert", "delete"):
            doc = op["doc_id"]
            with self._lock:  # lanes may write the same doc; versions must still increase
                v = int(op["version"]) if "version" in op else self.versions.get(doc, 0) + 1
                self.versions[doc] = max(v, self.versions.get(doc, 0))
            req = {"doc_id": doc, "version": v, **({"text": op["text"]} if kind == "upsert" else {})}
            # The owner shard, computed with the SAME rule the router uses (crc32 % n), so the
            # replay can animate the write to the right box without reimplementing crc32 in JS.
            return req, {"shard": shard_of(doc, self.n_shards)}
        if kind == "search":
            req = {"query": op["query"], "k": int(op.get("k", 10)), "strategy": op.get("strategy", "hybrid")}
            req.update({k: op[k] for k in ("fusion", "alpha") if k in op})
            return req, {}
        if kind == "wait_visible":
            return {"query": op["query"], "doc_id": op["doc_id"], "timeout": float(op.get("timeout", 2.0)),
                    "strategy": op.get("strategy", "hybrid"), "k": int(op.get("k", 10)),
                    "absent": bool(op.get("absent", False))}, {"shard": shard_of(op["doc_id"], self.n_shards)}
        if kind == "migrate":
            return ({"model": op["model"], "reencode": bool(op.get("reencode", False)),
                     "throttle_s": float(op.get("throttle_s", 0.0))}, {"from_model": self.active_model})
        if kind in ("cutover", "rollback", "finalize"):
            return ({"model": op["model"]} if "model" in op else {}), {"from_model": self.active_model}
        return {k: v for k, v in op.items() if k not in ("op", "soft", "actor") and not k.startswith("_")}, {}

    # writes
    def _do_upsert(self, op, req, http, t0):
        return self._write("/upsert", op, req, http)

    def _do_delete(self, op, req, http, t0):
        return self._write("/delete", op, req, http)

    def _write(self, path, op, req, http) -> dict:
        body = _call(http, "POST", path, json=req)
        t_ack = self.rec.now()
        if body.get("applied"):
            with self._lock:
                self.last_ack[req["doc_id"]] = t_ack
        resp = {k: body[k] for k in ("applied", "visible", "mode", "shard", "attempts", "reason", "current_version")
                if k in body}
        checks = []
        if "expect_applied" in op:
            want = bool(op["expect_applied"])
            checks.append({"kind": "expect_applied", "doc_id": req["doc_id"], "want": want,
                           "got": bool(body.get("applied")), "pass": bool(body.get("applied")) == want,
                           "hard": not op.get("soft", False)})
        return {"resp": resp, "checks": checks}

    # searches
    def _do_search(self, op, req, http, t0):
        body = _call(http, "POST", "/search", json=req)
        ids = [r["doc_id"] for r in body.get("results", [])]
        partial = bool(body.get("partial", False))
        hard = not op.get("soft", False)
        checks = []
        for key, want_present in (("expect", True), ("expect_absent", False)):
            for doc in op.get(key, []):
                present = doc in ids
                ack = self.last_ack.get(doc)
                checks.append({"kind": key, "doc_id": doc, "found": present, "pass": present == want_present,
                               "hard": hard, "rank": ids.index(doc) + 1 if present else None,
                               # Time between the doc's last ack and the moment this search was SENT:
                               # the number the freshness guarantee is about.
                               "since_ack_ms": round((t0 - ack) * 1e3, 1) if ack is not None and ack <= t0 else None})
        if op.get("expect_ok"):
            # "ok" = what a user would call a working search: no error (else we would not be
            # here), every shard answered, and at least one hit.
            checks.append({"kind": "ok", "pass": not partial and bool(ids), "hard": hard,
                           "detail": "partial" if partial else ("no results" if not ids else "")})
        if "expect_partial" in op:
            want = bool(op["expect_partial"])
            checks.append({"kind": "partial", "want": want, "got": partial, "pass": partial == want, "hard": hard})
        if "expect_missing_shards" in op:
            want = sorted(int(s) for s in op["expect_missing_shards"])
            got = sorted(body.get("missing_shards", []))
            checks.append({"kind": "missing_shards", "want": want, "got": got, "pass": want == got, "hard": hard})
        if ref := op.get("expect_same_as"):
            checks.append(self._compare(ref, ids, body.get("model"), hard, exact=True))
        if ov := op.get("expect_overlap_with"):
            checks.append(self._compare(ov["ref"], ids, body.get("model"), hard, exact=False,
                                        min_overlap=float(ov.get("min", 0.5))))
        if name := op.get("save_as"):
            with self._lock:
                self.saved[name] = {"ids": ids, "query": req["query"], "strategy": req["strategy"], "k": req["k"],
                                    "model": body.get("model")}
        resp = {"n": len(ids), "partial": partial, "missing_shards": body.get("missing_shards", []),
                "server_ms": body.get("latency_ms"), "model": body.get("model"),
                "results": [{"doc_id": r["doc_id"], "score": round(float(r.get("score", 0.0)), 4)}
                            for r in body.get("results", [])], **({"saved_as": name} if name else {})}
        return {"resp": resp, "checks": checks}

    def _compare(self, ref: str, ids: list[str], model: str | None, hard: bool, exact: bool,
                 min_overlap: float = 0.0) -> dict:
        """same_as (exact ranked list) or overlap (fraction of the saved ids still present)."""
        saved = self.saved.get(ref)
        kind = "same_as" if exact else "overlap"
        if saved is None:
            return {"kind": kind, "ref": ref, "pass": False, "hard": hard,
                    "detail": f"no answer saved as {ref!r} yet (save_as must run first)"}
        overlap = len(set(saved["ids"]) & set(ids)) / max(len(saved["ids"]), 1)
        out = {"kind": kind, "ref": ref, "hard": hard, "overlap": round(overlap, 3), "ref_model": saved["model"],
               "model": model, "diff": answer_diff(saved["ids"], ids), "ref_ids": saved["ids"]}
        if exact:
            out["pass"] = saved["ids"] == ids
        else:
            out["min"], out["pass"] = min_overlap, overlap >= min_overlap
        return out

    def _do_wait_visible(self, op, req, http, t0):
        """Poll /search until the doc appears (or disappears with ``absent``); measure the lag."""
        doc, absent = req["doc_id"], req["absent"]
        interval = float(op.get("interval", 0.02))
        sreq = {"query": req["query"], "k": req["k"], "strategy": req["strategy"]}
        ack = self.last_ack.get(doc)
        polls = 0
        while True:
            polls += 1
            ids = [r["doc_id"] for r in _call(http, "POST", "/search", json=sreq).get("results", [])]
            t = self.rec.now()
            if (doc in ids) != absent:
                # The lag is an UPPER bound: it includes the round-trip of the search that saw it.
                lag = round((t - ack) * 1e3, 1) if ack is not None else None
                return {"resp": {"polls": polls, "lag_ms": lag, "first_poll": polls == 1,
                                 "waited_ms": round((t - t0) * 1e3, 1)},
                        "checks": [{"kind": "visible", "doc_id": doc, "absent": absent, "pass": True,
                                    "hard": True, "lag_ms": lag}]}
            if t - t0 >= req["timeout"]:
                return {"resp": {"polls": polls, "lag_ms": None, "waited_ms": round((t - t0) * 1e3, 1)},
                        "checks": [{"kind": "visible", "doc_id": doc, "absent": absent, "pass": False,
                                    "hard": True, "lag_ms": None}]}
            time.sleep(interval)

    # misc
    def _do_sleep(self, op, req, http, t0):
        time.sleep(float(op["seconds"]))
        return {"checks": []}

    def _do_flush(self, op, req, http, t0):
        timeout = float(op.get("timeout", 10))
        body = _call(http, "POST", "/admin/flush", params={"timeout": timeout}, timeout=timeout + 5)
        return {"resp": body, "checks": [{"kind": "drained", "pass": bool(body.get("drained")), "hard": True}]}

    def _status_event(self, http, src: str) -> dict:
        st = compact_status(_call(http, "GET", "/admin/status"))
        # Also a "status" sample, so the replay's cluster panel updates even with --no-sample.
        self.rec.emit({"type": "status", "t": round(self.rec.now(), 4), "src": src, **st})
        return st

    def _do_status(self, op, req, http, t0):
        return {"resp": self._status_event(http, "op"), "checks": []}

    def _do_label(self, op, req, http, t0):
        return {"checks": []}

    # coordination between lanes (no HTTP): "admin cut over" -> "elephant may check the answers now"
    def _do_signal(self, op, req, http, t0):
        self._signal(op["name"]).set()
        return {"checks": []}

    def _do_wait_signal(self, op, req, http, t0):
        timeout = float(op.get("timeout", 120))
        got = self._signal(op["name"]).wait(timeout)
        resp = {"waited_ms": round((self.rec.now() - t0) * 1e3, 1)}
        # Not a "check" (it would inflate the pass count); a missing signal is an op error.
        return {"resp": resp, "checks": []} if got else \
            {"resp": resp, "checks": [], "error": f"no signal {op['name']!r} within {timeout:g} s"}

    # admin: blue/green model migration (router.py "MODEL MIGRATION")
    def _do_migrate(self, op, req, http, t0):
        body = _call(http, "POST", "/admin/migrate", json=req)
        return {"resp": {"target": body.get("target"), "shards": body.get("shards")}, "checks": []}

    def _do_wait_ready(self, op, req, http, t0):
        """Poll /admin/status until EVERY shard reports ``model`` ready (backfill done)."""
        model, timeout = op["model"], float(op.get("timeout", 120))
        interval = float(op.get("interval", 0.1))
        polls = 0
        while True:
            polls += 1
            st = self._status_event(http, "wait_ready")  # each poll = one progress sample for the replay
            shards = st["shards"]
            progress = {sid: (s.get("models") or {}).get(model, {}).get("progress") for sid, s in shards.items()}
            ready = bool(shards) and all(s.get("healthy") and (s.get("models") or {}).get(model, {}).get("ready")
                                         for s in shards.values())
            waited = self.rec.now() - t0
            if ready or waited >= timeout:
                return {"resp": {"polls": polls, "waited_ms": round(waited * 1e3, 1), "progress": progress},
                        "checks": [{"kind": "ready", "model": model, "pass": ready, "hard": True}]}
            time.sleep(interval)

    def _switch(self, http, path: str, req: dict | None) -> dict:
        body = _call(http, "POST", path, json=req) if req is not None else _call(http, "POST", path)
        with self._lock:
            self.active_model = body.get("active_model", self.active_model)
        return {"resp": {"active_model": body.get("active_model"), "previous_model": body.get("previous_model")},
                "checks": []}

    def _do_cutover(self, op, req, http, t0):
        return self._switch(http, "/admin/cutover", {"model": op["model"]})

    def _do_rollback(self, op, req, http, t0):
        return self._switch(http, "/admin/rollback", None)

    def _do_finalize(self, op, req, http, t0):
        body = _call(http, "POST", "/admin/finalize", json={"model": op["model"]})
        return {"resp": {"dropped": body.get("dropped"), "missing_shards": body.get("missing_shards", [])},
                "checks": []}

    # cluster faults (need the Cluster object, i.e. --start-cluster)
    def _need_cluster(self):
        if self.cluster is None:
            raise RuntimeError("kill_shard/restart_shard need --start-cluster (the runner must own the processes)")
        return self.cluster

    def _do_kill_shard(self, op, req, http, t0):
        c, i = self._need_cluster(), int(op["shard"])
        c.kill(f"shard{i}")  # hard kill: no graceful shutdown, like a crash (cluster.py)
        return {"resp": {"killed": f"shard{i}"}, "checks": []}

    def _do_restart_shard(self, op, req, http, t0):
        c, i = self._need_cluster(), int(op["shard"])
        c.start_shard(i)  # base corpus + WAL replay happen before /health answers
        c.wait_healthy([c.shard_url(i)], float(op.get("timeout", 300)))
        h = _call(http, "GET", f"{c.shard_url(i)}/health")  # absolute URL: bypasses base_url
        return {"resp": {"restarted": f"shard{i}", "n_docs": h.get("n_docs"),
                         "replayed": (h.get("counters") or {}).get("replayed")}, "checks": []}

    # golden files (cross-run comparison; called by run_scenario, not from scenario files)
    def _do_golden_check(self, op, req, http, t0):
        golden = json.loads(Path(op["path"]).read_text(encoding="utf-8"))
        answers = golden.get("answers", {})
        common = [n for n in answers if n in self.saved]
        checks = []
        for name in common:
            ref, got = answers[name]["ids"], self.saved[name]["ids"]
            checks.append({"kind": "golden", "ref": name, "pass": ref == got, "hard": True,
                           "overlap": round(len(set(ref) & set(got)) / max(len(ref), 1), 3),
                           "diff": answer_diff(ref, got), "ref_ids": ref,
                           "ref_model": answers[name].get("model"), "model": self.saved[name].get("model")})
        if not common:
            checks.append({"kind": "golden", "ref": None, "pass": False, "hard": True,
                           "detail": "no save_as name in common with the golden file"})
        return {"resp": {"golden_meta": golden.get("meta", {}), "compared": common,
                         "only_in_golden": [n for n in answers if n not in self.saved],
                         "only_in_run": [n for n in self.saved if n not in answers]}, "checks": checks}

    # ---- events + console
    def _begin(self, lane: str, kind: str, line: int, req: dict, **extra) -> tuple[int, float]:
        oid, t0 = next(self._ids), self.rec.now()
        self.rec.emit({"type": "op_start", "id": oid, "t": round(t0, 4), "lane": lane, "op": kind,
                       "line": line, "req": req, **extra})
        return oid, t0

    def _end(self, oid: int, lane: str, kind: str, t0: float, req: dict, end: dict, who: str | None = None) -> None:
        t1 = self.rec.now()
        checks = end.get("checks", [])
        ok = not end.get("error") and all(c["pass"] for c in checks if c.get("hard"))
        ev = self.rec.emit({"type": "op_end", "id": oid, "t": round(t1, 4), "lane": lane, "op": kind, "ok": ok,
                            "latency_ms": round((t1 - t0) * 1e3, 2), "resp": end.get("resp", {}),
                            "checks": checks, "error": end.get("error")})
        self._print(t0, self.display.get(who, who) if who else lane, kind, req, ev)  # console names the actor

    def _print(self, t0: float, lane: str, kind: str, req: dict, ev: dict) -> None:
        con, resp = self.console, ev["resp"]
        c, ok_s, bad_s = con.c, con.ok_sym, con.bad_sym
        head = f"[{t0:8.3f}s] {lane:<10} "
        if kind == "label":
            con.line(head + c(f"── {req.get('text', '')} ──", "magenta"), important=True)
            return
        if ev["error"]:
            con.line(head + f"{kind.upper():<7} " + c(f"ERROR {bad_s} {ev['error']}", "red"), important=True)
            return
        ms = f"{ev['latency_ms']:.1f} ms"
        expected_err = next((ch for ch in ev["checks"] if ch["kind"] == "expect_error" and ch["pass"]), None)
        if expected_err:
            subject = f"{req['doc_id']} v{req.get('version')} " if kind in ("upsert", "delete") else ""
            text = f"{kind.upper():<7} {subject}failed as expected: {expected_err['detail']}"
        elif kind in ("upsert", "delete"):
            what = f"{kind.upper():<7} {req['doc_id']} v{req['version']} -> shard {shard_of(req['doc_id'], self.n_shards)}"
            if resp.get("applied"):
                state = (" async: queued, not yet visible" if resp.get("mode") == "async"
                         else " sync: applied, searchable now")
                msg = f"{c('OK', 'green')} applied (ack {ms},{state})"
            else:
                msg = c(f"no-op ({resp.get('reason')}, current v{resp.get('current_version')})", "yellow") + f" ({ms})"
            text = f"{what} ... {msg}"
        elif kind == "search":
            text = (f"SEARCH  \"{_short(req['query'])}\" [{req['strategy']} k={req['k']}] -> {resp['n']} hits "
                    f"({ms}, model {resp.get('model')})")
            if resp.get("partial"):
                text += c(f" PARTIAL: shards {resp.get('missing_shards')} missing", "yellow")
            if resp.get("saved_as"):
                text += c(f" saved as {resp['saved_as']}", "cyan")
        elif kind == "wait_visible":
            verb = "gone" if req["absent"] else "visible"
            if ev["checks"] and ev["checks"][0]["pass"]:
                # THE freshness number: time from the doc's ack to the first search that saw it.
                lag_ms = resp.get("lag_ms")
                polls = "first poll" if resp.get("first_poll") else f"{resp['polls']} polls"
                if lag_ms is None:
                    text = f"WAIT    {req['doc_id']} {verb.upper()} ({polls}; no ack in this run to measure from)"
                else:
                    budget = (c(f"freshness budget 1 s {ok_s}", "green") if lag_ms < 1000
                              else c("OVER the 1 s freshness budget", "red"))
                    text = f"WAIT    {req['doc_id']} {verb.upper()} after {lag_ms:.1f} ms ({polls}) · {budget}"
            else:
                text = f"WAIT    {req['doc_id']} NOT {verb} within {req['timeout']} s ({resp.get('polls')} polls)"
        elif kind == "sleep":
            text = f"SLEEP   {float(req['seconds']):.2f} s"
        elif kind == "flush":
            text = f"FLUSH   drained={resp.get('drained')} ({ms})"
        elif kind in ("status", "wait_ready"):
            parts = []
            for sid, s in resp.get("shards", {}).items() if kind == "status" else ():
                if s.get("missing"):
                    parts.append(f"shard{sid}: DOWN")
                    continue
                models = ",".join(m if v.get("ready") else f"{m} {100 * (v.get('progress') or 0):.0f}%"
                                  for m, v in (s.get("models") or {}).items())
                parts.append(f"shard{sid}: {s.get('n_docs')} docs, pending {s.get('pending', 0)}"
                             + (f", lag p95 {s['lag_p95']} ms" if s.get("lag_p95") is not None else "")
                             + (f", [{models}]" if models else ""))
            if kind == "status":
                text = f"STATUS  active={resp.get('active_model')} | " + " | ".join(parts)
            else:
                ready = bool(ev["checks"] and ev["checks"][0]["pass"])
                text = (f"READY   {req.get('model')} {'ready on every shard' if ready else 'NOT ready'} after "
                        f"{resp.get('waited_ms', 0):.0f} ms ({resp.get('polls')} polls, progress {resp.get('progress')})")
        elif kind == "migrate":
            text = c(f"ADMIN   migrate {self.active_model or ''} -> {req['model']}: build started on every shard "
                     f"({ms}); queries still use the old model", "cyan")
        elif kind in ("cutover", "rollback"):
            text = c(f"ADMIN   {kind}: queries now use {resp.get('active_model')} "
                     f"(previous {resp.get('previous_model')}) ({ms})", "cyan")
        elif kind == "finalize":
            text = c(f"ADMIN   finalize: dropped {req['model']} index (dropped={resp.get('dropped')}) ({ms})", "cyan")
        elif kind == "kill_shard":
            text = c(f"CHAOS   killed {resp.get('killed')} (hard kill, like a crash)", "red")
        elif kind == "restart_shard":
            text = c(f"CHAOS   restarted {resp.get('restarted')} in {ms}: {resp.get('n_docs')} docs, "
                     f"{resp.get('replayed')} WAL records replayed", "cyan")
        elif kind == "golden_check":
            text = (f"GOLDEN  compared {len(resp.get('compared', []))} saved answers with the golden file"
                    + (f" (not in this run: {resp['only_in_golden']})" if resp.get("only_in_golden") else ""))
        elif kind == "parallel":
            text = f"JOIN    lanes {', '.join(req['lanes'])} ({ms})"
        elif kind == "signal":
            text = c(f"SIGNAL  '{req.get('name')}' (lanes waiting for it continue)", "magenta")
        elif kind == "wait_signal":
            text = f"WAIT    for signal '{req.get('name')}': got it after {resp.get('waited_ms', 0):.0f} ms"
        else:
            text = kind.upper()
        con.line(head + text, important=not ev["ok"] or kind in ADMIN_OPS)
        pad = " " * (len(head) + 8)
        for ch in ev["checks"]:
            mark = c(ok_s, "green") if ch["pass"] else (c(bad_s, "red") if ch["hard"] else c("~ soft, allowed", "yellow"))
            important = ch["hard"] and not ch["pass"]
            k = ch["kind"]
            if k in ("expect", "expect_absent"):
                want = "expected" if k == "expect" else "expected absent"
                got = (c(f"FOUND (#{ch['rank']})", "green" if ch["pass"] else "red") if ch["found"]
                       else c("MISSING", "green" if ch["pass"] else ("red" if ch["hard"] else "yellow")))
                # PITFALL: this is when the search RAN relative to the ack, NOT how long the doc took
                # to become visible (that is wait_visible's lag).
                since = (f", searched {fmt_ms(ch['since_ack_ms'])} after its ack"
                         if ch.get("since_ack_ms") is not None else "")
                con.line(f"{pad}{want} {ch['doc_id']}: {got}{since} {mark}", important=important)
            elif k == "expect_applied":
                con.line(f"{pad}expect_applied={ch['want']}: got applied={ch['got']} {mark}", important=important)
            elif k in ("same_as", "overlap", "golden"):
                if ch.get("detail"):
                    con.line(f"{pad}{k} {ch.get('ref')}: {ch['detail']} {mark}", important=important)
                    continue
                d = ch["diff"]
                what = ("identical to" if ch["pass"] else "DIFFERS from") if k != "overlap" else \
                    f"overlap {ch['overlap']:.2f} (min {ch['min']}) with"
                diff = "" if ch["pass"] and k != "overlap" else \
                    f" [first diff at #{d['first_diff_rank']}, lost {d['missing'][:4]}, new {d['extra'][:4]}]"
                models = f" ({ch.get('ref_model')} -> {ch.get('model')})" if ch.get("ref_model") != ch.get("model") else ""
                con.line(f"{pad}{'golden' if k == 'golden' else 'answer'} {what} {ch.get('ref')}{models}{diff} {mark}",
                         important=important)
            elif k == "ok":
                con.line(f"{pad}expect_ok: {'ok' if ch['pass'] else ch.get('detail')} {mark}", important=important)
            elif k in ("partial", "missing_shards"):
                con.line(f"{pad}expect {k}={ch['want']}: got {ch['got']} {mark}", important=important)
            elif k == "expect_error":
                if not ch["pass"]:
                    con.line(f"{pad}{c(bad_s + ' ' + ch['detail'], 'red')}", important=True)
            elif not ch["pass"]:
                con.line(f"{pad}{c(bad_s + ' ' + k + ' failed', 'red' if ch['hard'] else 'yellow')}", important=important)


def _short(s: str, n: int = 40) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- summary
def summarize(events: list[dict]) -> dict:
    """Totals over an event log (used for the console summary, events.jsonl and the replay)."""
    ends = [e for e in events if e.get("type") == "op_end"]
    starts = {e["id"]: e for e in events if e.get("type") == "op_start"}
    leaf = [e for e in ends if e["op"] not in ("label", "parallel")]
    checks = [c for e in ends for c in e.get("checks", [])]
    hard, soft = [c for c in checks if c.get("hard")], [c for c in checks if not c.get("hard")]
    writes = [e for e in ends if e["op"] in ("upsert", "delete") and not e.get("error")
              and "applied" in (e.get("resp") or {})]
    searches = [e for e in ends if e["op"] == "search" and not e.get("error") and "n" in (e.get("resp") or {})]

    def pct(xs):
        return {"p50": round(percentile(xs, 50), 2), "p95": round(percentile(xs, 95), 2)} if xs else None

    def tally(kind):
        cs = [c for c in checks if c["kind"] == kind]
        return {"passed": sum(c["pass"] for c in cs), "total": len(cs)}

    # Freshness: every "expect"ed doc in a search, bucketed by time since that doc's ack.
    fresh = [c for c in checks if c["kind"] == "expect" and c.get("since_ack_ms") is not None]
    early = [c for c in fresh if c["since_ack_ms"] < 1000]
    late = [c for c in fresh if c["since_ack_ms"] >= 1000]
    lags = [{"doc_id": starts[e["id"]]["req"]["doc_id"], "lag_ms": e["resp"].get("lag_ms"), "ok": e["ok"],
             "polls": e["resp"].get("polls"), "absent": starts[e["id"]]["req"].get("absent", False)}
            for e in ends if e["op"] == "wait_visible" and e["id"] in starts and not e.get("error")]
    admin = [{"t": e["t"], "op": e["op"], "ok": e["ok"], "model": starts.get(e["id"], {}).get("req", {}).get("model"),
              "shard": starts.get(e["id"], {}).get("req", {}).get("shard")}
             for e in ends if e["op"] in ADMIN_OPS]
    errors = [e for e in ends if e.get("error")]
    hard_failed = [c for c in hard if not c["pass"]]
    return {
        "ops": len(leaf), "errors": len(errors),
        "checks_passed": sum(c["pass"] for c in hard), "checks_failed": len(hard_failed),
        "soft_found": sum(c["pass"] for c in soft), "soft_missed": sum(not c["pass"] for c in soft),
        "writes": len(writes), "writes_applied": sum(bool(e["resp"].get("applied")) for e in writes),
        "write_ack_ms": pct([e["latency_ms"] for e in writes]),
        "searches": len(searches), "search_ms": pct([e["latency_ms"] for e in searches]),
        "partial_searches": sum(bool(e["resp"].get("partial")) for e in searches),
        "same_as": tally("same_as"), "overlap": tally("overlap"), "golden": tally("golden"),
        "visibility_lags": lags, "admin": admin,
        "fresh_under_1s": {"found": sum(c["found"] for c in early), "total": len(early)},
        "fresh_after_1s": {"found": sum(c["found"] for c in late), "total": len(late)},
        "duration_s": round(max((e.get("t", 0) for e in events), default=0.0), 3),
        "passed": not errors and not hard_failed,
    }


def print_summary(s: dict, meta: dict, console: Console) -> None:
    c = console.c
    lines = [c(f"=== SUMMARY  {meta.get('scenario')}  mode={meta.get('mode')}  shards={meta.get('n_shards')}  "
               f"({s['duration_s']:.2f} s) ===", "bold"),
             f"ops run          {s['ops']}  (errors {s['errors']})",
             f"expectations     {s['checks_passed']} passed, {s['checks_failed']} failed"
             + (f"; soft: {s['soft_found']} ok, {s['soft_missed']} missed (allowed)" if s['soft_found'] + s['soft_missed'] else "")]
    w, q = s["write_ack_ms"], s["search_ms"]
    lines.append(f"writes           {s['writes']} ({s['writes_applied']} applied)"
                 + (f"   ack p50 {w['p50']:.1f} ms  p95 {w['p95']:.1f} ms" if w else ""))
    lines.append(f"searches         {s['searches']}" + (f"   p50 {q['p50']:.1f} ms  p95 {q['p95']:.1f} ms" if q else "")
                 + (f"   ({s['partial_searches']} partial)" if s["partial_searches"] else ""))
    answers = [f"{k} {s[k]['passed']}/{s[k]['total']}" for k in ("same_as", "overlap", "golden") if s[k]["total"]]
    if answers:
        lines.append(f"answer checks    {', '.join(answers)}")
    fu, fa = s["fresh_under_1s"], s["fresh_after_1s"]
    if fu["total"] or fa["total"]:
        # Buckets by when the search RAN after the doc's ack (not a visibility lag).
        lines.append("freshness        " + "; ".join(
            f"searched {name} after ack: found {b['found']}/{b['total']}"
            for name, b in (("<1 s", fu), (">=1 s", fa)) if b["total"]))
    if s["visibility_lags"]:
        lags = ", ".join(f"{v['doc_id']} {'gone' if v['absent'] else 'visible'} "
                         + (f"after {v['lag_ms']:.1f} ms" if v["lag_ms"] is not None else ("n/a" if v["ok"] else "TIMEOUT"))
                         for v in s["visibility_lags"][:6])
        more = f" (+{len(s['visibility_lags']) - 6} more)" if len(s["visibility_lags"]) > 6 else ""
        lines.append(f"visibility lags  {lags}{more}   (the freshness number: ack -> first search that saw it; budget 1 s)")
    if s["admin"]:
        lines.append("admin timeline   " + " -> ".join(
            f"{a['op']}{'(' + str(a['model'] if a['model'] is not None else a['shard']) + ')' if (a['model'] or a['shard'] is not None) else ''}"
            f"@{a['t']:.2f}s{'' if a['ok'] else '!'}" for a in s["admin"]))
    lines.append("RESULT           " + (c("PASS", "green") if s["passed"] else c("FAIL", "red")))
    for ln in lines:
        console.line(ln, important=True)


# ----------------------------------------------------------------------------- orchestration
@dataclass
class ScenarioResult:
    ok: bool
    summary: dict
    out_dir: Path
    events_path: Path
    replay_path: Path
    saved: dict


def start_cluster(mode: str, tiny: bool, state_root: Path, router_port: int, base_port: int,
                  n_shards: int = 3, models: str = DEFAULT_MODEL, threads: int | None = None,
                  shard_timeout: float = 2.0, extra_env: dict | None = None):
    """Start a fresh local Cluster (real processes) in ``mode`` ("sync"/"async"). Returns it."""
    from src.service.cluster import Cluster

    env = {"APPLY_MODE": mode, **(extra_env or {})}
    if tiny:
        # The 24-doc fixture corpus + a hash "embedder": no model download, starts in seconds.
        # Every model key ("minilm", "bge-small") gets its own hash seed, so a migration still
        # changes the dense vectors like a real model change would.
        env.update({"FIQA_DATA_DIR": str(TINY_CORPUS), "EMBED_BACKEND": "hash",
                    "ARTIFACTS_DIR": str(state_root / "artifacts")})
    c = Cluster(n_shards, base_port=base_port, router_port=router_port, state_root=state_root, models=models,
                env=env, threads=threads or (1 if tiny else 4), timeout_s=shard_timeout, fresh=True)
    try:
        c.start(wait_s=180 if tiny else 900)
    except BaseException:
        # PITFALL (see cluster.py): a failed start() leaves already-spawned children running.
        c.stop()
        raise
    return c


def save_golden(path: Path, saved: dict, meta: dict) -> Path:
    """Write every ``save_as`` answer of this run to a golden JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ("scenario", "mode", "n_shards", "active_model", "cluster", "started_at", "run_id")
    path.write_text(json.dumps({"meta": {k: meta.get(k) for k in keep}, "answers": saved}, indent=1),
                    encoding="utf-8")
    return path


def run_scenario(scenario: str | Path, *, url: str | None = None, start: bool = False, mode: str | None = None,
                 tiny: bool | None = None, n_shards: int | None = None, models: str | None = None,
                 out_dir: Path | None = None, sample_ms: float = 150.0, quiet: bool = False,
                 color: bool | None = None, router_port: int = 19000, base_port: int = 19101,
                 keep_state: bool = False, extra_env: dict | None = None, golden_save: Path | None = None,
                 golden_check: Path | None = None, print_fn=print) -> ScenarioResult:
    """Run one scenario file and record it. The Python API behind the CLI (tests use it).

    Either pass ``url`` (an already running router) or ``start=True`` to launch a fresh cluster.
    Cluster settings come from the arguments if given, else from the file's ``config`` op, else
    defaults (sync, 3 shards, real corpus, minilm). With ``url`` the apply mode and shard count
    are read from the cluster. ``sample_ms=0`` disables the background status sampler.
    """
    path = Path(scenario)
    ops = parse_scenario(path.read_text(encoding="utf-8"))  # parse FIRST: no cluster for a typo
    cfg = {k: v for k, v in ops[0].items() if k not in ("op", "_line")} if ops and ops[0]["op"] == "config" else {}
    console = Console(color, quiet, print_fn)
    if not url and not start:
        raise ValueError("pass url=... or start=True")
    if not start and any(o["op"] in ("kill_shard", "restart_shard") for o in _walk(ops)):
        raise ScenarioError("this scenario kills/restarts shards: run it with --start-cluster")
    # Precedence: explicit argument (CLI flag) > config header > default.
    mode = mode or cfg.get("mode") or "sync"
    tiny = bool(cfg.get("tiny", False)) if tiny is None else tiny
    n_shards = int(n_shards or cfg.get("shards") or 3)
    models = models or cfg.get("models") or DEFAULT_MODEL
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = secrets.token_hex(3)  # 6 hex chars; {run} in the scenario
    state_root = STATE_ROOT / f"{path.stem}-{mode}-{stamp}-{run_id}"
    cluster = None
    if start:
        console.line(f"starting {'tiny' if tiny else 'REAL-corpus'} cluster ({mode} mode, {n_shards} shards, "
                     f"models {models}, router :{router_port}, shards :{base_port}+) ...", important=True)
        t = time.perf_counter()
        cluster = start_cluster(mode, tiny, state_root, router_port, base_port, n_shards, models,
                                extra_env=extra_env)
        url = cluster.router_url
        console.line(f"cluster up in {time.perf_counter() - t:.1f} s", important=True)
    try:
        with make_client(url, 30) as http:
            health = _call(http, "GET", "/health")
            status = compact_status(_call(http, "GET", "/admin/status"))
        shards = status["shards"]
        modes = {s.get("mode") for s in shards.values() if s.get("mode")}
        actual_mode = modes.pop() if len(modes) == 1 else ("mixed" if modes else "unknown")
        n = int(health.get("n_shards") or len(shards) or n_shards)
        if not start and cfg.get("shards") and int(cfg["shards"]) != n:
            console.line(console.c(f"WARNING: the scenario's config asks for {cfg['shards']} shards but the cluster "
                                   f"at {url} has {n}", "yellow"), important=True)
        out_dir = Path(out_dir or DEFAULT_OUT / f"{path.stem}-{actual_mode}-{stamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = Recorder(out_dir / "events.jsonl")
        meta = {"type": "meta", "t": 0.0, "scenario": path.stem, "file": str(path), "mode": actual_mode,
                "url": url, "n_shards": n, "run_id": run_id, "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                "cluster": ("started by runner, " + ("tiny fixture corpus + hash embedder" if tiny else "real FiQA corpus"))
                if start else "existing cluster", "active_model": health.get("active_model"), "models": models,
                "config": cfg, "description": cfg.get("description"), "lanes": lane_table(ops), "actors": resolve_actors(ops),
                "base_docs": sum(s.get("n_docs") or 0 for s in shards.values()), "sample_ms": sample_ms}
        rec.emit(meta)
        rec.emit({"type": "status", "t": 0.0, "src": "initial", **status})
        console.line(console.c(f"scenario {path.name}: mode={actual_mode}, {n} shards, model={health.get('active_model')}, "
                               f"url={url}, run id {run_id}", "cyan"), important=True)
        sampler = StatusSampler(url, rec, sample_ms / 1000) if sample_ms > 0 else None
        if sampler:
            sampler.start()
        runner = Runner(url, rec, console, n, run_id, cluster=cluster, active_model=health.get("active_model"))
        runner.display = {a["name"]: a["display"] for a in meta["actors"]}   # console shows personas ("user" -> "fox")
        try:
            runner.run_lane("main", ops, {})
            if golden_check:
                with make_client(url, 30) as http:
                    runner._leaf({"op": "golden_check", "_line": 0, "path": str(golden_check)}, "main", {}, http)
        finally:
            if sampler:
                sampler.stop()
        summary = summarize(rec.events)
        rec.emit({"type": "summary", "t": summary["duration_s"], **summary})
        rec.close()
        replay = write_replay(rec.events, out_dir / "replay.html")
        print_summary(summary, meta, console)
        if golden_save:
            console.line(f"golden saved     {save_golden(golden_save, runner.saved, meta)} "
                         f"({len(runner.saved)} answers)", important=True)
        console.line(f"events           {rec.path}", important=True)
        console.line(f"replay           {replay}", important=True)
        return ScenarioResult(summary["passed"], summary, out_dir, rec.path, replay, dict(runner.saved))
    finally:
        if cluster:
            cluster.stop()
            # Keep the shard/router logs with the run (handy when something failed), drop the WALs.
            if out_dir is not None and Path(out_dir).exists():
                logs = Path(out_dir) / "cluster_logs"
                logs.mkdir(exist_ok=True)
                for f in state_root.glob("*.log"):
                    shutil.copy(f, logs / f.name)
            if not keep_state:
                shutil.rmtree(state_root, ignore_errors=True)


def replay_only(events_path: Path, out: Path | None = None) -> Path:
    """Regenerate replay.html from an existing events.jsonl."""
    events = [json.loads(ln) for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not any(e.get("type") == "summary" for e in events):  # an interrupted run has no summary yet
        events.append({"type": "summary", **summarize(events)})
    return write_replay(events, out or events_path.with_name("replay.html"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="File format and examples: docs/scenarios.md")
    ap.add_argument("scenario", nargs="?", help="scenario .jsonl file")
    ap.add_argument("--url", help="router URL of an already running cluster, e.g. http://127.0.0.1:8000")
    ap.add_argument("--start-cluster", action="store_true", help="start a fresh local cluster for this run")
    ap.add_argument("--mode", choices=["sync", "async"], help="apply mode for --start-cluster (overrides config)")
    corpus = ap.add_mutually_exclusive_group()
    corpus.add_argument("--tiny", action="store_true", default=None,
                        help="with --start-cluster: 24-doc fixture corpus + hash embedder (overrides config)")
    corpus.add_argument("--real", dest="tiny", action="store_false", help="force the real FiQA corpus")
    ap.add_argument("--shards", type=int, help="number of shards for --start-cluster (overrides config)")
    ap.add_argument("--models", help="models every shard indexes at start, comma list (overrides config)")
    ap.add_argument("--router-port", type=int, default=19000)
    ap.add_argument("--base-port", type=int, default=19101, help="shard i listens on base-port + i")
    ap.add_argument("--async-max-lag", type=float, help="ASYNC_MAX_LAG_S for --start-cluster (default 0.5)")
    ap.add_argument("--async-batch-max", type=int, help="ASYNC_BATCH_MAX for --start-cluster (default 32)")
    ap.add_argument("--golden-save", type=Path, help="write every save_as answer to this JSON file")
    ap.add_argument("--golden-check", type=Path, help="compare save_as answers with this golden JSON file")
    ap.add_argument("--out", type=Path, help="output dir (default toy_runs/scenarios/<name>-<mode>-<time>/)")
    ap.add_argument("--sample-ms", type=float, default=150.0, help="status sampling period")
    ap.add_argument("--no-sample", action="store_true", help="do not sample /admin/status in the background")
    ap.add_argument("--keep-state", action="store_true", help="keep the started cluster's WAL/state dir")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only labels, admin ops, failures, summary")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--replay-only", type=Path, metavar="EVENTS_JSONL", help="just regenerate replay.html")
    ap.add_argument("--shard-of", nargs="+", metavar="DOC_ID",
                    help="print the owner shard of each doc id (crc32 %% --shards) and exit")
    a = ap.parse_args(argv)
    # PITFALL: a cp1252 Windows console raises UnicodeEncodeError on symbols; replace instead.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if a.shard_of:
        for d in a.shard_of:
            print(f"{d}\tshard {shard_of(d, a.shards or 3)}")
        return 0
    if a.replay_only:
        print(f"replay: {replay_only(a.replay_only, a.out / 'replay.html' if a.out else None)}")
        return 0
    if not a.scenario or bool(a.url) == a.start_cluster:
        ap.error("give a scenario file and exactly one of --url or --start-cluster")
    extra = {k: str(v) for k, v in (("ASYNC_MAX_LAG_S", a.async_max_lag), ("ASYNC_BATCH_MAX", a.async_batch_max))
             if v is not None}
    try:
        res = run_scenario(a.scenario, url=a.url, start=a.start_cluster, mode=a.mode, tiny=a.tiny,
                           n_shards=a.shards, models=a.models, out_dir=a.out,
                           sample_ms=0 if a.no_sample else a.sample_ms, quiet=a.quiet,
                           color=False if a.no_color else None, router_port=a.router_port, base_port=a.base_port,
                           keep_state=a.keep_state, extra_env=extra, golden_save=a.golden_save,
                           golden_check=a.golden_check)
    except ScenarioError as e:
        print(f"scenario error: {e}", file=sys.stderr)
        return 2
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
