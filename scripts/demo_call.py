"""Colourful, explicit HTTP calls for the hand-run demo guide (docs/guides/demo_security_a2a_memory.md).

Every call prints the REQUEST it sends (method, URL, headers, body) and the RESPONSE it gets back (status, time,
headers that matter, body), so nothing hides behind a helper. Colours say what happened:

    cyan     an admin command (anything under /admin, and the toggle / alloff shortcuts)
    green    success: 2xx, a COMPLETED task
    yellow   in between: 304 Not Modified, a task still SUBMITTED / WORKING, a guardrail finding in "log" mode
    red      refused or failed: 4xx / 5xx, a JSON-RPC error, a REJECTED / FAILED / AUTH_REQUIRED task, a blocked search
    magenta  the "=" lines: the one value you asked to see with --pick

USAGE (from the repo root, venv active; the guide wraps it as ``call``)

    python -m scripts.demo_call GET  /health
    python -m scripts.demo_call POST /search '{"query": "what is a roth ira?", "k": 3}' --as alice --pick 'len(d["results"])'
    python -m scripts.demo_call toggle identity=on tenant_filter=on          # admin: POST /admin/security
    python -m scripts.demo_call alloff                                      # admin: every toggle off
    python -m scripts.demo_call send http://127.0.0.1:8300 "what is a roth ira?" --as alice --skill retrieve
    python -m scripts.demo_call rpc  http://127.0.0.1:8300 ListTasks --as alice
    python -m scripts.demo_call burst 30 10 POST /search '{"query": "x", "k": 1}' --as bob   # rate-limit demo

    A path that starts with "/" goes to the router ($R, default http://127.0.0.1:8000); give a full URL for anything else.

WHO IS CALLING (--as)
    alice (tenant acme), bob (globex), admin (every tenant), none (no Authorization header, the default),
    forged (alice's token with its tenant edited to globex, signature unchanged), or any token with --token.
    Tokens are minted on the spot with scripts/mint_token.py, so they are signed with AUTH_SECRET like the router's.
    --ttl N gives the token N seconds to live, --sleep S waits S seconds after minting (an expired-token demo).

PICKING WHAT TO LOOK AT
    --pick EXPR prints one value under the response as an "=" line (repeat it for several). "label=EXPR" prints
    "= label: value". When an A2A task did not complete, its reason is printed in red under the status line, and
    picks that need the finished task say so instead of failing.

GETTING A VALUE OUT
    Everything human goes to stderr. --out EXPR prints one value to stdout, so a shell can keep it:
        CTX=$(python -m scripts.demo_call send ... --out 'd["result"]["task"]["contextId"]')
    In --pick / --out expressions, ``d`` is the parsed JSON body, ``status`` the HTTP status, ``headers`` the
    response headers, and ``collections`` / ``json`` are imported. A few names are ready-made picks:
        doc_ids   the doc ids of a /search response or of an A2A task's passages
        tenants   how many /search results came from each tenant
        state     an A2A task's state, and its status message if it has one
        memory    what memory did for an A2A task (metadata.memory)
        llm_saw   the exact text the LLM read in an "answer" task
        answer    the answer an "answer" task wrote
        toggles_on  the security toggles that are not off (on GET /admin/security)

    Long strings and long lists are shortened, and ``timings_ms``, ``history`` and ``artifactId`` are left out of
    responses; ``--full`` shows everything.
    NO_COLOR=1 turns colours off, FORCE_COLOR=1 keeps them when stderr is not a terminal.

    >>> classify(200, {"result": {"task": {"status": {"state": "TASK_STATE_WORKING"}}}})
    ('yellow', 'task TASK_STATE_WORKING')
    >>> classify(200, {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Task not found"}})
    ('red', 'JSON-RPC error -32001')
    >>> classify(200, {"results": [], "security": {"blocked": "topic_scope", "findings": []}})
    ('red', 'blocked by topic_scope')
    >>> classify(200, {"results": [{}], "security": {"findings": [{"rule": "topic_scope", "mode": "skipped"}]}})
    ('yellow', 'allowed, a guardrail could not run (see findings)')
    >>> render({"doc_id": "7", "text": "x" * 300}, full=False)[2][-18:]
    'x... (+140 chars)"'
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

ROUTER = os.environ.get("R", "http://127.0.0.1:8000")
TOGGLES = ["identity", "authz", "tenant_filter", "pii_redaction", "audit", "guard_harmful", "guard_injection",
           "guard_scope", "rate_limit", "web_fallback"]
_COLOR = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "dim": "2",
          "bold": "1"}
MAX_STR, MAX_LIST, ONE_LINE = 160, 6, 110
# Left out of printed responses unless --full: noise for a reader (history echoes the request back).
HIDDEN = ("timings_ms", "history", "artifactId")
_print_lock = threading.Lock()


def _task(d):
    r = d.get("result") if isinstance(d, dict) else None
    return (r.get("task") or r) if isinstance(r, dict) else {}


PRESETS = {
    "doc_ids": lambda d: [h["doc_id"] for h in d["results"]] if "results" in d
    else _task(d)["artifacts"][0]["parts"][0]["data"]["doc_ids"],
    "tenants": lambda d: dict(collections.Counter(h.get("tenant") for h in d["results"])),
    "state": lambda d: " | ".join([_task(d)["status"]["state"]] + [p["text"] for p in (
        _task(d)["status"].get("message") or {}).get("parts", []) if "text" in p]),
    "memory": lambda d: _task(d)["metadata"]["memory"],
    "llm_saw": lambda d: _task(d)["metadata"].get("llm_saw"),
    "answer": lambda d: next(a for a in _task(d)["artifacts"] if a["name"] == "answer")["parts"][0]["text"],
    "toggles_on": lambda d: [k for k, v in d.items() if isinstance(v, dict) and v.get("value") != "off"],
}


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(os.environ.get("FORCE_COLOR")) or sys.stderr.isatty()


def paint(text: str, *styles: str) -> str:
    if not styles or not use_color():
        return text
    return "\033[" + ";".join(_COLOR[s] for s in styles) + "m" + text + "\033[0m"


def say(line: str = "") -> None:
    with _print_lock:
        print(line, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ tokens
def token(who: str, ttl: int | None) -> str | None:
    if who == "none":
        return None
    from scripts.mint_token import token_for

    if who == "forged":
        h, p, s = token_for("alice", ttl_s=ttl or 3600).split(".")
        claims = json.loads(base64.urlsafe_b64decode(p + "=="))
        claims["tenant"] = "globex"   # the payload says globex, the signature still covers alice's acme payload
        p = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
        return f"{h}.{p}.{s}"
    return token_for(who, ttl_s=ttl or 3600)


def show_token(tok: str, who: str) -> str:
    try:
        claims = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "=="))
        about = f"{who}: sub={claims.get('sub')} tenant={claims.get('tenant')} role={claims.get('role')}"
    except Exception:  # noqa: BLE001  - a garbage --token still gets printed
        about = who
    return f"Bearer {tok[:12]}...{tok[-6:]}  ({about})"


# ------------------------------------------------------------------------------------------------ JSON display
def _short(obj, full: bool):
    if full:
        return obj
    if isinstance(obj, str) and len(obj) > MAX_STR:
        return obj[:MAX_STR] + f"... (+{len(obj) - MAX_STR} chars)"
    if isinstance(obj, list):
        items = [_short(x, full) for x in obj[:MAX_LIST]]
        return items + ([f"... {len(obj) - MAX_LIST} more"] if len(obj) > MAX_LIST else [])
    if isinstance(obj, dict):
        return {k: _short(v, full) for k, v in obj.items() if full or k not in HIDDEN}
    return obj


def render(obj, full: bool, indent: int = 0) -> list[str]:
    """JSON as lines: anything that fits on one line stays on one line, the rest is expanded."""
    obj = _short(obj, full) if indent == 0 else obj
    pad = "  " * indent
    flat = json.dumps(obj, ensure_ascii=False)
    if len(flat) + len(pad) <= ONE_LINE or not isinstance(obj, (dict, list)) or not obj:
        return [pad + flat]
    lines = [pad + ("{" if isinstance(obj, dict) else "[")]
    items = list(obj.items()) if isinstance(obj, dict) else [(None, v) for v in obj]
    for i, (k, v) in enumerate(items):
        sub = render(v, full, indent + 1)
        if k is not None:
            sub[0] = "  " * (indent + 1) + json.dumps(k) + ": " + sub[0].lstrip()
        if i < len(items) - 1:
            sub[-1] += ","
        lines += sub
    return lines + [pad + ("}" if isinstance(obj, dict) else "]")]


# ------------------------------------------------------------------------------------------------ verdict colour
def classify(status: int, d) -> tuple[str, str]:
    """(colour, short reason) for a response: HTTP status first, then what the body says."""
    if status >= 400:
        return "red", {401: "refused", 403: "refused", 429: "refused", 404: "not found"}.get(status, "error")
    if 300 <= status < 400:
        return "yellow", "not modified" if status == 304 else "redirect"
    if isinstance(d, dict):
        if "error" in d and "jsonrpc" in d:
            return "red", f"JSON-RPC error {d['error'].get('code')}"
        task = (d.get("result") or {}).get("task") if isinstance(d.get("result"), dict) else None
        task = task or (d.get("result") if isinstance(d.get("result"), dict) and "status" in d["result"] else None)
        if task and isinstance(task.get("status"), dict):
            state = task["status"].get("state", "")
            if state.endswith(("COMPLETED",)):
                return "green", f"task {state}"
            if state.endswith(("SUBMITTED", "WORKING")):
                return "yellow", f"task {state}"
            return "red", f"task {state}"
        sec = d.get("security") or {}
        if sec.get("blocked"):
            return "red", f"blocked by {sec['blocked']}"
        if any(f.get("mode") == "skipped" for f in sec.get("findings") or []):
            return "yellow", "allowed, a guardrail could not run (see findings)"
        if any(f.get("mode") == "log" for f in sec.get("findings") or []):
            return "yellow", "allowed, finding logged"
        if d.get("partial"):
            return "yellow", "partial results"
    return "green", "ok"


# ------------------------------------------------------------------------------------------------ one call
def call(method: str, url: str, body=None, who: str = "none", raw_token: str | None = None, ttl: int | None = None,
         sleep: float = 0.0, headers: list[str] | None = None, picks: list[str] | None = None, out: str | None = None,
         full: bool = False, admin: bool = False, quiet: bool = False, timeout: float = 120.0, view=None):
    import httpx

    url = ROUTER.rstrip("/") + url if url.startswith("/") else url
    tok = raw_token or token(who, ttl)
    if sleep:
        say(paint(f"  (token minted; waiting {sleep:g} s before sending)", "dim"))
        time.sleep(sleep)
    hdrs = {}
    for h in headers or []:
        k, _, v = h.partition(":")
        hdrs[k.strip()] = v.strip()
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    if body is not None:
        hdrs.setdefault("Content-Type", "application/json")
    admin = admin or "/admin/" in url
    if not quiet:
        say(paint(f"▶ {method} {url}", "cyan" if admin else "blue", "bold")
            + paint(f"   as {who if not raw_token else 'the given token'}", "dim"))
        for k, v in hdrs.items():
            say(paint(f"  {k}: {show_token(v[7:], who if not raw_token else 'token') if k == 'Authorization' else v}",
                      "dim"))
        if body is not None:
            for line in render(body, full):
                say("  " + line)
    # Build the client BEFORE starting the clock: creating one loads TLS certificates (~0.6 s on Windows),
    # which is not the service's time. trust_env=False: no proxy lookup from the environment.
    client = httpx.Client(trust_env=False, timeout=timeout)
    t0 = time.perf_counter()
    try:
        with client as c:
            r = c.request(method, url, headers=hdrs, json=body)
    except httpx.HTTPError as e:
        say(paint(f"◀ no response: {type(e).__name__}: {e}  (is the service at {url.split('/', 3)[2]} running?)",
                  "red", "bold"))
        sys.exit(2)
    ms = (time.perf_counter() - t0) * 1000
    try:
        d = r.json()
    except ValueError:
        d = r.text or None
    colour, why = classify(r.status_code, d)
    if not quiet:
        say(paint(f"◀ {r.status_code} {r.reason_phrase}", colour, "bold") + paint(f"   {why}, {ms:.0f} ms", colour))
        for k in ("etag", "www-authenticate", "retry-after"):
            if k in r.headers:
                say(paint(f"  {k}: {r.headers[k]}", "dim"))
        failed = task_failure(d)
        if failed:
            say(paint(f"  the task ended {failed}", "red", "bold"))
        if d not in (None, ""):
            for line in render(view(d) if view and not full else d, full):
                say("  " + line)
    env = {"d": d, "status": r.status_code, "headers": r.headers, "collections": collections, "json": json}
    failed = task_failure(d)
    for expr in picks or []:
        label, val = pick(expr, env)
        if failed and isinstance(val, str) and val.startswith("(could not evaluate"):
            val = f"(not available: the task ended {failed.split(':')[0]})"
        text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        head = f"= {label}: " if label else "= "
        if "\n" in text:   # multi-line text (the prompt an LLM read): print it as text, indented
            say(paint(head.rstrip(), "magenta", "bold"))
            for line in text.splitlines():
                say(paint("  | " + line, "magenta"))
        else:
            say(paint(head + text, "magenta", "bold"))
    if not quiet:
        say()
    if out:
        print(pick(out, env)[1])
    return r.status_code, d


def task_failure(d) -> str | None:
    """'TASK_STATE_FAILED: <reason>' when the response is an A2A task that ended without completing, else None.

    >>> task_failure({"result": {"task": {"status": {"state": "TASK_STATE_FAILED",
    ...     "message": {"parts": [{"text": "RuntimeError: no LLM"}]}}}}})
    'TASK_STATE_FAILED: RuntimeError: no LLM'
    >>> task_failure({"result": {"task": {"status": {"state": "TASK_STATE_COMPLETED"}}}}) is None
    True
    """
    status = _task(d).get("status") if isinstance(d, dict) else None
    state = (status or {}).get("state", "")
    if not state or state.endswith(("COMPLETED", "SUBMITTED", "WORKING")):
        return None
    why = " ".join(p.get("text", "") for p in (status.get("message") or {}).get("parts", []) if p.get("text"))
    return f"{state}: {why}" if why else state


def pick(expr: str, env: dict):
    """(label, value) of a --pick: a preset name, "label=expression", or a bare expression.

    >>> pick("n=len(d)", {"d": [1, 2]}), pick("d[0] == 1", {"d": [1]}), pick("doc_ids", {"d": {"results": []}})
    (('n', 2), (None, True), ('doc_ids', []))
    """
    m = re.match(r"^([A-Za-z_]\w*)=(?!=)(.+)$", expr, re.S)
    label, src = (m.group(1), m.group(2)) if m else (expr if expr in PRESETS else None, expr)
    try:
        preset = PRESETS.get(src)
        return label, preset(env["d"]) if preset else eval(src, env)  # noqa: S307  - typed by the user
    except Exception as e:  # noqa: BLE001
        return label, f"(could not evaluate: {type(e).__name__}: {e})"


# ------------------------------------------------------------------------------------------------ shortcuts
def a2a_message(text: str | None, skill: str | None, k: int | None, context: str | None, meta: str | None,
                data: str | None) -> dict:
    parts = ([{"text": text}] if text else []) + ([{"data": json.loads(data)}] if data else [])
    md = {**({"skill": skill} if skill else {}), **({"k": k} if k else {}), **json.loads(meta or "{}")}
    m = {"messageId": uuid.uuid4().hex[:12], "role": "ROLE_USER", "parts": parts}
    if md:
        m["metadata"] = md
    if context:
        m["contextId"] = context
    return m


def burst(n: int, parallel: int, method: str, url: str, body, tok: str | None, who: str) -> None:
    """n identical calls, ``parallel`` at a time; prints the count per status and the first refused response."""
    import httpx

    url = ROUTER.rstrip("/") + url if url.startswith("/") else url
    hdrs = {"Authorization": f"Bearer {tok}"} if tok else {}
    say(paint(f"▶ {n} x {method} {url}, {parallel} at a time", "blue", "bold") + paint(f"   as {who}", "dim"))
    if body is not None:
        say("  " + render(body, False)[0])
    with httpx.Client(timeout=30, trust_env=False) as c, ThreadPoolExecutor(parallel) as pool:
        rs = list(pool.map(lambda _: c.request(method, url, headers=hdrs, json=body), range(n)))
    counts = collections.Counter(r.status_code for r in rs)
    for code, cnt in sorted(counts.items()):
        colour = "green" if code < 300 else "red"
        say(paint(f"◀ {cnt:3d} x {code}", colour, "bold"))
    first = next((r for r in rs if r.status_code >= 400), None)
    if first is not None:
        say(paint(f"  first refused response ({first.status_code}):", "dim"))
        for line in render(first.json(), False):
            say("  " + line)
    say()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m scripts.demo_call", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", help="GET | POST | DELETE | toggle | alloff | send | rpc | burst")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--as", dest="who", default="none", help="alice | bob | admin | forged | none")
    ap.add_argument("--token", help="send this exact bearer token instead")
    ap.add_argument("--ttl", type=int, help="token lifetime in seconds")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait after minting the token")
    ap.add_argument("-H", "--header", action="append", default=[], help="extra header 'Name: value'")
    ap.add_argument("--pick", action="append", default=[], help="Python expression on d to print as an = line")
    ap.add_argument("--out", help="Python expression on d to print to stdout (for $(...))")
    ap.add_argument("--full", action="store_true", help="do not shorten strings and lists")
    ap.add_argument("--quiet", action="store_true", help="print only the --pick lines")
    ap.add_argument("--skill"), ap.add_argument("--k", type=int), ap.add_argument("--context")
    ap.add_argument("--meta", help="extra message metadata, JSON (e.g. '{\"memory\": false}')")
    ap.add_argument("--data", help="a data part, JSON (e.g. '{\"remember\": \"I am from the UK\"}')")
    ap.add_argument("--immediate", action="store_true", help="configuration.returnImmediately")
    ap.add_argument("--no-version", action="store_true", help="leave out the A2A-Version header")
    a = ap.parse_args(argv)

    def body_arg(s: str):
        try:
            return json.loads(s)
        except ValueError as e:
            say(paint(f"the body is not valid JSON ({e}):", "red", "bold"))
            say("  " + s)
            say(paint("  (keys and strings need double quotes; wrap the whole body in single quotes in the shell)", "dim"))
            sys.exit(2)
    common = dict(who=a.who, raw_token=a.token, ttl=a.ttl, sleep=a.sleep, headers=a.header, picks=a.pick,
                  out=a.out, full=a.full, quiet=a.quiet)
    what = a.what.lower()
    if what in ("get", "post", "delete", "put"):
        url, *rest = a.args
        body = body_arg(rest[0]) if rest else None
        call(what.upper(), url, body, **common)
    elif what in ("toggle", "alloff"):
        changes = {t: "off" for t in TOGGLES} if what == "alloff" else dict(x.split("=", 1) for x in a.args)
        on = {k: v for k, v in changes.items() if v != "off"}
        picks = common.pop("picks") or [f"'now on: {', '.join(f'{k}={v}' for k, v in on.items()) or 'nothing'}'"]
        compact = lambda d: {"changed": d.get("changed"),  # noqa: E731
                             "now": " ".join(f"{k}={v['value']}" for k, v in (d.get("now") or {}).items())}
        call("POST", "/admin/security", changes, **{**common, "who": "admin", "picks": picks}, admin=True,
             view=compact if isinstance(changes, dict) else None)
    elif what == "send":
        agent, *text = a.args
        params = {"message": a2a_message(" ".join(text) or None, a.skill, a.k, a.context, a.meta, a.data)}
        if a.immediate:
            params["configuration"] = {"returnImmediately": True}
        hdr = [] if a.no_version else ["A2A-Version: 1.0"]
        call("POST", agent.rstrip("/") + "/a2a", {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": params},
             **{**common, "headers": hdr + a.header})
    elif what == "rpc":
        agent, method, *rest = a.args
        hdr = [] if a.no_version else ["A2A-Version: 1.0"]
        call("POST", agent.rstrip("/") + "/a2a",
             {"jsonrpc": "2.0", "id": 1, "method": method, "params": body_arg(rest[0]) if rest else {}},
             **{**common, "headers": hdr + a.header})
    elif what == "burst":
        n, par, method, url, *rest = a.args
        burst(int(n), int(par), method.upper(), url, body_arg(rest[0]) if rest else None,
              a.token or token(a.who, a.ttl), a.who)
    else:
        ap.error(f"unknown command {a.what!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
