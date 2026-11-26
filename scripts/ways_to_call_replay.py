"""Four ways to call the router, recorded: the same question through Swagger, demo_call, search.py and ask.py.

    docker compose up -d                                        # the router on :8000
    python scripts/ways_to_call_replay.py --confirm-spend       # ask.py's agent makes a few Azure calls (~$0.004)
    python scripts/ways_to_call_replay.py --browser-cmd "node swagger_try.mjs {url} {png} {body}" --confirm-spend
    python scripts/ways_to_call_replay.py --render-only

Writes replays/data/ways_to_call.json and replays/ways_to_call.html (a nav on the left, one page per way, then a
comparison). Open the page straight from disk.

THE FOUR WAYS
    Router docs   the Swagger page the router serves at /docs: open POST /search, Try it out, Execute.
                  Nothing to install; a browser is the client (scripts/api_docs.py builds the page).
    SDK           scripts/demo_call.py, the small Python tool the demo guide uses. It mints tokens, prints the
                  whole exchange in colour and picks values out of the answer (docs/guides/demo_security_a2a_memory.md).
    CLI search.py the command line the assignment asks for in A1: ranked passages with scores, one POST /search.
    CLI ask.py    one question through an LLM agent. The agent checks the guardrails, then searches as many times
                  as it decides to; every one of those calls is a plain HTTP call to the same router.

HOW THE CALLS ARE CAPTURED
    The script starts a small logging proxy on 127.0.0.1:8099 and points every way at it instead of :8000. The
    proxy forwards each request unchanged and keeps a copy of the request and the response, so the page shows what
    actually reached the router, byte for byte (JSON bodies shortened for display). The Swagger page is loaded
    through the proxy too (its requests are relative), driven by the command given with --browser-cmd; without
    one, the Swagger page shows the screenshot from docs/assets/screenshots/swagger_search.png and no capture.

    >>> shorten({"results": [{"text": "x" * 300}]})["results"][0]["text"][-12:]
    '(+100 chars)'
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_style import NAV_CSS  # noqa: E402

OUT = ROOT / "replays"
DATA = OUT / "data" / "ways_to_call.json"
SHOT = OUT / "data" / "ways_swagger.png"
PROXY_PORT = 8099
QUESTION = "what is short selling?"     # the example query in the assignment's A1
SKIP = ("/docs", "/openapi.json", "/favicon.ico")
COLOR = "#475569"                       # not one of the five categories: this is how you use the system


def shorten(o, limit: int = 200):
    """Long strings cut to ``limit`` characters (with a note of how much was cut); lists and dicts recursed."""
    if isinstance(o, str):
        return o if len(o) <= limit else o[:limit] + f"... (+{len(o) - limit} chars)"
    if isinstance(o, list):
        return [shorten(x, limit) for x in o]
    if isinstance(o, dict):
        return {k: shorten(v, limit) for k, v in o.items() if k != "timings_ms"}
    return o


class Proxy:
    """A forwarding HTTP proxy that records every exchange (``self.log``) except the Swagger page's own assets."""

    def __init__(self, target: str, port: int = PROXY_PORT):
        import httpx

        self.target, self.log, self.lock = target.rstrip("/"), [], threading.Lock()
        client = httpx.Client(timeout=60, trust_env=False)
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _do(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length",
                                                                                     "accept-encoding", "connection")}
                t = time.perf_counter()
                r = client.request(self.command, proxy.target + self.path, content=raw, headers=hdrs)
                ms = round((time.perf_counter() - t) * 1e3, 1)
                body = r.content
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if self.path.split("?")[0] in SKIP or self.path.startswith("/static"):
                    return
                shown = {k: v for k, v in hdrs.items() if k.lower() in ("content-type", "authorization", "user-agent",
                                                                        "a2a-version", "accept")}
                if "Authorization" in shown or "authorization" in shown:
                    key = "Authorization" if "Authorization" in shown else "authorization"
                    shown[key] = shown[key][:20] + "..."
                with proxy.lock:
                    proxy.log.append({"method": self.command, "path": self.path, "headers": shown,
                                      "body": _json(raw), "status": r.status_code, "ms": ms,
                                      "response": shorten(_json(body))})

            do_GET = do_POST = do_DELETE = do_PUT = _do

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.url = f"http://127.0.0.1:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self) -> list[dict]:
        with self.lock:
            out, self.log = self.log, []
        return out


def _json(raw: bytes):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")[:300]


def run(cmd: list[str], env: dict | None = None) -> str:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}, timeout=300)
    return (p.stdout + p.stderr).strip()


def top_ids(calls: list[dict], n: int = 3) -> list[str]:
    """The top doc ids of the first POST /search answer in ``calls``."""
    for c in calls:
        if c["method"] == "POST" and c["path"].startswith("/search") and isinstance(c["response"], dict):
            return [h["doc_id"] for h in c["response"].get("results", [])][:n]
    return []


def record(router: str, confirm_spend: bool, browser_cmd: str | None) -> dict:
    proxy = Proxy(router)
    py = sys.executable
    body = {"query": QUESTION, "k": 3}
    ways = []

    typed = "Open /docs in a browser, expand POST /search, Try it out, paste the body, Execute"
    shot, swagger_calls = None, []
    if browser_cmd:
        cmd = browser_cmd.format(url=proxy.url, png=str(SHOT), body=shlex.quote(json.dumps(body)))
        out = run(shlex.split(cmd, posix=os.name != "nt"))
        swagger_calls = proxy.take()
        shot = SHOT if SHOT.exists() else None
        print("swagger:", out.splitlines()[-1] if out else "", f"({len(swagger_calls)} calls)")
    ways.append({"id": "swagger", "kind": "Router docs", "name": "Swagger at /docs", "typed": typed,
                 "body_typed": json.dumps(body), "output": None, "calls": swagger_calls,
                 "shot": "data/ways_swagger.png" if shot else "../docs/assets/screenshots/swagger_search.png",
                 "about": "The router describes its own API (OpenAPI) and serves Swagger UI on top of it. Every "
                          "endpoint has named example bodies; Try it out sends a real request from the browser and "
                          "shows the curl command, the response and how long it took. Good for exploring: nothing "
                          "to install, and the examples are checked by tests/test_api_docs.py."})

    cmd = [py, "-m", "scripts.demo_call", "POST", "/search", json.dumps(body), "--pick", "doc_ids"]
    out = run(cmd, {"R": proxy.url})
    ways.append({"id": "demo_call", "kind": "SDK", "name": "demo_call (Python)",
                 "typed": f"python -m scripts.demo_call POST /search '{json.dumps(body)}' --pick doc_ids",
                 "output": out.replace(proxy.url, "http://127.0.0.1:8000"), "calls": proxy.take(),
                 "about": "A thin Python layer over HTTP for scripted, repeatable calls: it signs tokens for alice, "
                          "bob or admin (--as), prints the request it sent and the response it got, colours them by "
                          "outcome, and --pick pulls one value out of the answer. The demo guide is written in it. "
                          "In code, the same call is http_search_fn(url, token) in src/agent/tools.py."})

    cmd = [py, "src/search.py", "--query", QUESTION, "--top-k", "3", "--url", proxy.url]
    out = run(cmd)
    ways.append({"id": "search_py", "kind": "CLI", "name": "search.py",
                 "typed": f'python src/search.py --query "{QUESTION}" --top-k 3', "output": out, "calls": proxy.take(),
                 "about": "The command line the assignment asks for (A1): one POST /search with the shipped fusion "
                          "settings spelled out, then rank, score, doc id and a snippet per passage. It never "
                          "touches an index; all the searching happens in the router and its shards."})

    if confirm_spend:
        cmd = [py, "scripts/ask.py", QUESTION, "--url", proxy.url, "--strategy", "single", "--backend", "azure",
               "--confirm-spend", "-k", "3"]
        out = run(cmd)
        calls = proxy.take()
    else:
        out, calls = "(not recorded: run with --confirm-spend)", []
    ways.append({"id": "ask_py", "kind": "CLI", "name": "ask.py (agent + LLM)",
                 "typed": f'python scripts/ask.py "{QUESTION}" --url http://127.0.0.1:8000 --strategy single '
                          "--backend azure --confirm-spend -k 3",
                 "output": out.replace(proxy.url, "http://127.0.0.1:8000"), "calls": calls,
                 "about": "One question through the single agent. Before its LLM sees anything it asks the router's "
                          "guardrails (POST /guardrails/check); then it searches, reads the snippets and may rewrite "
                          "the query and search again, up to 3 steps. The LLM never talks to the router: the agent "
                          "code does, with ordinary HTTP calls like the other three ways."})
    proxy.server.shutdown()
    for w in ways:
        w["top3"] = top_ids(w["calls"])
    return {"recorded": time.strftime("%Y-%m-%d %H:%M"), "router": router, "question": QUESTION, "ways": ways}


CSS = """
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7;color:#1f2328}
header{background:#1e2a3a;color:#fff;padding:10px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:2}
header h1{font-size:19px;margin:0} header button{margin-left:6px;padding:6px 14px;border:0;border-radius:4px;background:#3b82f6;color:#fff;cursor:pointer}
.wrap{display:flex} main{flex:1;padding:16px 24px;min-width:0}
.card{background:#fff;border:1px solid #ddd;border-left:6px solid #475569;border-radius:6px;padding:12px 16px;margin-bottom:12px}
.card h2{margin:0 0 4px;font-size:18px} .card p{margin:4px 0;font-size:14px;line-height:1.5;color:#333}
.kind{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:10px;background:#e2e8f0;color:#334155;margin-left:8px}
h3{font-size:14px;margin:14px 0 6px;color:#334155}
pre{background:#0f172a;color:#e2e8f0;padding:10px 12px;border-radius:6px;font-size:12px;white-space:pre-wrap;word-break:break-word;margin:0;max-height:380px;overflow:auto}
pre.typed{background:#111827;color:#a7f3d0;font-size:13px;max-height:none}
.call{background:#fff;border:1px solid #ddd;border-radius:6px;margin-bottom:10px;overflow:hidden}
.call .h{padding:7px 12px;font-family:Consolas,monospace;font-size:13px;background:#f8fafc;border-bottom:1px solid #eee}
.call .h b{color:#1d4ed8} .ok{color:#15803d;font-weight:700} .no{color:#dc2626;font-weight:700}
.call .cols{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}
.call .cols div small{display:block;font-size:11px;color:#64748b;margin-bottom:3px}
.shotbox{max-height:560px;overflow:auto;border:1px solid #ddd;border-radius:6px;background:#fff} img.shot{width:100%;display:block}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff} td,th{border:1px solid #e5e7eb;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f1f5f9} .same{background:#dcfce7;color:#166534;font-weight:700;border-radius:8px;padding:1px 8px}
.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0 4px}
.flow span{background:#e2e8f0;border-radius:6px;padding:6px 10px;font-size:13px} .flow i{font-style:normal;color:#64748b}
"""


def render(rec: dict) -> str:
    ways = rec["ways"]
    for w in ways:
        if w.get("shot") and w["shot"].startswith("data/") and (OUT / w["shot"]).exists():
            w["img"] = "data:image/png;base64," + base64.b64encode((OUT / w["shot"]).read_bytes()).decode()
        else:
            w["img"] = w.get("shot")
    items = [f"{w['kind']}: {w['name']}" for w in ways] + ["Compare the four"]
    nav = (f'<div class="cat" style="background:#e2e8f0;color:{COLOR}">Ways to call the router'
           f'<small>one router, four clients, the same HTTP underneath</small></div><ol>' +
           "".join(f'<li id="l{i}" onclick="go({i})" style="border-left-color:{COLOR}"><b>{i + 1}</b>{html.escape(t)}</li>'
                   for i, t in enumerate(items)) + "</ol>")
    data = json.dumps(rec, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Ways to call the router</title>
<style>{CSS}{NAV_CSS}</style></head><body>
<header><h1 id="h">Ways to call the router</h1><div><button onclick="go(cur-1)">&larr; prev</button><button onclick="go(cur+1)">next &rarr;</button></div></header>
<div class="wrap"><aside>{nav}</aside><main id="main"></main></div>
<script>
const R = {data};
const W = R.ways, N = W.length + 1;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
const js = o => o == null ? "(empty)" : esc(typeof o === "string" ? o : JSON.stringify(o, null, 1));
let cur = 0;
function call(c, i) {{
  return `<div class="call"><div class="h">${{i + 1}}. <b>${{esc(c.method)}} ${{esc(c.path)}}</b> &rarr; <span class="${{c.status < 300 ? "ok" : "no"}}">${{c.status}}</span> in ${{c.ms}} ms</div>` +
    `<div class="cols"><div><small>request headers and body, as the router received them</small><pre>${{js(c.headers)}}\\n\\n${{js(c.body)}}</pre></div>` +
    `<div><small>response body (long texts shortened)</small><pre>${{js(c.response)}}</pre></div></div></div>`;
}}
function way(w) {{
  let h = `<div class="card"><h2>${{esc(w.name)}}<span class="kind">${{esc(w.kind)}}</span></h2><p>${{esc(w.about)}}</p>` +
    `<div class="flow"><span>${{esc(w.name)}}</span><i>&rarr; HTTP &rarr;</i><span>router :8000</span><i>&rarr; fan-out &rarr;</i><span>3 shards</span></div></div>`;
  h += `<h3>What you type</h3><pre class="typed">${{esc(w.typed)}}${{w.id === "swagger" ? "\\n\\nbody: " + esc(w.body_typed) : ""}}</pre>`;
  if (w.img) h += `<h3>What you see (scroll: the request body, Execute, then the curl command and the response)</h3><div class="shotbox"><img class="shot" src="${{w.img}}" alt="Swagger Try it out"></div>`;
  if (w.output) h += `<h3>What you see</h3><pre>${{esc(w.output)}}</pre>`;
  h += `<h3>What reached the router (${{w.calls.length}} HTTP call${{w.calls.length === 1 ? "" : "s"}}, captured by a logging proxy)</h3>` +
    (w.calls.length ? w.calls.map(call).join("") : "<p>(none captured in this recording)</p>");
  return h;
}}
function compare() {{
  const ref = (W.find(w => w.top3.length) || {{top3: []}}).top3.join(",");
  const rows = W.map(w => {{
    const s = w.calls.filter(c => c.path.startsWith("/search")).length, g = w.calls.filter(c => c.path.startsWith("/guardrails")).length;
    const same = w.top3.length && w.top3.join(",") === ref;
    return `<tr><td><b>${{esc(w.name)}}</b><br><small>${{esc(w.kind)}}</small></td><td>${{w.calls.length}}</td><td>${{s}} search${{s === 1 ? "" : "es"}}${{g ? `, ${{g}} guardrail check` : ""}}</td>` +
      `<td><code>${{esc(w.top3.join(", "))}}</code> ${{same ? '<span class="same">same</span>' : ""}}</td><td>${{w.calls.map(c => c.ms).join(" / ")}} ms</td></tr>`;
  }}).join("");
  return `<div class="card"><h2>One router, four ways in</h2><p>Every way ends as an HTTP request to the same router, so the
  same question gets the same passages: the first search's top 3 are identical. What differs is who builds the request
  and what is done with the answer: a person in a browser, a script, a terminal, or an LLM agent that decides to search
  more than once.</p></div><table><tr><th>way</th><th>HTTP calls</th><th>what they were</th><th>top 3 of the first search</th><th>round trip through the proxy</th></tr>${{rows}}</table>
  <p style="font-size:12px;color:#666">Question: "${{esc(R.question)}}". Recorded ${{esc(R.recorded)}} against ${{esc(R.router)}}, every call through a
  logging proxy on :8099 that forwards it unchanged.</p>`;
}}
function go(i) {{
  if (i < 0 || i >= N) return;
  cur = i;
  document.querySelectorAll("aside li").forEach((l, k) => l.classList.toggle("cur", k === i));
  document.getElementById("h").textContent = `Ways to call the router · ${{i + 1}} of ${{N}}`;
  document.getElementById("main").innerHTML = i < W.length ? way(W[i]) : compare();
  location.hash = i + 1; window.scrollTo(0, 0);
}}
document.addEventListener("keydown", e => {{ if (e.key === "ArrowRight") go(cur + 1); if (e.key === "ArrowLeft") go(cur - 1); }});
window.addEventListener("hashchange", () => {{ const n = (parseInt(location.hash.slice(1)) || 1) - 1; if (n !== cur) go(n); }});
go(Math.max(0, (parseInt(location.hash.slice(1)) || 1) - 1));
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--router", default="http://127.0.0.1:8000")
    ap.add_argument("--confirm-spend", action="store_true", help="record ask.py with the Azure LLM (a few cents)")
    ap.add_argument("--browser-cmd", help="drives Swagger; {url}, {png} and {body} are filled in")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        rec = json.loads(DATA.read_text(encoding="utf-8"))
    else:
        rec = record(a.router, a.confirm_spend, a.browser_cmd)
        DATA.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "ways_to_call.html").write_text(render(rec), encoding="utf-8")
    print(f"wrote {(OUT / 'ways_to_call.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
