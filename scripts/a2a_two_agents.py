"""Two live A2A agents talking, recorded message by message and replayed as a sequence diagram.

    docker compose up -d                 # the router on :8000
    python scripts/a2a_two_agents.py     # starts the FiQA agent (:8300) and the coordinator (:8310) in this process
    python scripts/a2a_two_agents.py --render-only

Writes scenarios/SimpleScenarios/results/A2AReplay/messages.json and replay.html. Open replay.html in a browser and
press Play: every HTTP message between alice, bob, the coordinator, the FiQA agent and the router appears as an
arrow, in the order it happened; click one to read its JSON.

THE SCENES
    1. Discovery      alice reads the coordinator's card; the coordinator reads the FiQA agent's card.
    2. Delegation     alice asks a compound question; the coordinator splits it and sends each part to the FiQA
                      agent as its own task, with alice's token; the FiQA agent searches the router as alice.
    3. Tasks are private   bob asks the coordinator for alice's task: "Task not found". His task list is empty.
    4. Tenants        bob asks the same question; with the router's tenant_filter on, his passages are globex's
                      and alice's are acme's, although both went through the same two agents.
    5. Memory         alice tells the FiQA agent she lives in Canada; her next delegated parts are searched with
                      "Canada" added, bob's are not. alice deletes her memory at the end.

The router's identity and tenant_filter toggles are switched on for the run and restored afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scripts.mint_token import token_for  # noqa: E402
from src.a2a.client import A2AClient, A2AError, doc_ids  # noqa: E402
from src.a2a.coordinator import create_coordinator_app  # noqa: E402
from src.a2a.server import create_app  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.agent.tools import http_search_fn  # noqa: E402

OUT = ROOT / "scenarios" / "SimpleScenarios" / "results" / "A2AReplay"
PARTIES = ["alice", "bob", "coordinator :8310", "FiQA agent :8300", "router :8000"]
Q = "what is a roth ira, and how do I buy my first home?"
Q2 = "how are capital gains taxed, and what retirement account should I open?"


def shrink(o, depth: int = 0):
    """Keep the shape of a JSON message, cut long strings and lists so the replay stays readable."""
    if isinstance(o, str):
        return o if len(o) <= 140 else o[:140] + "..."
    if isinstance(o, list):
        return [shrink(x, depth + 1) for x in o[:5]] + ([f"... {len(o) - 5} more"] if len(o) > 5 else [])
    if isinstance(o, dict):
        return {k: shrink(v, depth + 1) for k, v in o.items()}
    return o


class Recorder:
    def __init__(self):
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()

    def now(self) -> float:
        return round((time.perf_counter() - self.t0) * 1e3, 1)

    def add(self, **e) -> int:
        with self.lock:
            e["seq"] = len(self.events)
            self.events.append(e)
            return e["seq"]

    def scene(self, title: str, text: str) -> None:
        self.add(kind="scene", title=title, text=text, t=self.now())

    def client(self, src: str, dst: str) -> httpx.Client:
        """An httpx client whose every request and response is recorded as two arrows."""
        def on_request(req: httpx.Request):
            body = json.loads(req.content) if req.content else None
            req.extensions["seq"] = self.add(kind="request", src=src, dst=dst, t=self.now(),
                                             label=label_of(req.method, req.url.path, body), body=shrink(body),
                                             http=f"{req.method} {req.url.path}")

        def on_response(resp: httpx.Response):
            resp.read()
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:200]
            self.add(kind="response", src=dst, dst=src, t=self.now(), status=resp.status_code,
                     label=answer_of(resp.status_code, body), body=shrink(body), reply_to=resp.request.extensions.get("seq"))

        return httpx.Client(timeout=60, event_hooks={"request": [on_request], "response": [on_response]})

    def search_fn(self, router: str, token: str | None, who: str):
        """The FiQA agent's search, recorded as FiQA agent -> router arrows."""
        raw = http_search_fn(router, token)

        def fn(query: str, strategy: str, k: int) -> list[dict]:
            s = self.add(kind="request", src="FiQA agent :8300", dst="router :8000", t=self.now(),
                         label="search " + (f'"{query}"' if len(query) <= 40 else f'"...{query[-38:]}"'), http="POST /search",
                         body={"query": query, "k": k, "strategy": strategy, "as": who})
            # FiQA corpus ids are numbers; anything else was written through /upsert by someone using the
            # stack, so the recording shows that it is there but not what it says.
            hits = [h if str(h["doc_id"]).isdigit() else {**h, "doc_id": "user-written-doc",
                    "text": "(a document written through /upsert)"} for h in raw(query, strategy, k)]
            tenants = sorted({h.get("tenant") or "?" for h in hits})
            self.add(kind="response", src="router :8000", dst="FiQA agent :8300", t=self.now(), status=200,
                     label=f"{len(hits)} hits, tenant {', '.join(tenants)}", reply_to=s,
                     body={"results": [{"doc_id": h["doc_id"], "tenant": h.get("tenant"),
                                        "text": h.get("text", "")[:100]} for h in hits[:5]]})
            return hits

        return fn


def label_of(method: str, path: str, body) -> str:
    if path.endswith("agent-card.json"):
        return "GET agent card"
    if isinstance(body, dict) and body.get("jsonrpc"):
        p = body.get("params") or {}
        if body["method"] == "SendMessage":
            parts = p.get("message", {}).get("parts", [])
            text = next((x["text"] for x in parts if "text" in x), "")
            return f'SendMessage "{text[:60]}"'
        if "id" in p:
            return f"{body['method']} {p['id'][:8]}..."
        return body["method"]
    if path == "/memory":
        return f"{method} /memory" + (f' "{body.get("text")}"' if isinstance(body, dict) and body.get("text") else "")
    return f"{method} {path}"


def answer_of(status: int, body) -> str:
    if status != 200:
        return f"HTTP {status}"
    if isinstance(body, dict):
        if "error" in body:
            return f"error {body['error'].get('code')}: {body['error'].get('message')}"
        res = body.get("result")
        if isinstance(res, dict):
            task = res.get("task", res if "status" in res else None)
            if task:
                return f"Task {task['status']['state'].replace('TASK_STATE_', '')}, {len(doc_ids(task))} docs"
            if "tasks" in res:
                return f"{len(res['tasks'])} tasks"
        if "name" in body and "skills" in body:
            return f"card: {body['name']}, skills {[s['id'] for s in body['skills']]}"
        if "entries" in body:
            return f"{len(body['entries'])} memory entries"
        if "deleted" in body:
            return f"deleted {body['deleted']}"
        if "id" in body and "text" in body:
            return f"remembered \"{body['text']}\""
    return "200"


def serve(app, port: int) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise SystemExit(f"nothing answered on port {port}")


def record(router: str) -> dict:
    rec = Recorder()
    fiqa_url, coord_url = "http://127.0.0.1:8300", "http://127.0.0.1:8310"
    tok = {u: token_for(u) for u in ("alice", "bob", "admin")}
    who = {tok["alice"]: "alice", tok["bob"]: "bob"}
    serve(create_app(router, memory=MemoryStore(), base_url=fiqa_url,
                     search_factory=lambda t: rec.search_fn(router, t, who.get(t, "anonymous"))), 8300)
    serve(create_coordinator_app(fiqa_url, coord_url,
                                 http_factory=lambda t: rec.client("coordinator :8310", "FiQA agent :8300")), 8310)
    admin = httpx.Client(base_url=router, headers={"Authorization": f"Bearer {tok['admin']}"}, timeout=30)
    original = {k: v["value"] for k, v in admin.get("/admin/security").json().items()}
    admin.post("/admin/security", json={**{k: "off" for k in original}, "identity": "on", "tenant_filter": "on"})
    out = {"recorded": time.strftime("%Y-%m-%d"), "router": router, "parties": PARTIES,
           "router_toggles": "identity and tenant_filter on (restored afterwards)"}
    try:
        rec.scene("1. Discovery", "alice's client reads the coordinator's agent card to learn its endpoint, protocol "
                  "version and skills. The coordinator does the same with the FiQA agent the first time it needs it.")
        alice = A2AClient(coord_url, tok["alice"], http=rec.client("alice", "coordinator :8310"))
        rec.scene("2. Delegation", f'alice asks the coordinator a compound question. It splits it, sends each part to '
                  'the FiQA agent as its own task (with alice\'s token), and fuses the two answers.')
        t_alice = alice.send(Q, k=5)
        rec.scene("3. Tasks are private", "bob asks the coordinator for alice's task by its id. To anyone but its owner "
                  "a task does not exist, so the answer is the same as for a made-up id.")
        bob = A2AClient(coord_url, tok["bob"], http=rec.client("bob", "coordinator :8310"))
        try:
            bob.get(t_alice["id"])
            bob_reads = True
        except A2AError:
            bob_reads = False
        bob.list()
        rec.scene("4. Tenants", "bob asks the same question through the same two agents. The router filters every "
                  "search to the caller's tenant, so bob gets globex's passages where alice got acme's.")
        t_bob = bob.send(Q, k=5)
        rec.scene("5. Memory", "alice tells the FiQA agent she lives in Canada (its memory API). Her next delegated "
                  "parts are searched with Canada added; bob's are not. Then alice deletes everything kept about her.")
        a_direct = rec.client("alice", "FiQA agent :8300")
        a_direct.post(f"{fiqa_url}/memory", json={"text": "I live in Canada"},
                      headers={"Authorization": f"Bearer {tok['alice']}"})
        t_alice2 = alice.send(Q2, k=5)
        t_bob2 = bob.send(Q2, k=5)
        a_direct.get(f"{fiqa_url}/memory", headers={"Authorization": f"Bearer {tok['alice']}"})
        a_direct.delete(f"{fiqa_url}/memory", headers={"Authorization": f"Bearer {tok['alice']}"})
        out["summary"] = {"alice_task": t_alice["status"]["state"], "alice_docs": doc_ids(t_alice),
                          "bob_reads_alices_task": bob_reads, "bob_docs": doc_ids(t_bob),
                          "alice_q2_docs": doc_ids(t_alice2), "bob_q2_docs": doc_ids(t_bob2)}
    finally:
        admin.post("/admin/security", json=original)
    out["events"] = rec.events
    searched = [e["body"]["query"] for e in rec.events if e["kind"] == "request" and e["dst"] == "router :8000"]
    out["summary"]["router_queries"] = searched
    return out


# ------------------------------------------------------------------ rendering
CSS = """
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f6f7f9;color:#1f2328}
header{background:#1e2a3a;color:#fff;padding:10px 20px;display:flex;gap:10px;align-items:center;position:sticky;top:0;z-index:5}
header h1{font-size:18px;margin:0 16px 0 0} header button{padding:6px 12px;border:0;border-radius:4px;background:#3b82f6;color:#fff;cursor:pointer}
header span{font-size:13px;color:#cbd5e1}
.wrap{display:flex} #seq{flex:1;min-width:0;padding:0 0 40px} #side{width:430px;flex:none;background:#fff;border-left:1px solid #ddd;height:calc(100vh - 46px);position:sticky;top:46px;overflow:auto;padding:12px 16px;box-sizing:border-box}
.heads{display:grid;grid-template-columns:repeat(5,1fr);position:sticky;top:46px;background:#f6f7f9;z-index:4;border-bottom:1px solid #ddd}
.heads div{text-align:center;font-weight:600;font-size:13px;padding:10px 4px}
.heads .p0{color:#1d4ed8}.heads .p1{color:#b45309}.heads .p2{color:#047857}.heads .p3{color:#7c3aed}.heads .p4{color:#475569}
.rows{position:relative;background-image:linear-gradient(to right,transparent calc(10% - 1px),#cbd5e1 calc(10% - 1px),#cbd5e1 10%,transparent 10%,transparent calc(30% - 1px),#cbd5e1 calc(30% - 1px),#cbd5e1 30%,transparent 30%,transparent calc(50% - 1px),#cbd5e1 calc(50% - 1px),#cbd5e1 50%,transparent 50%,transparent calc(70% - 1px),#cbd5e1 calc(70% - 1px),#cbd5e1 70%,transparent 70%,transparent calc(90% - 1px),#cbd5e1 calc(90% - 1px),#cbd5e1 90%,transparent 90%)}
.row{position:relative;height:38px;cursor:pointer} .row.hidden{visibility:hidden} .row:hover{background:rgba(59,130,246,.06)} .row.sel{background:rgba(59,130,246,.12)}
.arrow{position:absolute;top:24px;height:0;border-top:2px solid #334155}
.arrow.resp{border-top-style:dashed;border-top-color:#64748b} .arrow.err{border-top-color:#dc2626}
.arrow::after{content:"";position:absolute;top:-6px;border:5px solid transparent}
.arrow.right::after{right:-2px;border-left:8px solid #334155} .arrow.left::after{left:-2px;border-right:8px solid #334155}
.arrow.resp.right::after{border-left-color:#64748b} .arrow.resp.left::after{border-right-color:#64748b}
.arrow.err.right::after{border-left-color:#dc2626} .arrow.err.left::after{border-right-color:#dc2626}
.lbl{position:absolute;top:4px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center}
.lbl.err{color:#b91c1c;font-weight:600} .t{position:absolute;left:4px;top:4px;font-size:10px;color:#94a3b8}
.scene{background:#eef2ff;border-top:1px solid #c7d2fe;border-bottom:1px solid #c7d2fe;padding:8px 14px;font-size:13px}
.scene b{display:block;font-size:14px;margin-bottom:2px} .scene.hidden{display:none}
#side h3{margin:4px 0 6px;font-size:14px} #side pre{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:6px;font-size:11.5px;white-space:pre-wrap;word-break:break-word}
#side .hint{color:#64748b;font-size:13px} .sum{font-size:13px;line-height:1.5} .sum code{font-size:12px}
"""


def render(rec: dict) -> str:
    data = json.dumps(rec, ensure_ascii=False).replace("</", "<\\/")
    heads = "".join(f'<div class="p{i}">{p}</div>' for i, p in enumerate(rec["parties"]))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>A2A: two agents, recorded</title><style>{CSS}</style></head>
<body><header><h1>A2A replay: two agents</h1><button onclick="play()">&#9654; Play</button><button onclick="step()">Step</button>
<button onclick="showAll()">Show all</button><button onclick="reset()">Reset</button><span id="pos"></span></header>
<div class="wrap"><div id="seq"><div class="heads">{heads}</div><div class="rows" id="rows"></div></div>
<div id="side"><div id="detail"></div></div></div>
<script>
const R = {data};
const P = R.parties, col = n => P.indexOf(n);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
let shown = 0, timer = null;
function build() {{
  const rows = document.getElementById("rows");
  rows.innerHTML = R.events.map((e, i) => {{
    if (e.kind === "scene") return `<div class="scene hidden" id="r${{i}}"><b>${{esc(e.title)}}</b>${{esc(e.text)}}</div>`;
    const a = col(e.src), b = col(e.dst), lo = Math.min(a, b), hi = Math.max(a, b);
    const left = lo * 20 + 10, width = (hi - lo) * 20, dir = b > a ? "right" : "left";
    const err = e.kind === "response" && (e.status !== 200 || /^error/.test(e.label));
    return `<div class="row hidden" id="r${{i}}" onclick="pick(${{i}})"><span class="t">${{e.t}} ms</span>` +
      `<div class="lbl ${{err ? "err" : ""}}" style="left:${{left}}%;width:${{width}}%">${{esc(e.label)}}</div>` +
      `<div class="arrow ${{dir}} ${{e.kind === "response" ? "resp" : ""}} ${{err ? "err" : ""}}" style="left:${{left}}%;width:${{width}}%"></div></div>`;
  }}).join("");
}}
function update() {{
  R.events.forEach((e, i) => document.getElementById("r" + i).classList.toggle("hidden", i >= shown));
  document.getElementById("pos").textContent = `${{shown}} / ${{R.events.length}}`;
  if (shown > 0) {{ const el = document.getElementById("r" + (shown - 1)); el.scrollIntoView({{block: "nearest"}}); if (R.events[shown - 1].kind !== "scene") pick(shown - 1); }}
}}
function step() {{ if (shown < R.events.length) {{ shown++; update(); }} else stop(); }}
function play() {{ stop(); timer = setInterval(step, 650); }}
function stop() {{ if (timer) clearInterval(timer); timer = null; }}
function reset() {{ stop(); shown = 0; update(); summary(); }}
function showAll() {{ stop(); shown = R.events.length; update(); }}
function pick(i) {{
  document.querySelectorAll(".row.sel").forEach(x => x.classList.remove("sel"));
  document.getElementById("r" + i).classList.add("sel");
  const e = R.events[i];
  const head = e.kind === "request" ? `${{esc(e.src)}} &rarr; ${{esc(e.dst)}}: ${{esc(e.http || "")}}` : `${{esc(e.src)}} &rarr; ${{esc(e.dst)}}: HTTP ${{e.status}}`;
  document.getElementById("detail").innerHTML = `<h3>${{head}}</h3><div class="hint">at ${{e.t}} ms${{e.reply_to != null ? `, answer to message ${{e.reply_to}}` : ""}}</div><pre>${{esc(JSON.stringify(e.body, null, 1))}}</pre>`;
}}
function summary() {{
  const s = R.summary || {{}};
  document.getElementById("detail").innerHTML = `<h3>What this run shows</h3><div class="sum">
  <p>Five scenes, ${{R.events.filter(e => e.kind !== "scene").length}} recorded messages. Press <b>Play</b>, or <b>Step</b> through them; click an arrow to read its JSON.
  Solid arrows are requests, dashed ones answers, red ones errors.</p>
  <p>alice's compound question: <code>${{esc(s.alice_task)}}</code>, docs <code>${{esc((s.alice_docs || []).join(", "))}}</code></p>
  <p>bob reading alice's task: <code>${{s.bob_reads_alices_task ? "allowed (a bug)" : "Task not found"}}</code></p>
  <p>bob's answer to the same question: <code>${{esc((s.bob_docs || []).join(", "))}}</code></p>
  <p>What the router was asked, in order:<br>${{(s.router_queries || []).map(q => `<code>${{esc(q)}}</code>`).join("<br>")}}</p>
  <p class="hint">Recorded ${{esc(R.recorded)}} against ${{esc(R.router)}}; router toggles: ${{esc(R.router_toggles)}}.</p></div>`;
}}
build(); reset();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--router", default="http://127.0.0.1:8000")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        rec = json.loads((OUT / "messages.json").read_text(encoding="utf-8"))
    else:
        rec = record(a.router)
        (OUT / "messages.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        s = rec["summary"]
        print(f"{len(rec['events'])} events; bob reads alice's task: {s['bob_reads_alices_task']}")
        for q in s["router_queries"]:
            print("  router asked:", q)
    (OUT / "replay.html").write_text(render(rec), encoding="utf-8")
    print(f"wrote {(OUT / 'replay.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
