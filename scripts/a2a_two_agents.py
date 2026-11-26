"""Two live A2A agents talking, with a real LLM behind them, recorded message by message as a sequence diagram.

    docker compose up -d                                   # the router on :8000
    python scripts/a2a_two_agents.py --confirm-spend       # starts the FiQA agent (:8300) and the coordinator (:8310)
                                                           # in this process; 8 Azure calls, about $0.01
    python scripts/a2a_two_agents.py --backend fake        # the same run with the scripted fake LLM (free)
    python scripts/a2a_two_agents.py --render-only

Writes replays/data/a2a_messages.json and replays/a2a.html. Open the page in a browser: the nav on the left lists
the five scenes; Play or Step shows every message between alice, bob, the coordinator, the FiQA agent, the router
and the LLM as an arrow, in the order it happened; click one to read its JSON.

HOW TO READ THE ARROWS
    A request has the colour of whoever sent it (solid), and its answer has the same colour (dashed), so one
    caller's conversation is one colour all the way down: alice blue, bob brown, the coordinator green, the FiQA
    agent purple. Red is used only for errors. A pink tag under an arrow says what the FiQA agent's memory stored
    (or deleted) while answering that message.

THE SCENES
    1. Discovery      alice reads the coordinator's card; the coordinator reads the FiQA agent's card.
    2. Delegation     alice asks a compound question; the coordinator splits it and sends each part to the FiQA
                      agent's ``answer`` skill, with alice's token; the FiQA agent searches the router as alice and
                      its LLM writes a short answer per part.
    3. Tasks are private   bob asks the coordinator for alice's task: "Task not found". His task list is empty.
    4. Tenants        bob asks the same question; with the router's tenant_filter on, his passages are globex's
                      and alice's are acme's, so the LLM answers each of them from different passages.
    5. Memory         alice tells the FiQA agent she lives in Canada; her next delegated parts are searched, and
                      answered, with Canada in mind; bob's are not. alice deletes her memory at the end.

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
from scripts.replay_style import ERROR, NAV_CSS, nav_html  # noqa: E402
from src.a2a.client import A2AClient, A2AError, doc_ids  # noqa: E402
from src.a2a.coordinator import create_coordinator_app  # noqa: E402
from src.a2a.server import create_app  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.agent.tools import http_search_fn  # noqa: E402

OUT = ROOT / "replays"
DATA = OUT / "data" / "a2a_messages.json"
PAGE = OUT / "a2a.html"
LLM_LANE = "LLM (Azure)"
PARTIES = ["alice", "bob", "coordinator :8310", "FiQA agent :8300", "router :8000", LLM_LANE]
# A request has its sender's colour; its answer has the same colour, dashed. Red (ERROR) is for errors only.
PARTY_COLOR = {"alice": "#1d4ed8", "bob": "#92400e", "coordinator :8310": "#047857", "FiQA agent :8300": "#7c3aed",
               "router :8000": "#475569", LLM_LANE: "#be185d"}
TENANT = {"alice": "acme", "bob": "globex"}
Q = "what is a roth ira, and how do I buy my first home?"
Q2 = "how are capital gains taxed, and what retirement account should I open?"


def shrink(o, limit: int = 140):
    """Keep the shape of a JSON message, cut long strings and lists so the replay stays readable. Strings under a
    "text" key keep 700 characters: they are the passages, prompts and answers the replay is about."""
    if isinstance(o, str):
        return o if len(o) <= limit else o[:limit] + "..."
    if isinstance(o, list):
        return [shrink(x, limit) for x in o[:5]] + ([f"... {len(o) - 5} more"] if len(o) > 5 else [])
    if isinstance(o, dict):
        return {k: shrink(v, 700 if k in ("text", "input", "llm_saw") else 140) for k, v in o.items()}
    return o


class Recorder:
    def __init__(self):
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.memory: MemoryStore | None = None
        self.seen: dict[str, dict] = {}

    def now(self) -> float:
        return round((time.perf_counter() - self.t0) * 1e3, 1)

    def add(self, **e) -> int:
        with self.lock:
            e["seq"] = len(self.events)
            self.events.append(e)
            return e["seq"]

    def scene(self, title: str, text: str) -> None:
        self.add(kind="scene", title=title, text=text, t=self.now())

    def memory_change(self) -> list[dict]:
        """What the FiQA agent's memory gained or lost since the last look, per user (for the pink tags)."""
        out = []
        if self.memory is None:
            return out
        time.sleep(0.05)   # the agent writes the turn right after it completes the task
        for user, tenant in TENANT.items():
            now = {e["id"]: e for e in self.memory.export(tenant, user)}
            before = self.seen.get(user, {})
            out += [{"user": user, "op": "+", "kind": e["kind"], "text": e["text"]} for i, e in now.items() if i not in before]
            out += [{"user": user, "op": "-", "kind": e["kind"], "text": e["text"]} for i, e in before.items() if i not in now]
            self.seen[user] = now
        return out

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
            mem = self.memory_change() if dst == "FiQA agent :8300" else []
            self.add(kind="response", src=dst, dst=src, t=self.now(), status=resp.status_code,
                     label=answer_of(resp.status_code, body), body=shrink(body), reply_to=resp.request.extensions.get("seq"),
                     **({"memory": mem} if mem else {}))

        return httpx.Client(timeout=120, event_hooks={"request": [on_request], "response": [on_response]})

    def llm(self, llm):
        """Wrap the FiQA agent's LLM so every call is two arrows: FiQA agent -> LLM and back."""
        rec = self

        class Recorded:
            def complete(self, system, user, max_output_tokens=300, budget=None):
                s = rec.add(kind="request", src="FiQA agent :8300", dst=LLM_LANE, t=rec.now(), http="responses.create",
                            label=f"prompt: {user.splitlines()[0][:52]}",
                            body={"model": getattr(llm, "model", None), "instructions": system.splitlines()[0],
                                  "input": shrink(user, 700), "max_output_tokens": max_output_tokens})
                r = llm.complete(system, user, max_output_tokens, budget)
                rec.add(kind="response", src=LLM_LANE, dst="FiQA agent :8300", t=rec.now(), status=200, reply_to=s,
                        label=f"answer, {r.output_tokens} tokens: {r.text.strip()[:44]}",
                        body={"output_text": shrink(r.text, 700), "usage": {"input_tokens": r.input_tokens,
                              "output_tokens": r.output_tokens}, "cost_usd": round(r.cost_usd, 6)})
                return r

            def __getattr__(self, name):
                return getattr(llm, name)

        return Recorded()

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


def record(router: str, llm) -> dict:
    rec = Recorder()
    fiqa_url, coord_url = "http://127.0.0.1:8300", "http://127.0.0.1:8310"
    tok = {u: token_for(u) for u in ("alice", "bob", "admin")}
    who = {tok["alice"]: "alice", tok["bob"]: "bob"}
    rec.memory = MemoryStore()
    serve(create_app(router, llm=rec.llm(llm), memory=rec.memory, base_url=fiqa_url,
                     search_factory=lambda t: rec.search_fn(router, t, who.get(t, "anonymous"))), 8300)
    serve(create_coordinator_app(fiqa_url, coord_url, part_skill="answer",
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
                  'the FiQA agent\'s answer skill as its own task (with alice\'s token). For each part the FiQA agent '
                  'searches the router as alice and its LLM writes a short answer; the coordinator fuses the passages.')
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
                  "search to the caller's tenant, so bob gets globex's passages where alice got acme's, and the LLM "
                  "answers him from those.")
        t_bob = bob.send(Q, k=5)
        rec.scene("5. Memory", "alice tells the FiQA agent she lives in Canada (its memory API). Her next delegated "
                  "parts are searched with Canada added and the LLM reads the fact; bob's are not. Then alice deletes "
                  "everything kept about her.")
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
    calls = [e for e in rec.events if e["kind"] == "response" and e["src"] == LLM_LANE]
    out["summary"]["llm_calls"] = len(calls)
    out["summary"]["llm_usd"] = round(sum(e["body"]["cost_usd"] for e in calls), 5)
    out["summary"]["llm_model"] = getattr(llm, "model", None)
    return out


# ------------------------------------------------------------------ rendering
CSS = """
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f6f7f9;color:#1f2328}
header.top{background:#1e2a3a;color:#fff;padding:10px 20px;display:flex;gap:10px;align-items:center;position:sticky;top:0;z-index:6;height:26px}
header.top h1{font-size:18px;margin:0 16px 0 0} header.top button{padding:6px 12px;border:0;border-radius:4px;background:#3b82f6;color:#fff;cursor:pointer}
header.top span{font-size:13px;color:#cbd5e1}
.wrap{display:flex} aside{height:calc(100vh - 46px);position:sticky;top:46px}
#seq{flex:1;min-width:0;padding:0 0 40px}
#side{width:410px;flex:none;background:#fff;border-left:1px solid #ddd;height:calc(100vh - 46px);position:sticky;top:46px;overflow:auto;padding:12px 16px;box-sizing:border-box}
.heads{display:grid;grid-template-columns:repeat(6,1fr);position:sticky;top:46px;background:#f6f7f9;z-index:4;border-bottom:1px solid #ddd}
.heads div{text-align:center;font-weight:700;font-size:12.5px;padding:9px 2px 7px}
.heads div i{display:block;font-style:normal;font-weight:400;font-size:10.5px;color:#64748b}
.rows{position:relative} .life{position:absolute;top:0;bottom:0;width:0;border-left:1px dashed #cbd5e1}
.row{position:relative;height:38px;cursor:pointer} .row.mem{height:60px}
.row.hidden{visibility:hidden} .row:hover{background:rgba(59,130,246,.06)} .row.sel{background:rgba(59,130,246,.12)}
.arrow{position:absolute;top:24px;height:0;border-top:2.5px solid currentColor}
.arrow.resp{border-top-style:dashed}
.arrow::after{content:"";position:absolute;top:-6.5px;border:5.5px solid transparent}
.arrow.right::after{right:-3px;border-left:9px solid currentColor} .arrow.left::after{left:-3px;border-right:9px solid currentColor}
.lbl{position:absolute;top:4px;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;color:#1f2328}
.lbl.err{font-weight:700} .t{position:absolute;left:4px;top:4px;font-size:10px;color:#94a3b8}
.memtag{position:absolute;top:33px;font-size:11px;background:#fce7f3;color:#9d174d;border:1px solid #f9a8d4;border-radius:9px;padding:1px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scene{border-top:1px solid #c7d2fe;border-bottom:1px solid #c7d2fe;background:#fdf2f8;padding:8px 14px;font-size:13px}
.scene b{display:block;font-size:14px;margin-bottom:2px} .scene.hidden{display:none}
#side h3{margin:4px 0 6px;font-size:14px} #side pre{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:6px;font-size:11.5px;white-space:pre-wrap;word-break:break-word}
#side .hint{color:#64748b;font-size:13px} .sum{font-size:13px;line-height:1.5} .sum code{font-size:12px}
.memo{background:#fce7f3;border:1px solid #f9a8d4;color:#831843;border-radius:6px;padding:8px 10px;font-size:12.5px;margin-top:8px}
.key span{display:inline-block;margin:2px 8px 2px 0;font-size:12px} .key b{display:inline-block;width:22px;border-top:3px solid;vertical-align:middle;margin-right:4px}
"""
LANE_NOTE = {"alice": "acme user", "bob": "globex user", "coordinator :8310": "A2A agent, no LLM",
             "FiQA agent :8300": "A2A agent, memory", "router :8000": "search service", LLM_LANE: "gpt-6-sol"}


def render(rec: dict) -> str:
    data = json.dumps(rec, ensure_ascii=False).replace("</", "<\\/")
    parties = rec["parties"]
    heads = "".join(f'<div style="color:{PARTY_COLOR.get(p, "#333")}">{p}<i>{LANE_NOTE.get(p, "")}</i></div>' for p in parties)
    scenes = [(i, e) for i, e in enumerate(rec["events"]) if e["kind"] == "scene"]
    nav = nav_html([{"title": e["title"], "category": "extensions"} for _, e in scenes])
    key = "".join(f'<span><b style="color:{PARTY_COLOR[p]}"></b>{p.split(" ")[0]}</span>'
                  for p in ("alice", "bob", "coordinator :8310", "FiQA agent :8300")) + \
        f'<span><b style="color:{ERROR}"></b>error</span>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>A2A: two agents and an LLM, recorded</title>
<style>{CSS}{NAV_CSS}</style></head>
<body><header class="top"><h1>A2A replay: two agents and an LLM</h1><button onclick="play()">&#9654; Play</button>
<button onclick="step()">Step</button><button onclick="showAll()">Show all</button><button onclick="reset()">Reset</button>
<span id="pos"></span></header>
<div class="wrap"><aside>{nav}</aside><div id="seq"><div class="heads">{heads}</div><div class="rows" id="rows"></div></div>
<div id="side"><div id="detail"></div></div></div>
<script>
const R = {data};
const P = R.parties, col = n => P.indexOf(n), N = P.length;
const COLOR = {json.dumps(PARTY_COLOR)}, ERR = "{ERROR}";
const SCENES = R.events.map((e, i) => e.kind === "scene" ? i : -1).filter(i => i >= 0);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
const isErr = e => e.kind === "response" && (e.status !== 200 || /^error/.test(e.label));
const asker = e => e.kind === "request" ? e.src : e.dst;   // a response has the colour of whoever asked
const memText = m => m.map(x => `${{x.op === "+" ? "+" : "\\u2212"}} ${{x.user}}'s ${{x.kind}}: ${{x.text}}`).join(" \\u00b7 ");
let shown = 0, timer = null;
function build() {{
  const lanes = P.map((p, i) => `<div class="life" style="left:${{(i + 0.5) * 100 / N}}%"></div>`).join("");
  document.getElementById("rows").innerHTML = lanes + R.events.map((e, i) => {{
    if (e.kind === "scene") return `<div class="scene hidden" id="r${{i}}"><b>${{esc(e.title)}}</b>${{esc(e.text)}}</div>`;
    const a = col(e.src), b = col(e.dst), lo = Math.min(a, b), hi = Math.max(a, b);
    const left = (lo + 0.5) * 100 / N, width = (hi - lo) * 100 / N, dir = b > a ? "right" : "left";
    const c = isErr(e) ? ERR : COLOR[asker(e)];
    const mem = e.memory ? `<div class="memtag" style="left:${{left}}%;max-width:${{width}}%">memory ${{esc(memText(e.memory))}}</div>` : "";
    return `<div class="row hidden ${{e.memory ? "mem" : ""}}" id="r${{i}}" onclick="pick(${{i}})"><span class="t">${{e.t}} ms</span>` +
      `<div class="lbl ${{isErr(e) ? "err" : ""}}" style="left:${{left}}%;width:${{width}}%;${{isErr(e) ? "color:" + ERR : ""}}">${{esc(e.label)}}</div>` +
      `<div class="arrow ${{dir}} ${{e.kind === "response" ? "resp" : ""}}" style="left:${{left}}%;width:${{width}}%;color:${{c}}"></div>${{mem}}</div>`;
  }}).join("");
}}
function nav(i) {{
  document.querySelectorAll("aside li").forEach((l, k) => l.classList.toggle("cur", k === i));
}}
function update() {{
  R.events.forEach((e, i) => document.getElementById("r" + i).classList.toggle("hidden", i >= shown));
  document.getElementById("pos").textContent = `${{shown}} / ${{R.events.length}}`;
  const sc = SCENES.filter(s => s < shown).length - 1; nav(sc);
  if (shown > 0) {{ const el = document.getElementById("r" + (shown - 1)); el.scrollIntoView({{block: "nearest"}}); if (R.events[shown - 1].kind !== "scene") pick(shown - 1); }}
}}
function go(i) {{   // nav: show everything up to the end of scene i
  stop(); shown = i + 1 < SCENES.length ? SCENES[i + 1] : R.events.length; update();
  document.getElementById("r" + SCENES[i]).scrollIntoView({{block: "start"}}); window.scrollBy(0, -80);
}}
function step() {{ if (shown < R.events.length) {{ shown++; update(); }} else stop(); }}
function play() {{ stop(); timer = setInterval(step, 650); }}
function stop() {{ if (timer) clearInterval(timer); timer = null; }}
function reset() {{ stop(); shown = 0; update(); summary(); }}
function showAll() {{ stop(); shown = R.events.length; update(); }}
function pick(i) {{
  document.querySelectorAll(".row.sel").forEach(x => x.classList.remove("sel"));
  document.getElementById("r" + i).classList.add("sel");
  const e = R.events[i], c = isErr(e) ? ERR : COLOR[asker(e)];
  const head = e.kind === "request" ? `${{esc(e.src)}} &rarr; ${{esc(e.dst)}}: ${{esc(e.http || "")}}` : `${{esc(e.src)}} &rarr; ${{esc(e.dst)}}: ${{e.dst === "{LLM_LANE}" || e.src === "{LLM_LANE}" ? "reply" : "HTTP " + e.status}}`;
  const mem = e.memory ? `<div class="memo"><b>Memory changed while the FiQA agent answered this:</b><br>${{e.memory.map(m => `${{m.op === "+" ? "added" : "deleted"}} for ${{esc(m.user)}} (${{esc(m.kind)}}): ${{esc(m.text)}}`).join("<br>")}}</div>` : "";
  document.getElementById("detail").innerHTML = `<h3 style="color:${{c}}">${{head}}</h3><div class="hint">at ${{e.t}} ms${{e.reply_to != null ? `, answer to message ${{e.reply_to}}` : ""}}</div><pre>${{esc(JSON.stringify(e.body, null, 1))}}</pre>${{mem}}`;
}}
function summary() {{
  const s = R.summary || {{}};
  document.getElementById("detail").innerHTML = `<h3>What this run shows</h3><div class="sum">
  <p>Five scenes, ${{R.events.filter(e => e.kind !== "scene").length}} recorded messages, ${{s.llm_calls}} LLM calls
  (${{esc(s.llm_model || "")}}, $${{s.llm_usd}}). Press <b>Play</b>, <b>Step</b> through them, or pick a scene on the left;
  click an arrow to read its JSON.</p>
  <div class="key">{key}</div>
  <p class="hint">Solid: a request, in the colour of whoever sent it. Dashed: its answer, in the same colour. Pink tag: what the FiQA agent's memory stored or deleted.</p>
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
    ap.add_argument("--backend", default="azure", choices=["azure", "fake"])
    ap.add_argument("--confirm-spend", action="store_true", help="needed for --backend azure")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        rec = json.loads(DATA.read_text(encoding="utf-8"))
    else:
        if a.backend == "azure":
            if not a.confirm_spend:
                ap.error("--backend azure spends money: add --confirm-spend (or use --backend fake)")
            from src.agent.llm import get_llm
            llm = get_llm("azure", confirm_spend=True)
        else:
            from src.agent.fake_policies import heuristic_responder
            from src.agent.llm import FakeLLM
            llm = FakeLLM(heuristic_responder)
        rec = record(a.router, llm)
        DATA.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        s = rec["summary"]
        print(f"{len(rec['events'])} events; bob reads alice's task: {s['bob_reads_alices_task']}; "
              f"{s['llm_calls']} LLM calls, ${s['llm_usd']}")
        for q in s["router_queries"]:
            print("  router asked:", q)
    PAGE.write_text(render(rec), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())