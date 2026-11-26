"""Record a conversation with the A2A agent, answered by a real LLM, and show what memory did at every step.

    python scripts/memory_replay.py --confirm-spend   # needs the router on 127.0.0.1:8000; about 13 Azure calls (~$0.02)
    python scripts/memory_replay.py --backend fake    # the same steps with the scripted fake LLM (free, for a dry run)
    python scripts/memory_replay.py --render-only     # rebuild the page from the recorded data

Writes replays/data/memory_steps.json and replays/memory.html.

WHAT RUNS
    The real A2A server (src/a2a/server.py) in this process, with a fresh in-RAM MemoryStore, searching the
    Docker router with each caller's token. Every question goes to the ``answer`` skill: one search, then ONE LLM
    call that writes a short answer from the passages and cites them. Before and after each step the script
    exports the caller's memory, so the page can show, for every step:

        what retrieval saw   the query the agent searched (memory's additions highlighted) and the passages
        what the LLM read    the exact user prompt (task metadata ``llm_saw``): question, memory, passages
        what the LLM wrote   the answer, its tokens and its cost
        what memory did      the entries added or removed, and everything kept about the caller afterwards

    The memory's clock is a function the script controls (``MemoryStore(now=...)``), so "31 days later" is
    one line instead of a month of waiting. Everything else is the code the server runs.

THE SIX DUTIES (docs/topics/22_agent_memory.md) and the steps that show each one
    attribution  only an identified caller gets memory: the anonymous step stores nothing
    isolation    bob, even sending alice's contextId, gets none of alice's turns or facts
    expiry       after 31 days bob's entries are gone and his answer is generic again
    access       alice exports everything kept about her (with the email already redacted)
    deletion     alice deletes one entry, then everything
    opt-out      a message with ``memory: false`` neither reads nor writes memory
    (and redaction, which is not one of the six but goes with them: an email in a message is stored as [EMAIL])

WHY BOB CAN SEND ALICE'S contextId (step 9)
    A contextId is a conversation label. The server makes one up for a first message, and the client sends it back
    to continue that conversation; nothing stops a client from sending any label it likes, including one it saw
    elsewhere. So the label cannot be what protects alice. Memory is stored and read under (tenant, user,
    contextId), and tenant and user come from the signed token, so bob's lookup with alice's label finds nothing.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from scripts.mint_token import token_for  # noqa: E402
from scripts.replay_style import CATEGORIES, NAV_CSS, nav_html  # noqa: E402
from src.a2a.client import A2AClient  # noqa: E402
from src.a2a.server import create_app  # noqa: E402
from src.agent.memory import COUNTRIES, MemoryStore  # noqa: E402
from src.agent.tools import http_guardrail_check, http_search_fn  # noqa: E402

OUT = ROOT / "replays"
DATA = OUT / "data" / "memory_steps.json"
PAGE = OUT / "memory.html"
BASE = "http://agent"
TAXES = "how do I file my income taxes?"
K = 5


def mentions_canada(text: str) -> bool:
    """True if the text names Canada or a Canadian tax term.

    >>> mentions_canada("Put it in your RRSP first."), mentions_canada("File a 1040 with the IRS.")
    (True, False)
    """
    words = text.lower().replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
    return any(w in words for w in COUNTRIES["Canada"])


class Session:
    def __init__(self, router: str, llm):
        self.clock = {"offset": 0.0}
        self.memory = MemoryStore(":memory:", now=lambda: time.time() + self.clock["offset"])
        self.texts: dict[str, str] = {}     # doc id -> full passage, to mark Canada on more than the snippet

        def search_factory(token):
            raw = http_search_fn(router, token)

            def fn(q, strategy, k):
                hits = raw(q, strategy, k)
                self.texts.update({h["doc_id"]: h.get("text") or "" for h in hits})
                return hits
            return fn

        self.tc = TestClient(create_app(llm=llm, memory=self.memory, search_factory=search_factory, base_url=BASE,
                                        guard_factory=lambda tok: http_guardrail_check(router, tok)))
        self.tokens = {u: token_for(u) for u in ("alice", "bob")}
        self.steps: list[dict] = []
        self.contexts: dict[str, str] = {}

    def export(self, user: str | None) -> list[dict]:
        if user is None:
            return []
        r = self.tc.get("/memory", headers={"Authorization": f"Bearer {self.tokens[user]}"}).json()
        return [{"id": e["id"], "kind": e["kind"], "text": e["text"],
                 "context": (e["context_id"] or "")[:8], "days_left": round((e["expires"] - self.memory.now()) / 86400, 1)}
                for e in r["entries"]]

    def step(self, duty: str, who: str | None, title: str, note: str, action) -> dict:
        before = self.export(who)
        t0 = time.perf_counter()
        out = action()
        after = self.export(who)
        ids_before, ids_after = {e["id"] for e in before}, {e["id"] for e in after}
        s = {"n": len(self.steps) + 1, "duty": duty, "who": who or "anonymous", "title": title, "note": note,
             "ms": round((time.perf_counter() - t0) * 1e3, 1), **out,
             "memory_after": after,
             "added": [e for e in after if e["id"] not in ids_before],
             "removed": [e for e in before if e["id"] not in ids_after]}
        self.steps.append(s)
        print(f"{s['n']:2d} {duty:11s} {s['who']:9s} {title}", flush=True)
        return s

    def ask(self, who: str | None, question: str, context: str | None = None, memory: bool = True):
        """One A2A SendMessage to the answer skill. ``context`` names a conversation: same name = same contextId."""
        def action():
            client = A2AClient(BASE, self.tokens.get(who) if who else None, http=self.tc)
            ctx_id = self.contexts.get(context) if context else None
            task = client.send(question, context_id=ctx_id, skill="answer", k=K, memory=memory)
            if context and context not in self.contexts:
                self.contexts[context] = task["contextId"]
            meta = task["metadata"]
            mem = meta.get("memory", {})
            arts = {a["name"]: a for a in task.get("artifacts") or []}
            hits = arts["passages"]["parts"][0]["data"]["hits"] if "passages" in arts else []
            answer = arts["answer"]["parts"][0]["text"] if "answer" in arts else ""
            status = task["status"]
            return {"kind": "ask", "message": question, "context": task["contextId"][:8], "memory_flag": memory,
                    "state": status["state"].replace("TASK_STATE_", ""),
                    "status_text": " ".join(p.get("text", "") for p in (status.get("message") or {}).get("parts", [])),
                    "searched": mem.get("query_searched", question), "memory_enabled": mem.get("enabled", False),
                    "turn_used": mem.get("turn_used"), "facts_used": mem.get("facts_used", []),
                    "llm_saw": meta.get("llm_saw"), "answer": answer, "answer_canada": mentions_canada(answer),
                    "llm": {"calls": meta.get("llm_calls", 0), "in": meta.get("input_tokens", 0),
                            "out": meta.get("output_tokens", 0), "usd": meta.get("usd", 0.0)},
                    "hits": [{"doc_id": h["doc_id"], "snippet": h["snippet"][:150],
                              "canada": mentions_canada(self.texts.get(h["doc_id"], h["snippet"]))} for h in hits]}
        return action

    def rest(self, who: str, method: str, path: str, body: dict | None = None):
        def action():
            r = self.tc.request(method, path, json=body, headers={"Authorization": f"Bearer {self.tokens[who]}"})
            return {"kind": "rest", "message": f"{method} {path}" + (f" {json.dumps(body)}" if body else ""),
                    "status": r.status_code, "response": r.json()}
        return action

    def wait_days(self, days: float):
        def action():
            self.clock["offset"] += days * 86400
            return {"kind": "clock", "message": f"the memory's clock moves forward {days:g} days"}
        return action


def record(router: str, llm, backend: str) -> dict:
    s = Session(router, llm)
    s.step("attribution", None, "An anonymous caller states a fact",
           "No token, so nobody to attribute memory to (and nobody who could later ask to see or delete it). "
           "The question is answered; nothing is stored.",
           s.ask(None, "I live in Canada. " + TAXES))
    s.step("attribution", "alice", "alice's first question, memory still empty",
           "Searched as typed, so the LLM gets whatever the corpus ranks first. This is the baseline for step 6.",
           s.ask("alice", TAXES, "chat-1"))
    s.step("attribution", "alice", "alice states where she lives",
           "The rule-based extractor finds the country in the message and stores it as a fact under "
           "(acme, alice). This question already names Canada, so the fact adds nothing to its search.",
           s.ask("alice", "I live in Canada. How are capital gains taxed?", "chat-1"))
    s.step("attribution", "alice", "a follow-up in the same conversation",
           "\"what about dividends?\" points back (it starts with \"what about\"), so the last turn of this "
           "contextId is searched with it and shown to the LLM. Without it the LLM would not know what \"what about\" "
           "refers to.",
           s.ask("alice", "what about dividends?", "chat-1"))
    s.step("attribution", "alice", "a new topic, no pointer back",
           "No pronoun or \"what about\": the last turn is not added. The fact still is.",
           s.ask("alice", "should I pay off my mortgage early?", "chat-1"))
    s.step("attribution", "alice", "step 2's question again, in a new conversation",
           "A new contextId has no turns, but facts outlive conversations: compare the passages and the answer "
           "with step 2.",
           s.ask("alice", TAXES, "chat-2"))
    s.step("opt-out", "alice", "the same question with memory: false",
           "Message metadata {\"memory\": false}: memory is neither read (no Canada) nor written (no new turn).",
           s.ask("alice", TAXES, "chat-2", memory=False))
    s.step("redaction", "alice", "alice shares an email address",
           "The turn is PII-redacted before it is written, so memory never holds the address. (With the router's "
           "pii_redaction toggle on, the LLM would read [EMAIL] too; here every toggle is off.)",
           s.ask("alice", "my email is jane.doe@example.com, how do I report a phishing scam?", "chat-2"))
    s.step("isolation", "bob", "bob sends alice's contextId with a follow-up",
           "A contextId is only a conversation label: the client sends it back, and bob sends alice's on purpose. "
           "Memory is read by (tenant, user, contextId), and tenant and user come from bob's signed token, so "
           "alice's conversation does not exist for him: no turn, no Canada.",
           s.ask("bob", "what about dividends?", "chat-1"))
    s.step("isolation", "bob", "bob states his own fact",
           "Stored under (globex, bob). alice's memory is untouched.",
           s.ask("bob", "I'm from the UK. " + TAXES, "bob-1"))
    s.step("access", "alice", "alice asks what is kept about her",
           "GET /memory: every entry, oldest first, with the days until it expires. The email shows as [EMAIL].",
           s.rest("alice", "GET", "/memory"))
    target = next(e for e in s.steps[-1]["memory_after"] if "[EMAIL]" in e["text"])
    s.step("deletion", "alice", "alice deletes one entry",
           "DELETE /memory/{id} removes the turn with the (redacted) email; the rest stays.",
           s.rest("alice", "DELETE", f"/memory/{target['id']}"))
    s.step("deletion", "alice", "alice deletes everything",
           "DELETE /memory removes every entry kept about her.", s.rest("alice", "DELETE", "/memory"))
    s.step("deletion", "alice", "the follow-up again, after the deletion",
           "Same contextId as steps 2 to 5, but its turns and the Canada fact are gone: searched as typed, and the "
           "LLM has no idea what \"what about\" refers to.",
           s.ask("alice", "what about dividends?", "chat-1"))
    s.step("expiry", "bob", "31 days pass", "Every entry lives MEMORY_TTL_DAYS (30) days.", s.wait_days(31))
    s.step("expiry", "bob", "bob asks again",
           "His UK fact and his turns have expired: never returned after their end date, and purged on this "
           "step's write. The search and the answer are generic again.", s.ask("bob", TAXES, "bob-1"))
    by_n = {x["n"]: x for x in s.steps}
    asks = [x for x in s.steps if x["kind"] == "ask"]
    summary = {"canada_in_top_k": {f"step {n}": sum(h["canada"] for h in by_n[n]["hits"]) for n in (2, 6, 7)},
               "answer_mentions_canada": {f"step {n}": by_n[n]["answer_canada"] for n in (2, 6, 7)},
               "router": router, "backend": backend, "k": K,
               "llm_calls": sum(x["llm"]["calls"] for x in asks),
               "usd": round(sum(x["llm"]["usd"] for x in asks), 5),
               "model": getattr(s.tc.app, "llm_model", None)}
    return {"steps": s.steps, "summary": summary, "recorded": time.strftime("%Y-%m-%d")}


# ---------------------------------------------------------------------- rendering
CSS = """
body{margin:0;font:14px/1.45 Segoe UI,system-ui,sans-serif;color:#1d2430;background:#f4f5f7}
header.top{position:sticky;top:0;z-index:2;background:#1e2a3a;color:#fff;padding:10px 22px;display:flex;justify-content:space-between;align-items:center}
header.top h1{font-size:19px;margin:0} header.top button{margin-left:6px;padding:6px 14px;border:0;border-radius:4px;background:#3b82f6;color:#fff;cursor:pointer}
.wrap{display:flex} main{flex:1;padding:14px 22px;min-width:0}
.q{background:#fff;border:1px solid #ddd;border-left:6px solid #db2777;border-radius:6px;padding:10px 16px;margin-bottom:12px}
.q h2{font-size:17px;margin:2px 0} .q .note{color:#555;margin:4px 0 0}
.pill{display:inline-block;color:#fff;border-radius:10px;padding:1px 9px;font-size:12px;margin-right:6px}
.msg{display:inline-block;background:#dbeafe;border-radius:12px;padding:6px 12px;margin:8px 0 2px;font-size:15px}
.msg.clock{background:#fdf2d0}
.muted{color:#667085;font-size:12.5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden;min-width:0}
.panel h3{font-size:13px;margin:0;padding:8px 12px;border-bottom:1px solid #e5e7eb;text-transform:uppercase;letter-spacing:.03em}
.panel .b{padding:8px 12px}
.p-ret h3{background:#e0f2fe;color:#075985} .p-llm h3{background:#ede9fe;color:#5b21b6}
.p-ans h3{background:#dcfce7;color:#166534} .p-mem h3{background:#fce7f3;color:#9d174d}
mark{background:#fff3a3} code{font-family:Consolas,monospace;font-size:12.5px}
ol.hits{margin:6px 0 0 18px;padding:0;font-size:12.5px} ol.hits li{margin:3px 0}
.ca{background:#0f766e;color:#fff;border-radius:4px;padding:0 4px;font-size:11px}
pre{background:#0f172a;color:#e2e8f0;padding:8px 10px;border-radius:6px;font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:330px;overflow:auto;margin:0}
.answer{font-size:14.5px;line-height:1.5;white-space:pre-wrap}
.add{color:#15803d} .rem{color:#b91c1c;text-decoration:line-through}
table{border-collapse:collapse;width:100%;font-size:12px} td,th{border-bottom:1px solid #eee;padding:3px 5px;text-align:left;vertical-align:top}
.sum{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 14px;margin-top:12px;font-size:13px}
"""
DUTY_COLOR = {"attribution": "#1f6feb", "isolation": "#8250df", "expiry": "#9a6700", "access": "#1a7f37",
              "deletion": "#b91c1c", "opt-out": "#57606a", "redaction": "#0e7490"}


def searched_html(step: dict) -> str:
    q, msg = step["searched"], step["message"]
    if q == msg:
        return f"<code>{html.escape(q)}</code> <span class='muted'>(as typed)</span>"
    i = q.find(msg)
    if i < 0:
        return f"<code>{html.escape(q)}</code>"
    return f"<code><mark>{html.escape(q[:i])}</mark>{html.escape(msg)}<mark>{html.escape(q[i + len(msg):])}</mark></code>"


def step_html(s: dict) -> str:
    color = DUTY_COLOR[s["duty"]]
    head = (f"<div class='q'><span class='pill' style='background:{color}'>{s['duty']}</span>"
            f"<span class='muted'>step {s['n']} &middot; {html.escape(s['who'])}</span>"
            f"<h2>{html.escape(s['title'])}</h2><p class='note'>{html.escape(s['note'])}</p>")
    if s["kind"] == "clock":
        head += f"<div class='msg clock'>&#9201; {html.escape(s['message'])}</div>"
    else:
        head += f"<div class='msg'><b>{html.escape(s['who'])}</b>: {html.escape(s['message'])}</div>"
    if s["kind"] == "ask":
        extra = ([] if s["memory_flag"] else ["metadata memory: false"]) + [f"contextId {s['context']}",
                                                                            f"task {s['state']}", f"{s['ms']:.0f} ms"]
        head += f"<div class='muted'>{' &middot; '.join(extra)}</div>"
    head += "</div>"
    diff = "".join(f"<div class='add'>+ {e['kind']}: {html.escape(e['text'])}</div>" for e in s["added"]) + \
        "".join(f"<div class='rem'>&minus; {e['kind']}: {html.escape(e['text'])}</div>" for e in s["removed"])
    rows = "".join(f"<tr><td>{e['kind']}</td><td>{html.escape(e['text'])}</td><td>{e['context'] or '-'}</td>"
                   f"<td>{e['days_left']}</td></tr>" for e in s["memory_after"])
    mem = (f"<div class='panel p-mem'><h3>What memory did</h3><div class='b'>{diff or '<div class=muted>no change</div>'}"
           f"<div class='muted' style='margin-top:8px'>{html.escape(s['who'])}'s memory after this step</div>"
           + (f"<table><tr><th>kind</th><th>text</th><th>context</th><th>days left</th></tr>{rows}</table>"
              if rows else "<div class='muted'>empty</div>") + "</div></div>")
    if s["kind"] == "ask":
        reads = []
        if s["turn_used"]:
            reads.append(f"last turn: &ldquo;{html.escape(s['turn_used'])}&rdquo;")
        if s["facts_used"]:
            reads.append("facts: " + ", ".join(html.escape(f) for f in s["facts_used"]))
        read = "; ".join(reads) if reads else ("nothing" if s["memory_enabled"] else "memory off for this caller / message")
        hits = "".join(f"<li><code>{h['doc_id']}</code>{' <span class=ca>Canada</span>' if h['canada'] else ''} "
                       f"{html.escape(h['snippet'])}</li>" for h in s["hits"])
        ret = (f"<div class='panel p-ret'><h3>What retrieval saw</h3><div class='b'><div><span class='muted'>memory "
               f"read:</span> {read}</div><div><span class='muted'>searched as:</span> {searched_html(s)}</div>"
               f"<ol class='hits'>{hits}</ol></div></div>")
        llm = (f"<div class='panel p-llm'><h3>What the LLM read (one call)</h3><div class='b'>"
               + (f"<pre>{html.escape(s['llm_saw'])}</pre>" if s.get("llm_saw") else "<div class='muted'>no LLM call</div>")
               + "</div></div>")
        cost = s["llm"]
        ans = (f"<div class='panel p-ans'><h3>What the LLM answered</h3><div class='b'><div class='answer'>"
               f"{html.escape(s['answer'] or s.get('status_text') or '')}</div><div class='muted' style='margin-top:6px'>"
               f"{cost['calls']} call, {cost['in']} tokens in, {cost['out']} out, ${cost['usd']}"
               + (" &middot; <span class=ca>mentions Canada</span>" if s["answer_canada"] else "") + "</div></div></div>")
        return head + f"<div class='grid'>{ret}{llm}{ans}{mem}</div>"
    if s["kind"] == "rest":
        resp = (f"<div class='panel p-ret'><h3>The request and its response</h3><div class='b'><div class='muted'>"
                f"HTTP {s['status']}</div><pre>{html.escape(json.dumps(s['response'], indent=1))}</pre></div></div>")
        return head + f"<div class='grid'>{resp}{mem}</div>"
    return head + f"<div class='grid'>{mem}</div>"


def render(rec: dict) -> str:
    steps = rec["steps"]
    nav = nav_html([{"title": s["title"], "category": "extensions", "group": s["duty"]} for s in steps])
    sm = rec["summary"]
    ck, ac = sm["canada_in_top_k"], sm["answer_mentions_canada"]
    yes = lambda b: "yes" if b else "no"  # noqa: E731
    summary = (f"Passages that mention Canada in the top {sm['k']} for &ldquo;{TAXES}&rdquo;: {ck['step 2']} before "
               f"alice's fact (step 2), {ck['step 6']} with it (step 6), {ck['step 7']} with <code>memory: false</code> "
               f"(step 7). Does the LLM's answer talk about Canada? step 2: {yes(ac['step 2'])}, step 6: "
               f"{yes(ac['step 6'])}, step 7: {yes(ac['step 7'])}. {sm['llm_calls']} LLM calls ({sm['backend']}), "
               f"${sm['usd']} in total, recorded {html.escape(rec.get('recorded', ''))}.")
    data = json.dumps([step_html(s) for s in steps], ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Memory replay</title>
<style>{CSS}{NAV_CSS}</style></head><body>
<header class="top"><h1 id="h">Memory replay</h1><div><button onclick="go(cur-1)">&larr; prev</button>
<button onclick="go(cur+1)">next &rarr;</button></div></header>
<div class="wrap"><aside>{nav}</aside><main><div id="m"></div>
<div class="sum">{summary} One recorded session with the A2A agent (<code>src/a2a/server.py</code>, skill
<code>answer</code>), searching the Docker router. Explained in <code>docs/topics/22_agent_memory.md</code>.
Colour: {CATEGORIES['extensions']['label']} (memory). Arrow keys move between steps.</div></main></div>
<script>
const S = {data};
let cur = 0;
function go(i) {{
  if (i < 0 || i >= S.length) return;
  document.querySelectorAll("aside li").forEach(l => l.classList.remove("cur"));
  document.getElementById("l" + i).classList.add("cur");
  cur = i; document.getElementById("m").innerHTML = S[i];
  document.getElementById("h").textContent = `Memory replay · step ${{i + 1}} of ${{S.length}}`;
  location.hash = i + 1;
}}
document.addEventListener("keydown", e => {{ if (e.key === "ArrowRight") go(cur + 1); if (e.key === "ArrowLeft") go(cur - 1); }});
go(Math.max(0, (parseInt(location.hash.slice(1)) || 1) - 1));
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
        rec = record(a.router, llm, a.backend)
        rec["summary"]["model"] = getattr(llm, "model", None)
        DATA.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    PAGE.write_text(render(rec), encoding="utf-8")
    print("canada in top k:", rec["summary"]["canada_in_top_k"], "| answer mentions canada:",
          rec["summary"]["answer_mentions_canada"], "| usd:", rec["summary"]["usd"])
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
