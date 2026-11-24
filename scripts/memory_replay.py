"""Record a conversation with the A2A agent and show what its memory did at every step, as an HTML replay.

    python scripts/memory_replay.py                  # needs the router on 127.0.0.1:8000; no LLM calls
    python scripts/memory_replay.py --render-only    # rebuild replay.html from steps.json

Writes scenarios/SimpleScenarios/results/MemoryReplay/{steps.json, replay.html}.

WHAT RUNS
    The real A2A server (src/a2a/server.py) in this process, with a fresh in-RAM MemoryStore, searching the
    Docker router with each caller's token. Every step is one A2A message or one /memory call. Before and after
    each step the script exports the caller's memory, so the page can show exactly which entries were added
    or removed, next to the query the agent actually searched (task metadata ``memory.query_searched``).

    The memory's clock is a function the script controls (``MemoryStore(now=...)``), so "31 days later" is
    one line instead of a month of waiting. Everything else is the code the server runs.

THE SIX DUTIES (docs/topics/22_agent_memory.md) and the steps that show each one
    attribution  only an identified caller gets memory: the anonymous step stores nothing
    isolation    bob, even sending alice's contextId, gets none of alice's turns or facts
    expiry       after 31 days bob's entries are gone and his search is plain again
    access       alice exports everything kept about her (with the email already redacted)
    deletion     alice deletes one entry, then everything
    opt-out      a message with ``memory: false`` neither reads nor writes memory
    (and redaction, which is not one of the six but goes with them: an email in a message is stored as [EMAIL])
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
from src.a2a.client import A2AClient  # noqa: E402
from src.a2a.server import create_app  # noqa: E402
from src.agent.memory import COUNTRIES, MemoryStore  # noqa: E402
from src.agent.tools import http_search_fn  # noqa: E402

OUT = ROOT / "scenarios" / "SimpleScenarios" / "results" / "MemoryReplay"
BASE = "http://agent"
TAXES = "how do I file my income taxes?"


def mentions_canada(text: str) -> bool:
    words = text.lower().replace(",", " ").replace(".", " ").split()
    return any(w in words for w in COUNTRIES["Canada"])


class Session:
    def __init__(self, router: str):
        self.clock = {"offset": 0.0}
        self.memory = MemoryStore(":memory:", now=lambda: time.time() + self.clock["offset"])
        self.texts: dict[str, str] = {}     # doc id -> full passage, to mark Canada on more than the snippet

        def search_factory(token):
            raw = http_search_fn(router, token)

            def fn(q, strategy, k):
                hits = raw(q, strategy, k)
                self.texts.update({h['doc_id']: h.get('text', '') for h in hits})
                return hits
            return fn

        self.tc = TestClient(create_app(memory=self.memory, search_factory=search_factory, base_url=BASE))
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
        print(f"{s['n']:2d} {duty:11s} {s['who']:9s} {title}")
        return s

    def ask(self, who: str | None, question: str, context: str | None = None, memory: bool = True):
        """One A2A SendMessage (skill retrieve, k=5). ``context`` names a conversation: same name = same contextId."""
        def action():
            client = A2AClient(BASE, self.tokens.get(who) if who else None, http=self.tc)
            ctx_id = self.contexts.get(context) if context else None
            task = client.send(question, context_id=ctx_id, k=5, memory=memory)
            if context and context not in self.contexts:
                self.contexts[context] = task["contextId"]
            meta = task["metadata"].get("memory", {})
            hits = task["artifacts"][0]["parts"][0]["data"]["hits"] if task.get("artifacts") else []
            return {"kind": "ask", "message": question, "context": task["contextId"][:8], "memory_flag": memory,
                    "state": task["status"]["state"].replace("TASK_STATE_", ""),
                    "searched": meta.get("query_searched", question), "memory_enabled": meta.get("enabled", False),
                    "turn_used": meta.get("turn_used"), "facts_used": meta.get("facts_used", []),
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


def record(router: str) -> dict:
    s = Session(router)
    s.step("attribution", None, "An anonymous caller states a fact",
           "No token, so nobody to attribute memory to (and nobody who could later ask to see or delete it). "
           "The question is answered; nothing is stored.",
           s.ask(None, "I live in Canada. " + TAXES))
    s.step("attribution", "alice", "alice's first question, memory still empty",
           "Searched as typed. This is the baseline for step 6.", s.ask("alice", TAXES, "chat-1"))
    s.step("attribution", "alice", "alice states where she lives",
           "The rule-based extractor finds the country in the message and stores it as a fact under "
           "(acme, alice). This question already names Canada, so the fact adds nothing to its search.",
           s.ask("alice", "I live in Canada. How are capital gains taxed?", "chat-1"))
    s.step("attribution", "alice", "a follow-up in the same conversation",
           "\"what about dividends?\" points back (it starts with \"what about\"), so the last turn of this "
           "contextId is searched with it. That turn already says Canada, so the fact is not added twice.",
           s.ask("alice", "what about dividends?", "chat-1"))
    s.step("attribution", "alice", "a new topic, no pointer back",
           "No pronoun or \"what about\": the last turn is not added. The fact still is.",
           s.ask("alice", "should I pay off my mortgage early?", "chat-1"))
    s.step("attribution", "alice", "step 2's question again, in a new conversation",
           "A new contextId has no turns, but facts outlive conversations: compare the passages with step 2.",
           s.ask("alice", TAXES, "chat-2"))
    s.step("opt-out", "alice", "the same question with memory: false",
           "Message metadata {\"memory\": false}: memory is neither read (no Canada) nor written (no new turn).",
           s.ask("alice", TAXES, "chat-2", memory=False))
    s.step("redaction", "alice", "alice shares an email address",
           "The turn is PII-redacted before it is written, so memory never holds the address. The search "
           "itself still sends the address to the router, which searches the question as it arrives (its audit "
           "log stores the question redacted, 17_security.md).",
           s.ask("alice", "my email is jane.doe@example.com, how do I report a phishing scam?", "chat-2"))
    s.step("isolation", "bob", "bob sends alice's contextId with a follow-up",
           "Memory is read by (tenant, user, contextId). bob is another user, so alice's conversation does not "
           "exist for him: no turn, no Canada.",
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
           "Same contextId as steps 2 to 5, but its turns and the Canada fact are gone: searched as typed.",
           s.ask("alice", "what about dividends?", "chat-1"))
    s.step("expiry", "bob", "31 days pass", "Every entry lives MEMORY_TTL_DAYS (30) days.", s.wait_days(31))
    s.step("expiry", "bob", "bob asks again",
           "His UK fact and his turns have expired: never returned after their end date, and purged on this "
           "step's write. The search is plain again.", s.ask("bob", TAXES, "bob-1"))
    by_n = {x["n"]: x for x in s.steps}
    summary = {"canada_in_top5": {"step 2 (no fact)": sum(h["canada"] for h in by_n[2]["hits"]),
                                  "step 6 (Canada fact)": sum(h["canada"] for h in by_n[6]["hits"]),
                                  "step 7 (opt-out)": sum(h["canada"] for h in by_n[7]["hits"])},
               "router": router, "llm_calls": 0}
    return {"steps": s.steps, "summary": summary}


# ---------------------------------------------------------------------- rendering
DUTY_COLOR = {"attribution": "#1f6feb", "isolation": "#8250df", "expiry": "#9a6700", "access": "#1a7f37",
              "deletion": "#c62828", "opt-out": "#57606a", "redaction": "#0e7490"}


def _searched_html(step: dict) -> str:
    q, msg = step["searched"], step["message"]
    if q == msg:
        return f"<code>{html.escape(q)}</code> <span class='muted'>(as typed)</span>"
    i = q.find(msg)
    if i < 0:
        return f"<code>{html.escape(q)}</code>"
    return (f"<code><mark>{html.escape(q[:i])}</mark>{html.escape(msg)}<mark>{html.escape(q[i + len(msg):])}</mark></code>")


def render(rec: dict) -> str:
    cards = []
    for s in rec["steps"]:
        color = DUTY_COLOR[s["duty"]]
        body = [f"<div class='msg clock'>&#9201; {html.escape(s['message'])}</div>" if s["kind"] == "clock" else
                f"<div class='msg'><b>{html.escape(s['who'])}</b>: {html.escape(s['message'])}</div>"]
        if s["kind"] == "ask":
            extra = []
            if not s["memory_flag"]:
                extra.append("metadata memory: false")
            extra.append(f"contextId {s['context']}")
            body.append(f"<div class='muted'>{' &middot; '.join(extra)} &middot; task {s['state']} &middot; {s['ms']} ms</div>")
            reads = []
            if s["turn_used"]:
                reads.append(f"last turn: \"{html.escape(s['turn_used'])}\"")
            if s["facts_used"]:
                reads.append("facts: " + ", ".join(html.escape(f) for f in s["facts_used"]))
            body.append("<div><span class='lbl'>memory read</span> "
                        + ("; ".join(reads) if reads else ("<span class='muted'>nothing</span>" if s["memory_enabled"]
                                                            else "<span class='muted'>memory off for this caller / message</span>"))
                        + "</div>")
            body.append(f"<div><span class='lbl'>searched as</span> {_searched_html(s)}</div>")
            body.append("<ol class='hits'>" + "".join(
                f"<li><code>{h['doc_id']}</code>{' <span class=ca>Canada</span>' if h['canada'] else ''} "
                f"{html.escape(h['snippet'])}</li>" for h in s["hits"][:5]) + "</ol>")
        elif s["kind"] == "rest":
            body.append(f"<div class='muted'>HTTP {s['status']}</div><pre>{html.escape(json.dumps(s['response'], indent=1))}</pre>")
        diff = "".join(f"<div class='add'>+ {e['kind']}: {html.escape(e['text'])}</div>" for e in s["added"]) + \
               "".join(f"<div class='rem'>&minus; {e['kind']}: {html.escape(e['text'])}</div>" for e in s["removed"])
        mem = "".join(f"<tr><td>{e['kind']}</td><td>{html.escape(e['text'])}</td><td>{e['context'] or '-'}</td>"
                      f"<td>{e['days_left']}</td></tr>" for e in s["memory_after"])
        side = (f"<div class='lbl'>memory changes</div>{diff or '<div class=muted>none</div>'}"
                f"<div class='lbl' style='margin-top:8px'>{html.escape(s['who'])}'s memory after this step</div>"
                + (f"<table><tr><th>kind</th><th>text</th><th>context</th><th>days left</th></tr>{mem}</table>"
                   if mem else "<div class='muted'>empty</div>"))
        cards.append(f"""<section class="step" id="s{s['n']}">
  <header><span class="n">{s['n']}</span><span class="duty" style="background:{color}">{s['duty']}</span>
  <h3>{html.escape(s['title'])}</h3></header>
  <p class="note">{html.escape(s['note'])}</p>
  <div class="grid"><div>{''.join(body)}</div><div class="side">{side}</div></div>
</section>""")
    sm = rec["summary"]["canada_in_top5"]
    legend = "".join(f"<span class='duty' style='background:{c}'>{d}</span> " for d, c in DUTY_COLOR.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Memory replay</title>
<style>
body {{ margin:0; font:14px/1.45 system-ui, Segoe UI, sans-serif; color:#1d2430; background:#f6f7f9; }}
.top {{ position:sticky; top:0; z-index:2; background:#1d2430; color:#fff; padding:10px 18px; display:flex; gap:10px; align-items:center; }}
.top h1 {{ font-size:17px; margin:0 14px 0 0; }}
.top button {{ background:#2f81f7; color:#fff; border:0; border-radius:5px; padding:5px 11px; cursor:pointer; font:inherit; }}
.intro {{ max-width:1250px; margin:14px auto 4px; padding:0 18px; }}
.step {{ max-width:1250px; margin:12px auto; background:#fff; border:1px solid #d9dde3; border-radius:8px; padding:10px 16px; }}
.step.hidden {{ display:none; }} .step.current {{ box-shadow:0 0 0 3px #2f81f7; }}
header {{ display:flex; align-items:center; gap:10px; }} h3 {{ margin:0; font-size:15px; }}
.n {{ background:#1d2430; color:#fff; border-radius:50%; width:24px; height:24px; display:inline-flex; align-items:center; justify-content:center; font-size:12px; }}
.duty {{ color:#fff; border-radius:9px; padding:1px 8px; font-size:12px; }}
.note {{ color:#57606a; margin:6px 0 8px; }}
.grid {{ display:grid; grid-template-columns: 1.35fr 1fr; gap:16px; }}
.msg.clock {{ background:#fdf2d0; }}
.msg {{ background:#dbeafe; border-radius:10px; padding:6px 10px; margin-bottom:4px; display:inline-block; }}
.muted {{ color:#667085; font-size:12.5px; }} .lbl {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.04em; color:#667085; margin-right:6px; }}
mark {{ background:#fff3a3; }} code {{ font-family:Consolas, monospace; font-size:12.5px; }}
.hits {{ margin:6px 0 0 18px; padding:0; font-size:12.5px; }} .hits li {{ margin:2px 0; }}
.ca {{ background:#d1242f; color:#fff; border-radius:4px; padding:0 4px; font-size:11px; }}
.side {{ border-left:1px solid #e5e7eb; padding-left:14px; font-size:12.5px; }}
.add {{ color:#1a7f37; }} .rem {{ color:#c62828; text-decoration:line-through; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} td, th {{ border-bottom:1px solid #eee; padding:2px 4px; text-align:left; vertical-align:top; }}
pre {{ background:#0f172a; color:#e2e8f0; padding:8px; border-radius:6px; font-size:11.5px; white-space:pre-wrap; max-height:260px; overflow:auto; }}
</style></head><body>
<div class="top"><h1>Memory replay</h1><button onclick="play()">&#9654; Play</button><button onclick="step()">Step</button>
<button onclick="showAll()">Show all</button><button onclick="reset()">Reset</button><span id="pos"></span></div>
<div class="intro">
<p>One recorded session with the FiQA A2A agent (<code>src/a2a/server.py</code>), searching the Docker router, with
no LLM calls. Each card is one A2A message or one <code>/memory</code> call. On the left: what memory read and the
query the agent actually searched (the parts memory added are <mark>highlighted</mark>), then the top passages
(<span class="ca">Canada</span> marks a passage whose full text says Canada, Canadian, RRSP, TFSA or CRA). On the right: the entries
added or removed by the step, and everything kept about the caller afterwards.</p>
<p>Passages mentioning Canada in the top 5 for "{TAXES}": {sm['step 2 (no fact)']} before alice's fact (step 2),
{sm['step 6 (Canada fact)']} with it (step 6), {sm['step 7 (opt-out)']} with <code>memory: false</code> (step 7).
Duties: {legend}. Explained in <code>docs/topics/22_agent_memory.md</code>.</p></div>
{''.join(cards)}
<script>
const steps = [...document.querySelectorAll('.step')]; let i = 0, timer = null;
function show() {{ steps.forEach((s, k) => {{ s.classList.toggle('hidden', k >= i); s.classList.toggle('current', k === i - 1); }});
  document.getElementById('pos').textContent = i + ' / ' + steps.length;
  if (i > 0) steps[i - 1].scrollIntoView({{behavior: 'smooth', block: 'center'}}); }}
function step() {{ if (i < steps.length) {{ i++; show(); }} else clearInterval(timer); }}
function play() {{ clearInterval(timer); if (i >= steps.length) i = 0; timer = setInterval(step, 2500); step(); }}
function showAll() {{ clearInterval(timer); i = steps.length; show(); window.scrollTo(0, 0); }}
function reset() {{ clearInterval(timer); i = 0; show(); }}
showAll();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--router", default="http://127.0.0.1:8000")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        rec = json.loads((OUT / "steps.json").read_text(encoding="utf-8"))
    else:
        rec = record(a.router)
        (OUT / "steps.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "replay.html").write_text(render(rec), encoding="utf-8")
    print("canada in top 5:", rec["summary"]["canada_in_top5"])
    print(f"wrote {(OUT / 'replay.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
