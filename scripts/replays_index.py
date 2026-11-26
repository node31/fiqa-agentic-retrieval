"""Write replays/index.html: one page that lists every recorded replay in replays/, grouped by category.

    python scripts/replays_index.py

Every page it links to opens straight from disk (no server): the data is inside each page.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_style import CATEGORIES  # noqa: E402

OUT = ROOT / "replays"
PAGES = [
    ("distributed", "scenarios_sync_tiny.html", "Scenarios: sync, tiny corpus",
     "The eight scripted scenarios (reads, writes, deletes, a model migration with and without a rollback, "
     "idempotent writes) on a cluster that applies every write before acking it.", "scripts/run_simple_scenarios.py"),
    ("distributed", "scenarios_async_tiny.html", "Scenarios: async, tiny corpus",
     "The same eight, with writes acked from the WAL and applied by background workers within 1 s.",
     "scripts/run_simple_scenarios.py"),
    ("distributed", "scenarios_sync_full.html", "Scenarios: sync, full corpus (57,638 passages)",
     "The eight scenarios on the real corpus, sync writes.", "scripts/run_simple_scenarios.py"),
    ("distributed", "scenarios_async_full.html", "Scenarios: async, full corpus",
     "The eight scenarios on the real corpus, async writes.", "scripts/run_simple_scenarios.py"),
    ("security", "security.html", "Security replay: 14 cases, each protection off and on",
     "Who is calling and what they may read and write (tenants), bad tokens, harmful questions, prompt injection, "
     "off-topic abuse, PII in questions and answers, and the rate limit. The nav groups the cases by category: "
     "Agentic Security, Agentic Privacy and Agentic Cost Optimization.", "scripts/security_replay.py"),
    ("extensions", "a2a.html", "A2A replay: two agents and an LLM",
     "A coordinator agent delegates to the FiQA agent over A2A; every message, search and LLM call is an arrow "
     "coloured by who sent it, with memory tags.", "scripts/a2a_two_agents.py"),
    ("extensions", "memory.html", "Memory replay: 16 steps with a real LLM",
     "Attribution, isolation, expiry, access, deletion and opt-out, with what retrieval saw, what the LLM read and "
     "what it answered at every step.", "scripts/memory_replay.py"),
]


def build() -> str:
    cats = []
    for key, meta in CATEGORIES.items():
        rows = [p for p in PAGES if p[0] == key]
        also = [p for p in PAGES if p[0] == "security" and key in ("privacy", "cost")]
        if not rows and not also:
            continue
        cards = "".join(
            f'<a class="card" href="{f}" style="border-left-color:{meta["color"]}"><b>{html.escape(t)}</b>'
            f'<span>{html.escape(d)}</span><code>python {html.escape(s)}</code></a>' for _, f, t, d, s in rows)
        if also:
            cards += (f'<a class="card" href="security.html" style="border-left-color:{meta["color"]}"><b>In the '
                      f'security replay</b><span>The {html.escape(meta["label"])} cases sit in their own group of '
                      f'its nav.</span></a>')
        cats.append(f'<section><h2 style="background:{meta["light"]};color:{meta["color"]}">{html.escape(meta["label"])}'
                    f'<small>{html.escape(meta["what"])}</small></h2><div class="cards">{cards}</div></section>')
    cats.append('<section><h2 style="background:#e2e8f0;color:#475569">Using the system<small>how a person, a script, '
                'a terminal or an agent calls the router</small></h2><div class="cards">'
                '<a class="card" href="ways_to_call.html" style="border-left-color:#475569"><b>Ways to call the router: '
                'Swagger, demo_call, search.py, ask.py</b><span>The same question four ways, with every HTTP call '
                'captured on its way to the router; all four get the same top 3.</span>'
                '<code>python scripts/ways_to_call_replay.py</code></a></div></section>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Replays</title><style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7;color:#1f2328}}
header{{background:#1e2a3a;color:#fff;padding:16px 28px}} header h1{{margin:0;font-size:22px}}
header p{{margin:6px 0 0;color:#cbd5e1;font-size:14px}}
main{{max-width:1100px;margin:18px auto;padding:0 20px}}
h2{{font-size:16px;padding:8px 14px;border-radius:6px;margin:22px 0 10px}} h2 small{{font-weight:400;margin-left:10px;font-size:13px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.card{{display:block;background:#fff;border:1px solid #ddd;border-left:6px solid;border-radius:6px;padding:12px 16px;
text-decoration:none;color:inherit}} .card:hover{{box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.card b{{display:block;font-size:15px;margin-bottom:4px}} .card span{{display:block;font-size:13.5px;color:#444;line-height:1.45}}
.card code{{display:inline-block;margin-top:8px;font-size:12px;color:#555;background:#f3f4f6;padding:1px 6px;border-radius:4px}}
</style></head><body><header><h1>Recorded replays</h1>
<p>Every page opens straight from this folder (no server). Each has a nav on the left; the arrow keys move between
its steps. Colours are the five categories used everywhere in this repository.</p></header>
<main>{''.join(cats)}</main></body></html>"""


def main() -> int:
    (OUT / "index.html").write_text(build(), encoding="utf-8")
    print(f"wrote {(OUT / 'index.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
