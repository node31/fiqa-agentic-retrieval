"""Build one page per cluster type in replays/, with a left nav over its eight recorded scenario replays.

    python scripts/scenario_pages.py      # reads scenarios/SimpleScenarios/results/<cluster>/<scenario>/, no cluster needed

Writes replays/scenarios_sync_tiny.html, scenarios_async_tiny.html, scenarios_sync_full.html and
scenarios_async_full.html. Each page holds, for every scenario: what the CSV says it is about (its header comment),
the run's summary from console.txt (expectations, write and search times, RESULT), and the recorded animated replay
itself. The replay is copied into the page as a string and shown in an iframe (``srcdoc``), so a page works when it
is opened straight from disk and the 32 recorded replay.html files are not touched. A tab row at the top switches
between the four cluster types; the nav, its colour and the prev / next keys work as in every page in replays/.
Two links in the top card open the current scenario's replay on a page of its own, for a bigger view: "original
file" is the recorded replay.html under scenarios/ (a relative link, so it needs the whole repo), and "this copy in
a new tab" opens the copy held in this page (it works even when replays/ is copied on its own).

    >>> describe("# Scenario 2: ReadAfterWrite\\n#\\n# Ann writes, Bob reads.\\nt,who\\n")
    ('ReadAfterWrite', 'Ann writes, Bob reads.')
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_style import CATEGORIES, NAV_CSS, nav_html  # noqa: E402

SRC = ROOT / "scenarios" / "SimpleScenarios"
OUT = ROOT / "replays"
CLUSTERS = [("sync_tiny", "SimpleSyncTinyCluster", "sync · tiny"), ("async_tiny", "SimpleAsyncTinyCluster", "async · tiny"),
            ("sync_full", "SimpleSyncFullCluster", "sync · full"), ("async_full", "SimpleAsyncFullCluster", "async · full")]
ABOUT = {"sync": "a write is applied (embedded, indexed) before the ack, so the next search sees it",
         "async": "a write is acked once it is in the WAL; background workers apply it, within 1 s",
         "tiny": "a small corpus, for fast runs", "full": "the real corpus, 57,638 passages"}
TITLES = {"1": "read", "2": "read after write", "3": "write, then delete", "4": "read, migrate, read",
          "5": "write, migrate, read", "6": "write during a migration", "7": "write during a migration, then roll back",
          "8": "idempotent writes"}


def describe(csv_text: str) -> tuple[str, str]:
    """(short title, description) from a scenario CSV's leading # comment block: the first paragraph after the
    title line (the later paragraphs are run commands and the column format)."""
    lines = [ln[1:].strip() for ln in csv_text.splitlines() if ln.startswith("#")]
    head = lines[0].split(":", 1)[-1].strip() if lines else ""
    paras, cur = [], []
    for ln in lines[1:]:
        if ln:
            cur.append(ln)
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return head, (paras[0] if paras else "")


def summary(console: str) -> tuple[str, str]:
    """(RESULT word, the summary block) from a run's console.txt."""
    block = console[console.find("=== SUMMARY"):] if "=== SUMMARY" in console else console[-1500:]
    result = next((ln.split()[-1] for ln in block.splitlines() if ln.startswith("RESULT")), "?")
    keep = [ln for ln in block.splitlines() if not ln.startswith(("events ", "replay "))]
    return result, "\n".join(keep).strip()


def scenarios() -> list[str]:
    return sorted(p.stem for p in SRC.glob("[0-9]_*.csv"))


CSS = """
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7;color:#1f2328}
header{background:#1e2a3a;color:#fff;padding:10px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:2}
header h1{font-size:19px;margin:0} header button{margin-left:6px;padding:6px 14px;border:0;border-radius:4px;background:#3b82f6;color:#fff;cursor:pointer}
.tabs a{display:inline-block;margin-left:6px;padding:5px 12px;border-radius:14px;font-size:13px;color:#cbd5e1;text-decoration:none;border:1px solid #475569}
.tabs a.cur{background:#2563eb;border-color:#2563eb;color:#fff}
.wrap{display:flex} main{flex:1;padding:14px 22px;min-width:0}
.card{background:#fff;border:1px solid #ddd;border-left:6px solid #2563eb;border-radius:6px;padding:10px 16px;margin-bottom:12px;display:flex;gap:18px}
.card .txt{flex:1;min-width:0} .card h2{font-size:17px;margin:2px 0 4px} .card p{margin:0;font-size:14px;color:#333;line-height:1.45}
.open{margin-top:8px;font-size:13px} .open a{display:inline-block;margin-right:8px;padding:3px 11px;border-radius:12px;background:#dbeafe;color:#1d4ed8;font-weight:600;text-decoration:none;border:1px solid #93c5fd}
.open a:hover{background:#2563eb;color:#fff}
.pass{display:inline-block;background:#dcfce7;color:#166534;font-weight:700;border-radius:10px;padding:1px 9px;font-size:12px;margin-left:6px}
pre{background:#0f172a;color:#e2e8f0;padding:8px 10px;border-radius:6px;font-size:11.5px;margin:0;white-space:pre-wrap;flex:0 0 470px;max-height:150px;overflow:auto}
iframe{width:100%;height:calc(100vh - 250px);min-height:620px;border:1px solid #ddd;border-radius:6px;background:#fff}
aside li .pass{margin-left:4px;font-size:10px;padding:0 6px}
"""


def page(key: str, folder: str, label: str) -> str:
    items, data = [], []
    for name in scenarios():
        run = SRC / "results" / folder / name
        title, about = describe((SRC / f"{name}.csv").read_text(encoding="utf-8"))
        result, summ = summary((run / "console.txt").read_text(encoding="utf-8"))
        n = name.split("_", 1)[0]
        items.append({"title": TITLES.get(n, title or name), "category": "distributed",
                      "badge": f'<span class="pass">{html.escape(result)}</span>'})
        data.append({"n": n, "name": name, "title": f"{TITLES.get(n, title)} ({title})", "about": about,
                     "result": result, "summary": summ, "replay": (run / "replay.html").read_text(encoding="utf-8"),
                     "file": f"../scenarios/SimpleScenarios/results/{folder}/{name}/replay.html"})
    mode, size = key.split("_")
    tabs = "".join(f'<a href="scenarios_{k}.html"{" class=cur" if k == key else ""}>{html.escape(lb)}</a>'
                   for k, _, lb in CLUSTERS)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    color = CATEGORIES["distributed"]["color"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Scenarios · {html.escape(label)}</title>
<style>{CSS}{NAV_CSS}</style></head><body>
<header><h1 id="h">Scenarios · {html.escape(label)}</h1><div class="tabs">{tabs}
<button onclick="go(cur-1)">&larr; prev</button><button onclick="go(cur+1)">next &rarr;</button></div></header>
<div class="wrap"><aside>{nav_html(items)}</aside><main>
<div class="card" style="border-left-color:{color}"><div class="txt"><h2 id="t"></h2><p id="a"></p>
<p style="margin-top:6px;font-size:12px;color:#666"><b>{html.escape(mode)}</b>: {html.escape(ABOUT[mode])} ·
<b>{html.escape(size)}</b>: {html.escape(ABOUT[size])}. The replay below plays the recorded run; its own controls
start, pause and step it.</p>
<div class="open">Open this replay on its own page: <a id="of" target="_blank" rel="noopener">original file &#8599;</a><a href="#" onclick="openCopy();return false">this copy in a new tab &#8599;</a></div></div><pre id="s"></pre></div>
<iframe id="f" title="recorded replay"></iframe></main></div>
<script>
const D = {blob};
let cur = 0;
function go(i) {{
  if (i < 0 || i >= D.length) return;
  document.querySelectorAll("aside li").forEach(l => l.classList.remove("cur"));
  document.getElementById("l" + i).classList.add("cur");
  cur = i; const s = D[i];
  document.getElementById("h").textContent = `Scenarios · {label} · ${{s.n}} of ${{D.length}}`;
  document.getElementById("t").innerHTML = `${{s.n}}. ${{s.title}} <span class="pass">${{s.result}}</span>`;
  document.getElementById("a").textContent = s.about;
  document.getElementById("s").textContent = s.summary;
  document.getElementById("f").srcdoc = s.replay;
  document.getElementById("of").href = s.file;
  location.hash = i + 1;
}}
function openCopy() {{
  const url = URL.createObjectURL(new Blob([D[cur].replay], {{type: "text/html"}}));
  window.open(url, "_blank");
}}
document.addEventListener("keydown", e => {{ if (e.key === "ArrowRight") go(cur + 1); if (e.key === "ArrowLeft") go(cur - 1); }});
go(Math.max(0, (parseInt(location.hash.slice(1)) || 1) - 1));
</script></body></html>"""


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for key, folder, label in CLUSTERS:
        p = OUT / f"scenarios_{key}.html"
        p.write_text(page(key, folder, label), encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
