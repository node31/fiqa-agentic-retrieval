"""One look for every replay page in replays/: the five category colours and the left navigation pane.

Every replay (security, memory, A2A, the sync/async scenario pages) and the V3 deck use the same five categories,
so a colour means the same thing everywhere:

    distributed   Distributed                  sync vs async writes, sharding, model migration
    security      Agentic Security             who is calling (AuthN), what they may do (AuthZ), guardrails, LLM abuse
    privacy       Agentic Privacy              PII redaction, the audit log
    cost          Agentic Cost Optimization    fine-tuning, LLM cost levers, LLM query rewrite, rate limiting
    extensions    Agentic Extensions           agent to agent (A2A), memory, web search

A page builds its nav with ``nav_html(items)``: each item has a ``category`` key, items are grouped under a
coloured heading per category, and an optional ``group`` gives a smaller sub-heading inside it. Red is kept out of
the palette on purpose: replays use it only for errors and refusals.

    >>> [c["label"] for c in CATEGORIES.values()][:2]
    ['Distributed', 'Agentic Security']
    >>> "Agentic Privacy" in nav_html([{"title": "PII", "category": "privacy"}])
    True
"""
from __future__ import annotations

import html

CATEGORIES = {
    "distributed": {"label": "Distributed", "color": "#2563eb", "light": "#dbeafe",
                    "what": "sync vs async writes, sharding, zero-downtime model migration"},
    "security": {"label": "Agentic Security", "color": "#7c3aed", "light": "#ede9fe",
                 "what": "who is calling (AuthN), what they may do (AuthZ), guardrails, LLM abuse"},
    "privacy": {"label": "Agentic Privacy", "color": "#0d9488", "light": "#ccfbf1",
                "what": "PII redaction, a tamper-evident audit log"},
    "cost": {"label": "Agentic Cost Optimization", "color": "#d97706", "light": "#fef3c7",
             "what": "fine-tuning, LLM cost levers, LLM query rewrite, rate limiting"},
    "extensions": {"label": "Agentic Extensions", "color": "#db2777", "light": "#fce7f3",
                   "what": "agent to agent (A2A), memory, web search"},
}
ERROR = "#dc2626"   # errors and refusals only; no category uses it

NAV_CSS = """
aside{width:310px;flex:none;background:#fff;border-right:1px solid #ddd;min-height:calc(100vh - 52px);overflow-y:auto}
aside .cat{margin:12px 10px 4px;padding:5px 10px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:.02em}
aside .cat small{display:block;font-weight:400;opacity:.8;letter-spacing:0;margin-top:1px}
aside h3{font-size:11px;text-transform:uppercase;color:#888;margin:8px 16px 2px}
aside ol{list-style:none;margin:0;padding:0}
aside li{padding:7px 16px;font-size:13px;color:#444;border-left:5px solid transparent;cursor:pointer}
aside li:hover{background:#f6f8fa} aside li.cur{background:#f3f4f6;color:#111;font-weight:600}
aside li b{display:inline-block;width:24px;color:#888}
"""


def nav_order(items: list[dict]) -> list[int]:
    """Item indices in nav order: by category (CATEGORIES order), then as given. Items keep their own numbers.

    >>> nav_order([{"category": "privacy"}, {"category": "security"}, {"category": "privacy"}])
    [1, 0, 2]
    """
    keys = list(CATEGORIES)
    return sorted(range(len(items)), key=lambda i: (keys.index(items[i].get("category", "security")), i))


def nav_html(items: list[dict], current: int | None = None) -> str:
    """The left pane: items grouped by category (CATEGORIES order), then by ``group``. Each <li> gets id="l<i>"
    and onclick="go(<i>)" with i the item's index in ``items``; the number shown is i + 1, so an item keeps its
    number even when the grouping moves it. Pages walk ``nav_order`` for prev / next."""
    out, cat, grp = [], None, None
    for i in nav_order(items):
        it = items[i]
        c = it.get("category", "security")
        meta = CATEGORIES[c]
        if c != cat:
            if cat is not None:
                out.append("</ol>")
            out.append(f'<div class="cat" style="background:{meta["light"]};color:{meta["color"]}">'
                       f'{html.escape(meta["label"])}<small>{html.escape(meta["what"])}</small></div><ol>')
            cat, grp = c, None
        if it.get("group") and it["group"] != grp:
            out.append(f'</ol><h3>{html.escape(it["group"])}</h3><ol>')
            grp = it["group"]
        cur = ' class="cur"' if i == current else ""
        badge = it.get("badge", "")
        out.append(f'<li id="l{i}"{cur} onclick="go({i})" style="border-left-color:{meta["color"]}">'
                   f'<b>{i + 1}</b>{html.escape(it["title"])}{badge}</li>')
    return "".join(out) + ("</ol>" if items else "")


def legend_html(keys: list[str] | None = None) -> str:
    """A one-line legend of the categories (all of them, or ``keys``)."""
    keys = keys or list(CATEGORIES)
    return " ".join(f'<span style="display:inline-block;padding:2px 9px;margin:2px;border-radius:10px;font-size:12px;'
                    f'background:{CATEGORIES[k]["light"]};color:{CATEGORIES[k]["color"]};font-weight:600">'
                    f'{html.escape(CATEGORIES[k]["label"])}</span>' for k in keys)
