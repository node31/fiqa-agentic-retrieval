"""Web fallback on the live stack: how often it fires, whether the web returns anything, what it adds in latency.

    python scripts/web_fallback_eval.py                  # needs the stack: docker compose up -d
    -> results/web_fallback_eval.json; the toggles are restored at the end

SETS
    dev        the 500 FiQA dev questions: real questions the corpus is built for. The fallback should
               almost never fire on these (the threshold is the 1st percentile of their best score).
    off_topic  the 30 off-topic questions of data/guardrails_redteam.jsonl: the corpus has nothing; this is
               where the "low confidence" trigger should fire.
    newer      finance questions about things after the corpus was written (FiQA's posts end around 2018),
               sent with "web": true, the way an agent asks once its strong tier still found little.
Each: share of questions where the fallback ran, share where the web returned at least one result,
and the extra latency (router-side ``web_fallback.latency_ms``). Plus one example of what is sent for a
question that contains an email address (``pii_example.query_sent``).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scripts.mint_token import token_for  # noqa: E402
from src.beir_data import load_split  # noqa: E402

OUT = ROOT / "results" / "web_fallback_eval.json"
NEWER = ["what is the 401k contribution limit for 2026", "how does the SECURE 2.0 Act change required minimum distributions",
         "what happened to Silicon Valley Bank in 2023", "how do spot bitcoin ETFs work",
         "what is the FDIC insurance limit after the 2023 bank failures", "how does the inflation reduction act EV tax credit work",
         "what is a 529 to Roth IRA rollover", "what are I bonds and why were they popular in 2022",
         "how did the 2022 interest rate hikes affect mortgage rates", "what is the student loan SAVE plan"]


def run(c: httpx.Client, h: dict, questions: list[str], web: bool) -> dict:
    fired = with_results = 0
    lat = []
    for q in questions:
        r = c.post("/search", json={"query": q, "k": 3, "web": web}, headers=h)
        r.raise_for_status()
        body = r.json()
        if "web_fallback" in body:
            fired += 1
            lat.append(body["web_fallback"]["latency_ms"])
            with_results += bool(body["web_results"])
    n = len(questions)
    return {"n": n, "fired": fired, "fired_share": round(fired / n, 3),
            "web_returned_results": with_results,
            "web_latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "web_latency_ms_max": round(max(lat), 1) if lat else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args(argv)
    c = httpx.Client(base_url=a.url, timeout=60)
    admin = {"Authorization": f"Bearer {token_for('admin')}"}
    original = {k: v["value"] for k, v in c.get("/admin/security", headers=admin).json().items()}
    try:
        c.post("/admin/security", json={k: "off" for k in original} | {"web_fallback": "on"}, headers=admin).raise_for_status()
        dev, _ = load_split("dev")
        red = [json.loads(x) for x in (ROOT / "data" / "guardrails_redteam.jsonl").read_text(encoding="utf-8").splitlines() if x]
        res = {"provider": None,
               "dev": run(c, {}, list(dev.values()), web=False),
               "off_topic": run(c, {}, [r["text"] for r in red if r["group"] == "off_topic"], web=False),
               "newer_requested": run(c, {}, NEWER, web=True)}
        sample = c.post("/search", json={"query": NEWER[0], "k": 3, "web": True}).json()
        res["provider"] = sample["web_fallback"]["provider"]
        res["example"] = {"question": NEWER[0], "web_results": sample["web_results"]}
        # what leaves the system: the question is PII-redacted before it goes to the provider
        pii_q = "my email is jane.doe@example.com, what is a roth ira"
        res["pii_example"] = {"question": pii_q, "query_sent": c.post(
            "/search", json={"query": pii_q, "k": 3, "web": True}).json()["web_fallback"]["query_sent"]}
        for k in ("dev", "off_topic", "newer_requested"):
            print(k, res[k], flush=True)
    finally:
        c.post("/admin/security", json=original, headers=admin)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
