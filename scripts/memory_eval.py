"""Does agent memory help retrieval? Two measurements, both on the real corpus, in-process (no servers).

    python scripts/memory_eval.py --confirm-spend          # first run: Azure writes the follow-ups (~100 calls)
    python scripts/memory_eval.py --confirm-spend --llm-rewrite   # also the LLM-rewrite variant (~100 calls)
    python scripts/memory_eval.py                          # later runs: free (follow-ups are in data/)
    -> results/memory_eval.json

1. CONVERSATION MEMORY (short-term): follow-up questions
    Azure splits each of the 100 plain dev questions (the agent eval's sample) into two chat messages: a
    first message that sets the topic, and a follow-up that only makes sense after it ("I have stock options
    from work." / "How are they taxed when I sell?"). Together they ask the original question, so the
    original's relevance labels still apply (data/followups_plain.jsonl). Searched as:
      original        the question as one message (the upper bound)
      follow-up only  what a system WITHOUT memory searches
      memory rule     src/agent/memory.contextualize: prepend the last turn when the follow-up looks
                      dependent (a pronoun, "what about ...")
      always prepend  prepend the last turn to every message
      LLM rewrite     one LLM call rewrites the follow-up into a standalone query using the last turn
2. USER MEMORY (long-term): a country fact
    Ten generic questions ("what retirement account should I open?") for a user whose memory says "I live
    in <country>", for Canada, the UK, India and Australia. No labels exist for "the right answer for a
    Canadian", so the measure is the share of the top 10 passages that mention the country (its name or
    words like RRSP / ISA / rupee / superannuation; memory.COUNTRIES), with and without the fact.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_eval import load_sets  # noqa: E402
from src.agent.memory import COUNTRIES, MemoryStore, contextualize, needs_context  # noqa: E402
from src.metrics import ndcg_at_k, recall_at_k  # noqa: E402

FOLLOWUPS = ROOT / "data" / "followups_plain.jsonl"
OUT = ROOT / "results" / "memory_eval.json"
SPLIT_PROMPT = """ROLE: followup_writer
Split the user's personal-finance question into two chat messages, the way a person would type it in a
conversation: a short FIRST message that sets the topic, then a FOLLOWUP that only makes sense after the
first one, because it refers back with words like it, that, this, they or "what about" instead of naming
the topic again. Together the two must ask exactly the original question; drop nothing.
Reply with ONE JSON object and nothing else: {"first": "...", "followup": "..."}"""
REWRITE_PROMPT = """ROLE: query_rewriter
You rewrite the last message of a conversation into ONE standalone search query that keeps every detail
the conversation gives. Reply with ONE JSON object and nothing else: {"query": "..."}"""
GENERIC = ["what retirement account should I open?", "how are capital gains taxed?",
           "how do I file my income taxes?", "should I buy or rent a house?", "how do I build a good credit score?",
           "how can I save for my children's education?", "what is the best way to send money abroad?",
           "how much tax do I pay on dividends?", "how do I start investing in index funds?",
           "what happens to my pension if I change jobs?"]


def followups(confirm_spend: bool) -> list[dict]:
    if FOLLOWUPS.exists():
        return [json.loads(x) for x in FOLLOWUPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not confirm_spend:
        sys.exit(f"{FOLLOWUPS.relative_to(ROOT)} is missing: run once with --confirm-spend (Azure, ~100 calls)")
    from src.agent.llm import get_llm, parse_json

    llm = get_llm("azure", confirm_spend=True)
    rows = []
    for it in load_sets()["plain"]:
        try:
            j = parse_json(llm.complete(SPLIT_PROMPT, it["question"], max_output_tokens=600).text)
            rows.append({"id": it["id"], "question": it["question"], "first": j["first"].strip(),
                         "followup": j["followup"].strip()})
        except Exception as e:  # noqa: BLE001 - a bad reply skips one question, it does not stop the run
            print(f"skip {it['id']}: {e}", flush=True)
    FOLLOWUPS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def mentions(text: str, country: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower().replace("u.k.", "uk")))
    return bool(words & set(COUNTRIES[country]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--llm-rewrite", action="store_true", help="also score an LLM rewrite of each follow-up (Azure)")
    ap.add_argument("--model", default="minilm-ft")
    a = ap.parse_args(argv)
    rows = followups(a.confirm_spend)
    from src.service.router import single_node_router

    router = single_node_router(models=(a.model,))
    loop = asyncio.new_event_loop()

    def search(q: str, k: int = 10) -> list[dict]:
        return loop.run_until_complete(router.search(q, k=k, include_text=True))["results"]

    gold = {it["id"]: {d: 1 for d in it["gold"]} for it in load_sets()["plain"]}
    llm = None
    if a.llm_rewrite:
        if not a.confirm_spend:
            ap.error("--llm-rewrite calls Azure: add --confirm-spend")
        from src.agent.llm import get_llm, parse_json
        llm = get_llm("azure", confirm_spend=True)

    # ---- 1. follow-ups
    conds = ["original", "follow-up only", "memory rule", "always prepend"] + (["LLM rewrite"] if llm else [])
    per, sums = [], {c: [0.0, 0.0] for c in conds}
    for r in rows:
        mem = MemoryStore()
        mem.remember_turn("t", "u", "ctx", r["first"], [])
        queries = {"original": r["question"], "follow-up only": r["followup"],
                   "memory rule": contextualize(mem, "t", "u", "ctx", r["followup"])["query"],
                   "always prepend": f"{r['first']} {r['followup']}"}
        if llm:
            conv = f"Conversation:\nuser: {r['first']}\nuser: {r['followup']}\n\nRewrite the last message."
            queries["LLM rewrite"] = parse_json(llm.complete(REWRITE_PROMPT, conv, max_output_tokens=600).text)["query"]
        row = {"id": r["id"], "needs_context": needs_context(r["followup"]), "queries": queries, "recall@10": {}}
        for c, q in queries.items():
            ids = [h["doc_id"] for h in search(q)]
            rec, nd = recall_at_k(ids, gold[r["id"]]), ndcg_at_k(ids, gold[r["id"]])
            sums[c][0] += rec
            sums[c][1] += nd
            row["recall@10"][c] = round(rec, 4)
        per.append(row)
    n = len(rows)
    follow = {"n": n, "share_flagged_as_follow_up": round(sum(p["needs_context"] for p in per) / n, 3),
              "conditions": {c: {"recall@10": round(s[0] / n, 4), "ndcg@10": round(s[1] / n, 4)} for c, s in sums.items()}}
    print(f"\nfollow-ups (n={n}, {follow['share_flagged_as_follow_up']:.0%} flagged by the rule)")
    for c, m in follow["conditions"].items():
        print(f"  {c:16s} R@10 {m['recall@10']:.3f}  nDCG@10 {m['ndcg@10']:.3f}")

    # ---- 2. country facts
    prefs = {}
    for country in COUNTRIES:
        mem = MemoryStore()
        mem.remember_fact("t", "u", f"I live in {country}")
        base = withf = 0
        for q in GENERIC:
            base += sum(mentions(h.get("text", ""), country) for h in search(q))
            withf += sum(mentions(h.get("text", ""), country) for h in search(contextualize(mem, "t", "u", None, q)["query"]))
        prefs[country] = {"without_memory": round(base / (10 * len(GENERIC)), 3),
                          "with_memory": round(withf / (10 * len(GENERIC)), 3)}
        print(f"  {country:10s} top-10 passages mentioning it: {prefs[country]['without_memory']:.0%} -> "
              f"{prefs[country]['with_memory']:.0%}")
    example_q, example_country = "how do I file my income taxes?", "India"
    mem = MemoryStore()
    mem.remember_fact("t", "u", f"I live in {example_country}")
    example = {"question": example_q, "fact": f"I live in {example_country}",
               "without": [" ".join(h["text"].split())[:120] for h in search(example_q, 3)],
               "with_fact": [" ".join(h["text"].split())[:120] for h in
                             search(contextualize(mem, "t", "u", None, example_q)["query"], 3)]}
    OUT.write_text(json.dumps({"model": a.model, "search": "hybrid, in-process", "follow_ups": follow,
                               "follow_up_examples": [{k: p[k] for k in ("id", "queries", "recall@10")} for p in per[:5]],
                               "country_facts": {"questions": GENERIC, "share_of_top10_mentioning_country": prefs,
                                                 "example": example}},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
