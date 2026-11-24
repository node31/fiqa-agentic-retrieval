"""A2A end to end on the running stack: discovery, delegation between agents, conversation memory, isolation.

    docker compose up -d                       # the router on :8000
    python scripts/a2a_demo.py                 # starts the FiQA A2A agent on :8300 in this process
    -> results/a2a_demo.json

WHAT IT SHOWS
    1. Discovery: a client reads /.well-known/agent-card.json and picks the JSON-RPC interface and a skill.
    2. Delegation: a COORDINATOR agent (this script) gets compound questions ("what is X, and how do I Y?").
       It splits each one with the rule-based splitter, sends every part to the FiQA agent as its own A2A
       task in one contextId, and fuses the answers with reciprocal rank fusion. Coverage@10 on the 100
       compound questions, against sending the whole question as one task. Skill "retrieve": no LLM, free.
    3. Memory: alice tells the agent where she lives, asks a question and a follow-up; the task metadata
       shows what was searched. Then she reads her memory, and deletes it.
    4. Isolation: bob cannot read alice's task, and his search gets nothing from alice's memory.
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
from src.a2a import protocol as P  # noqa: E402
from src.a2a.client import A2AClient, A2AError, doc_ids  # noqa: E402
from src.agent.fake_policies import split_question  # noqa: E402
from src.agent.memory import MemoryStore  # noqa: E402
from src.metrics import coverage_at_k  # noqa: E402

OUT = ROOT / "results" / "a2a_demo.json"


def rrf(rankings: list[list[str]], k: int = 10, c: int = 60) -> list[str]:
    """Reciprocal rank fusion: each list votes 1 / (c + rank) for its docs."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0.0) + 1.0 / (c + r)
    return sorted(score, key=score.get, reverse=True)[:k]


def start_agent(router: str, port: int) -> None:
    import uvicorn

    from src.a2a.server import create_app

    app = create_app(router, llm=None, memory=MemoryStore(), base_url=f"http://127.0.0.1:{port}")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise SystemExit("the A2A agent did not start")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--router", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=8300)
    a = ap.parse_args(argv)
    start_agent(a.router, a.port)
    url = f"http://127.0.0.1:{a.port}"
    res: dict = {}

    # 1. discovery
    coord = A2AClient(url)
    res["card"] = {"name": coord.card["name"], "interface": coord.card["supportedInterfaces"][0],
                   "skills": coord.skills}
    print(f"discovered {coord.card['name']!r} at {coord.rpc_url}, skills {coord.skills}")

    # 2. delegation on the compound set
    items = [json.loads(x) for x in (ROOT / "data" / "compound_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    whole = split = 0.0
    tasks = parts_total = 0
    t0 = time.perf_counter()
    examples = []
    for it in items:
        halves = {h: [d for d, x in it["gold"].items() if x == h] for h in ("a", "b")}
        one = coord.send(it["question"])
        whole += coverage_at_k(doc_ids(one), halves)
        parts = split_question(it["question"])
        ctx = one["contextId"]
        rankings = [doc_ids(coord.send(p, context_id=ctx, memory=False)) for p in parts]
        fused = rrf(rankings)
        split += coverage_at_k(fused, halves)
        tasks += 1 + len(parts)
        parts_total += len(parts)
        if len(examples) < 3:
            examples.append({"question": it["question"], "parts": parts, "coverage_whole": coverage_at_k(doc_ids(one), halves),
                             "coverage_split": coverage_at_k(fused, halves)})
    n = len(items)
    res["delegation"] = {"n": n, "coverage@10_one_task": round(whole / n, 3), "coverage@10_split_and_delegate": round(split / n, 3),
                         "avg_parts": round(parts_total / n, 2), "a2a_tasks": tasks,
                         "wall_s": round(time.perf_counter() - t0, 1), "examples": examples}
    print(f"delegation: Coverage@10 {res['delegation']['coverage@10_one_task']:.2f} as one task -> "
          f"{res['delegation']['coverage@10_split_and_delegate']:.2f} split into {parts_total / n:.2f} tasks each "
          f"({tasks} tasks in {res['delegation']['wall_s']} s)")

    # 3. memory, as alice
    alice = A2AClient(url, token_for("alice"))
    t1 = alice.send("I live in Canada. What is the best account to save for retirement?", k=3)
    t2 = alice.send("can I take money out of it early?", context_id=t1["contextId"], k=3)
    t3 = alice.send("how are capital gains taxed?", k=3)            # a new conversation
    hdr = {"Authorization": f"Bearer {token_for('alice')}"}
    mem = httpx.get(f"{url}/memory", headers=hdr).json()["entries"]
    res["memory"] = {"turns": [{"asked": P.text_of(t["history"][0]), "searched": t["metadata"]["memory"]["query_searched"],
                                "top3": [h["snippet"][:100] for h in t["artifacts"][0]["parts"][0]["data"]["hits"]]}
                               for t in (t1, t2, t3)],
                     "export": [{"kind": e["kind"], "text": e["text"]} for e in mem]}
    for t in res["memory"]["turns"]:
        print(f"alice asked {t['asked']!r}\n   searched {t['searched']!r}")

    # 4. isolation
    bob = A2AClient(url, token_for("bob"))
    try:
        bob.get(t1["id"])
        bob_sees = True
    except A2AError as e:
        bob_sees = e.code != P.TASK_NOT_FOUND
    bob_hdr = {"Authorization": f"Bearer {token_for('bob')}"}
    bob_before = len(httpx.get(f"{url}/memory", headers=bob_hdr).json()["entries"])
    tb = bob.send("how are capital gains taxed?", k=3)
    res["isolation"] = {"bob_can_read_alices_task": bob_sees,
                        "bob_searched": tb["metadata"]["memory"]["query_searched"],
                        "bob_memory_entries_before_his_search": bob_before}
    res["memory"]["forget"] = httpx.delete(f"{url}/memory", headers=hdr).json()
    res["memory"]["after_forget"] = len(httpx.get(f"{url}/memory", headers=hdr).json()["entries"])
    print(f"bob can read alice's task: {bob_sees}; bob searched {res['isolation']['bob_searched']!r}; "
          f"alice forgot {res['memory']['forget']['deleted']} entries")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
