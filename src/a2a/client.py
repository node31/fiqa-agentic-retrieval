"""An A2A client: discover an agent from its card, send it a question, read the task back.

    python -m src.a2a.client "what is a roth ira?"                             # skill retrieve, no token
    python -m src.a2a.client "should I pay off my mortgage early?" --skill single --token $(python -m scripts.mint_token alice)
    python -m src.a2a.client --card                                            # print the agent card

    >>> c = A2AClient.__new__(A2AClient)              # no network in a doctest: just the request shape
    >>> c.rpc_url, c.headers, c._next = "http://x/a2a", {"A2A-Version": "1.0"}, 0
    >>> req = c._request("GetTask", {"id": "t1"})
    >>> req["jsonrpc"], req["method"], req["params"]
    ('2.0', 'GetTask', {'id': 't1'})

WHY A CLIENT READS THE CARD FIRST
    The card says where the JSON-RPC endpoint is (``supportedInterfaces``), which protocol version it speaks,
    what skills exist and how to authenticate. A client that hard-codes "/a2a" breaks the day the agent
    moves; one that reads the card follows it. (Spec 8.3.2: take the first interface you support.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

from src.a2a import protocol as P


class A2AError(Exception):
    def __init__(self, error: dict):
        super().__init__(f"{error.get('code')}: {error.get('message')}")
        self.code, self.error = error.get("code"), error


class A2AClient:
    def __init__(self, agent_url: str, token: str | None = None, timeout: float = 120, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=timeout)   # tests pass FastAPI's TestClient (also an httpx.Client)
        self.card = self.http.get(f"{agent_url.rstrip('/')}/.well-known/agent-card.json").raise_for_status().json()
        iface = next((i for i in self.card["supportedInterfaces"] if i["protocolBinding"] == "JSONRPC"), None)
        if iface is None:
            raise A2AError({"code": None, "message": "the agent offers no JSON-RPC interface"})
        if iface["protocolVersion"].split(".")[0] != P.PROTOCOL_VERSION.split(".")[0]:
            raise A2AError({"code": None, "message": f"the agent speaks A2A {iface['protocolVersion']}, "
                                                     f"this client {P.PROTOCOL_VERSION}"})
        self.rpc_url = iface["url"]
        self.headers = {"A2A-Version": P.PROTOCOL_VERSION, **({"Authorization": f"Bearer {token}"} if token else {})}
        self._next = 0

    @property
    def skills(self) -> list[str]:
        return [s["id"] for s in self.card["skills"]]

    def _request(self, method: str, params: dict) -> dict:
        self._next += 1
        return {"jsonrpc": "2.0", "id": self._next, "method": method, "params": params}

    def call(self, method: str, params: dict) -> dict:
        r = self.http.post(self.rpc_url, json=self._request(method, params), headers=self.headers)
        r.raise_for_status()   # 401: bad token (transport-level), the only non-200 this server sends
        body = r.json()
        if "error" in body:
            raise A2AError(body["error"])
        return body["result"]

    def send(self, question: str, skill: str = "retrieve", context_id: str | None = None, k: int = 10,
             remember: str | None = None, memory: bool = True, wait: bool = True) -> dict:
        """SendMessage. Returns the Task (or, for a pure "remember" message, the agent's Message)."""
        meta = {"skill": skill, "k": k, **({} if memory else {"memory": False})}
        msg = P.user_message(question, context_id=context_id, data={"remember": remember} if remember else None,
                             metadata=meta)
        out = self.call("SendMessage", {"message": msg, "configuration": {"returnImmediately": not wait}})
        return out.get("task") or out.get("message")

    def get(self, task_id: str, history_length: int | None = None) -> dict:
        return self.call("GetTask", {"id": task_id, **({"historyLength": history_length} if history_length is not None else {})})

    def wait(self, task_id: str, poll_s: float = 0.2, timeout_s: float = 120) -> dict:
        """Poll GetTask until the task is terminal or interrupted (the no-streaming way to wait)."""
        t0 = time.time()
        while True:
            task = self.get(task_id)
            if task["status"]["state"] in P.TERMINAL + (P.INPUT_REQUIRED, P.AUTH_REQUIRED) or time.time() - t0 > timeout_s:
                return task
            time.sleep(poll_s)

    def list(self, context_id: str | None = None, status: str | None = None) -> dict:
        return self.call("ListTasks", {k: v for k, v in (("contextId", context_id), ("status", status)) if v})

    def cancel(self, task_id: str) -> dict:
        return self.call("CancelTask", {"id": task_id})


def doc_ids(task: dict) -> list[str]:
    """The doc ids from a completed task's ``passages`` artifact."""
    for art in task.get("artifacts", []):
        for p in art["parts"]:
            if isinstance(p.get("data"), dict) and "doc_ids" in p["data"]:
                return p["data"]["doc_ids"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--agent", default=os.environ.get("A2A_AGENT_URL", "http://localhost:8300"))
    ap.add_argument("--skill", default="retrieve")
    ap.add_argument("--context", help="contextId: reuse one to continue a conversation")
    ap.add_argument("--remember", help="a fact to remember about you, e.g. 'I live in Canada' (needs --token)")
    ap.add_argument("--token", default=os.environ.get("FIQA_TOKEN"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--card", action="store_true", help="print the agent card and exit")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    c = A2AClient(a.agent, a.token)
    if a.card:
        print(json.dumps(c.card, indent=1))
        return 0
    if not a.question and not a.remember:
        ap.error("give a question (or --remember)")
    try:
        out = c.send(a.question or "", a.skill, a.context, a.k, remember=a.remember)
    except A2AError as e:
        print(f"A2A error {e}")
        return 2
    if out.get("role") == "ROLE_AGENT":           # a direct Message (the "remember" case)
        print(P.text_of(out))
        return 0
    st = out["status"]
    print(f"task {out['id']}  context {out['contextId']}  state {st['state']}")
    if st.get("message"):
        print("  " + P.text_of(st["message"]))
    meta = out.get("metadata", {})
    mem = meta.get("memory") or {}
    if mem.get("turn_used") or mem.get("facts_used"):
        print(f"  memory: searched {mem['query_searched']!r}")
    for art in out.get("artifacts", []):
        for h in next(p["data"]["hits"] for p in art["parts"] if "data" in p)[:5]:
            print(f"  [{h['doc_id']}] {h['snippet'][:110]}")
    print(f"  llm calls {meta.get('llm_calls')}, ${meta.get('usd')}, {meta.get('latency_s')} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
