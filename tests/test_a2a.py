"""A2A server + client and agent memory, in-process (no router, no network, no LLM spend).

The search backend is a stub that records every query it receives, so the tests can check exactly what
memory changed about the search.
"""
from __future__ import annotations

import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts.mint_token import token_for
from src.a2a import protocol as P
from src.a2a.client import A2AClient, A2AError, doc_ids
from src.a2a.server import create_app
from src.agent.fake_policies import heuristic_responder
from src.agent.llm import FakeLLM
from src.agent.memory import MemoryStore

BASE = "http://testserver"


class StubSearch:
    def __init__(self, delay: float = 0.0, fail: int | None = None):
        self.queries: list[str] = []
        self.delay, self.fail = delay, fail

    def factory(self, token):
        def fn(query, strategy, k):
            self.queries.append(query)
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                req = httpx.Request("POST", "http://router/search")
                raise httpx.HTTPStatusError("x", request=req,
                                            response=httpx.Response(self.fail, json={"detail": "nope"}, request=req))
            return [{"doc_id": str(i), "score": 1.0 / i, "text": f"passage {i} about {query}"} for i in range(1, k + 1)]
        return fn


def make(search=None, guard=None, llm=None, memory=None):
    search = search or StubSearch()
    app = create_app(llm=llm, memory=memory, search_factory=search.factory, guard_factory=guard, base_url=BASE)
    return TestClient(app), search


def client(tc, user=None):
    return A2AClient(BASE, token_for(user) if user else None, http=tc)


def rpc(tc, method, params, version="1.0", token=None):
    headers = {**({"A2A-Version": version} if version else {}), **({"Authorization": f"Bearer {token}"} if token else {})}
    return tc.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, headers=headers).json()


def test_agent_card_is_discoverable_and_cacheable():
    tc, _ = make()
    r = tc.get("/.well-known/agent-card.json")
    card = r.json()
    for field in ("name", "description", "version", "supportedInterfaces", "capabilities", "defaultInputModes",
                  "defaultOutputModes", "skills"):
        assert field in card
    assert card["supportedInterfaces"][0] == {"url": f"{BASE}/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    assert [s["id"] for s in card["skills"]] == ["retrieve", "single", "multi", "cascade", "answer"]
    assert card["capabilities"]["streaming"] is False and card["capabilities"]["pushNotifications"] is False
    assert tc.get("/.well-known/agent-card.json", headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_retrieve_task_completes_with_a_passages_artifact():
    tc, search = make()
    task = client(tc).send("what is a roth ira?", k=3)
    assert task["status"]["state"] == P.COMPLETED and task["contextId"]
    assert doc_ids(task) == ["1", "2", "3"] and search.queries == ["what is a roth ira?"]
    data = task["artifacts"][0]["parts"][0]["data"]
    assert data["hits"][0]["snippet"].startswith("passage 1") and task["metadata"]["llm_calls"] == 0


@pytest.mark.parametrize("method,params,version,code", [
    ("GetTask", {"id": "x"}, None, P.VERSION_NOT_SUPPORTED),           # no header = 0.3, not supported
    ("GetTask", {"id": "x"}, "0.3", P.VERSION_NOT_SUPPORTED),
    ("NoSuchMethod", {}, "1.0", P.METHOD_NOT_FOUND),
    ("SendMessage", {"message": {"role": "ROLE_USER", "parts": [{"text": "hi"}]}}, "1.0", P.INVALID_PARAMS),  # no messageId
    ("SendMessage", {"message": P.user_message("hi", metadata={"skill": "nope"})}, "1.0", P.INVALID_PARAMS),
    ("SendMessage", {"message": P.user_message("hi", metadata={"k": 500})}, "1.0", P.INVALID_PARAMS),
    ("SendStreamingMessage", {}, "1.0", P.UNSUPPORTED_OPERATION),
    ("CreateTaskPushNotificationConfig", {}, "1.0", P.PUSH_NOT_SUPPORTED),
    ("GetTask", {"id": "missing"}, "1.0", P.TASK_NOT_FOUND),
    ("ListTasks", {"status": "TASK_STATE_RUNNING"}, "1.0", P.INVALID_PARAMS),
])
def test_protocol_errors(method, params, version, code):
    tc, _ = make()
    out = rpc(tc, method, params, version)
    assert out["error"]["code"] == code and out["id"] == 1


def test_malformed_requests():
    tc, _ = make()
    assert tc.post("/a2a", content=b"{not json", headers={"A2A-Version": "1.0"}).json()["error"]["code"] == P.PARSE_ERROR
    batch = tc.post("/a2a", json=[{"jsonrpc": "2.0", "id": 1, "method": "GetTask"}], headers={"A2A-Version": "1.0"})
    assert batch.json()["error"]["code"] == P.INVALID_REQUEST
    assert tc.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {}},
                   headers={"A2A-Version": "1.0", "Authorization": "Bearer forged.token.here"}).status_code == 401


def test_tasks_belong_to_their_caller():
    tc, _ = make()
    alice, bob, anon = client(tc, "alice"), client(tc, "bob"), client(tc)
    t = alice.send("what is a roth ira?")
    assert alice.get(t["id"])["id"] == t["id"]
    for other in (bob, anon):                       # another caller's task looks exactly like a missing one
        with pytest.raises(A2AError) as e:
            other.get(t["id"])
        assert e.value.code == P.TASK_NOT_FOUND
        assert other.list()["totalSize"] == 0
    assert alice.list(context_id=t["contextId"])["totalSize"] == 1


def test_return_immediately_then_poll_and_cancel():
    tc, _ = make(StubSearch(delay=0.4))
    c = client(tc)
    t = c.send("what is a roth ira?", wait=False)
    assert t["status"]["state"] in (P.SUBMITTED, P.WORKING)
    assert c.wait(t["id"])["status"]["state"] == P.COMPLETED
    with pytest.raises(A2AError) as e:              # finished: nothing to cancel
        c.cancel(t["id"])
    assert e.value.code == P.TASK_NOT_CANCELABLE
    t2 = c.send("how do I buy a house?", wait=False)
    assert c.cancel(t2["id"])["status"]["state"] == P.CANCELED
    time.sleep(0.6)                                 # the work finishes afterwards, but the cancel wins
    assert c.get(t2["id"])["status"]["state"] == P.CANCELED and c.get(t2["id"])["artifacts"] == []


def test_follow_up_messages_must_use_a_new_task_in_the_same_context():
    tc, _ = make()
    c = client(tc)
    t = c.send("what is a roth ira?")
    out = rpc(tc, "SendMessage", {"message": P.user_message("and early withdrawals?", task_id=t["id"])})
    assert out["error"]["code"] == P.UNSUPPORTED_OPERATION      # the task is finished
    out = rpc(tc, "SendMessage", {"message": P.user_message("x", context_id="other", task_id=t["id"])})
    assert out["error"]["code"] == P.INVALID_PARAMS             # contextId does not match the task's


def test_llm_skill_runs_and_guardrails_refuse_before_the_llm():
    calls = []

    def responder(system, user):
        calls.append(system.split("\n", 1)[0])
        return heuristic_responder(system, user)

    tc, _ = make(llm=FakeLLM(responder))
    t = client(tc).send("what is a roth ira?", skill="single")
    assert t["status"]["state"] == P.COMPLETED and t["metadata"]["llm_calls"] >= 1 and calls
    calls.clear()
    guard = lambda tok: lambda text: {"allowed": False, "findings": [{"rule": "harmful_intent", "category": "fraud"}]}  # noqa: E731
    tc, search = make(llm=FakeLLM(responder), guard=guard)
    t = client(tc).send("how do I launder money?", skill="single")
    assert t["status"]["state"] == P.REJECTED and "fraud" in P.text_of(t["status"]["message"])
    assert calls == [] and search.queries == []                 # no LLM call, no search


def test_answer_skill_writes_an_answer_from_what_the_llm_may_see():
    seen = []

    def responder(system, user):
        seen.append(user)
        return heuristic_responder(system, user)

    redacting = lambda tok: lambda text: {"allowed": True, "findings": [], "text": text.replace("a@b.co", "[EMAIL]")}  # noqa: E731
    tc, _ = make(llm=FakeLLM(responder), guard=redacting)
    t = client(tc).send("mail a@b.co: what is a roth ira?", skill="answer", k=2)
    assert t["status"]["state"] == P.COMPLETED and [a["name"] for a in t["artifacts"]] == ["passages", "answer"]
    assert P.text_of(t["artifacts"][1]).startswith("(fake answer) From [1]: passage 1 about")
    assert t["metadata"]["llm_calls"] == 1 and t["metadata"]["llm_saw"] == seen[0]
    assert seen[0].splitlines()[0] == "Question: mail [EMAIL]: what is a roth ira?"   # the question as the guard returned it
    assert "[2] passage 2 about" in seen[0]


def test_every_llm_skill_works_from_the_redacted_question():
    # The gap security replay case 6 used to show: single / multi / cascade sent the user's raw words to the LLM.
    redacting = lambda tok: lambda text: {"allowed": True, "findings": [], "pii_redacted": {"EMAIL": 1},  # noqa: E731
                                          "text": text.replace("a@b.co", "[EMAIL]")}
    for skill in ("single", "multi", "cascade"):
        seen = []

        def responder(system, user):
            seen.append(user)
            return heuristic_responder(system, user)

        tc, search = make(llm=FakeLLM(responder), guard=redacting)
        t = client(tc).send("mail a@b.co: what is a roth ira, and how do I open one?", skill=skill)
        assert t["status"]["state"] == P.COMPLETED, skill
        assert seen and not any("a@b.co" in u for u in seen), skill                 # the LLM never read it
        assert search.queries and not any("a@b.co" in q for q in search.queries), skill
        assert t["metadata"]["pii_redacted"] == {"EMAIL": 1} and "[EMAIL]" in t["metadata"]["memory"]["query_searched"]


def test_answer_skill_reads_web_results_when_asked():
    from src.agent.tools import Hits

    asked, seen = [], []
    pages = [{"title": "Chair of the Federal Reserve", "url": "https://en.wikipedia.org/wiki/Chair_of_the_Federal_Reserve",
            "snippet": "The chair is the head of the Board of Governors.", "source": "web"}]

    def factory(tok):
        def fn(query, strategy, k, web=False):
            asked.append(web)
            hits = [{"doc_id": "1", "score": 0.5, "text": "an old forum passage"}]
            return Hits(hits, pages if web else [], {"reason": "requested"} if web else None)
        return fn

    stub = StubSearch()
    stub.factory = factory

    def responder(system, user):
        seen.append(user)
        return heuristic_responder(system, user)

    tc, _ = make(stub, llm=FakeLLM(responder))
    t = client(tc).send("who chairs the Fed?", skill="answer", k=1)          # no web: same as before
    assert asked == [False] and [a["name"] for a in t["artifacts"]] == ["passages", "answer"] and "web" not in t["metadata"]
    out = rpc(tc, "SendMessage", {"message": P.user_message("who chairs the Fed?",
                                                            metadata={"skill": "answer", "k": 1, "web": True})})
    t = out["result"]["task"]
    assert asked == [False, True] and [a["name"] for a in t["artifacts"]] == ["passages", "answer", "web"]
    assert seen[-1] == t["metadata"]["llm_saw"] and "[W1] Chair of the Federal Reserve: The chair is the head" in seen[-1]
    assert t["metadata"]["web"]["fallback"] == {"reason": "requested"}
    assert t["metadata"]["web"]["results"][0]["title"] == "Chair of the Federal Reserve"
    for bad in ({"skill": "retrieve", "web": True}, {"skill": "answer", "web": "yes"}):     # only answer reads the web
        assert rpc(tc, "SendMessage", {"message": P.user_message("x", metadata=bad)})["error"]["code"] == P.INVALID_PARAMS


def test_answer_skill_makes_no_llm_call_without_passages():
    empty = StubSearch()
    empty.factory = lambda tok: lambda q, s, k: []
    calls = []
    tc, _ = make(empty, llm=FakeLLM(lambda s, u: calls.append(u) or "x"))
    t = client(tc).send("how do I bake cookies?", skill="answer")
    assert t["status"]["state"] == P.COMPLETED and t["metadata"]["llm_calls"] == 0 and calls == []
    assert "no LLM call" in P.text_of(t["artifacts"][1])


def test_llm_skill_without_an_llm_fails_cleanly():
    tc, _ = make()
    t = client(tc).send("what is a roth ira?", skill="cascade")
    assert t["status"]["state"] == P.FAILED and "no LLM" in P.text_of(t["status"]["message"])


@pytest.mark.parametrize("status,state", [(401, P.AUTH_REQUIRED), (403, P.REJECTED), (429, P.REJECTED),
                                          (500, P.FAILED)])
def test_router_errors_become_task_states(status, state):
    tc, _ = make(StubSearch(fail=status))
    assert client(tc).send("what is a roth ira?")["status"]["state"] == state


# ------------------------------------------------------------------------------------- memory
def test_memory_carries_the_conversation_and_the_users_facts():
    mem = MemoryStore()
    tc, search = make(memory=mem)
    alice = client(tc, "alice")
    t1 = alice.send("I live in Canada. What is a registered retirement account?")
    t2 = alice.send("can I withdraw from it early?", context_id=t1["contextId"])
    assert search.queries[-1] == "I live in Canada. What is a registered retirement account? can I withdraw from it early?"
    assert t2["metadata"]["memory"]["turn_used"]
    t3 = alice.send("what retirement account should I open?")                 # new conversation: the fact stays
    assert search.queries[-1] == "what retirement account should I open? Canada"
    assert t3["metadata"]["memory"]["facts_used"] == ["I live in Canada"]
    client(tc, "bob").send("what retirement account should I open?")        # bob: nothing of alice's
    client(tc).send("what retirement account should I open?")               # anonymous: no memory at all
    assert search.queries[-2:] == ["what retirement account should I open?"] * 2
    kept = len(mem.export("acme", "alice"))
    alice.send("what retirement account should I open?", memory=False)       # opt out per message
    assert search.queries[-1] == "what retirement account should I open?"
    assert len(mem.export("acme", "alice")) == kept                           # ...and nothing written


def test_memory_api_export_redacts_and_forget_deletes():
    mem = MemoryStore()
    tc, search = make(memory=mem)
    alice = client(tc, "alice")
    reply = alice.send("", remember="I'm from the UK")
    assert reply["role"] == "ROLE_AGENT" and "I live in UK" in P.text_of(reply)
    alice.send("my email is jane.doe@example.com, what is an ISA?")
    h = {"Authorization": f"Bearer {token_for('alice')}"}
    entries = tc.get("/memory", headers=h).json()["entries"]
    assert [e["kind"] for e in entries] == ["fact", "turn"]
    assert "jane.doe" not in str(entries) and "[EMAIL]" in entries[1]["text"]     # stored redacted
    assert tc.get("/memory", headers={"Authorization": f"Bearer {token_for('bob')}"}).json()["entries"] == []
    assert tc.get("/memory").status_code == 401
    assert tc.delete(f"/memory/{entries[0]['id']}", headers=h).json() == {"deleted": 1}
    assert tc.delete("/memory", headers=h).json() == {"deleted": 1}
    assert tc.get("/memory", headers=h).json()["entries"] == []


def test_memory_expires_and_a_new_country_replaces_the_old():
    clock = [1000.0]
    mem = MemoryStore(ttl_days=1, now=lambda: clock[0])
    mem.remember_fact("acme", "alice", "I live in Canada")
    mem.remember_fact("acme", "alice", "I moved to India")
    assert [f["data"]["value"] for f in mem.facts("acme", "alice")] == ["India"]
    clock[0] += 86400 + 1
    assert mem.export("acme", "alice") == []
    with pytest.raises(ValueError):
        mem.remember_fact("", "alice", "I live in Canada")       # no identity, no memory


def test_forget_removes_the_text_from_the_file(tmp_path):
    path = tmp_path / "memory.sqlite"
    mem = MemoryStore(str(path))
    mem.remember_fact("acme", "alice", "my note zebra-quokka-42")
    assert mem.forget("acme", "alice") == 1
    mem.db.close()
    assert b"zebra-quokka-42" not in path.read_bytes()      # overwritten, not just unlinked


def test_concurrent_tasks_do_not_mix_up():
    tc, search = make(StubSearch(delay=0.05))
    c = client(tc)
    out = {}

    def ask(q):
        out[q] = c.send(q, k=2)

    threads = [threading.Thread(target=ask, args=(f"question {i}",)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(t["status"]["state"] == P.COMPLETED for t in out.values())
    assert {q: t["artifacts"][0]["parts"][0]["data"]["hits"][0]["snippet"] for q, t in out.items()} == \
        {q: f"passage 1 about {q}" for q in out}


def test_coordinator_delegates_each_part_as_the_caller():
    """Two agents: the coordinator (itself an A2A server) splits the question and asks the FiQA agent each part,
    with the caller's token, so the FiQA agent's tasks belong to alice, and bob sees neither agent's tasks."""
    from src.a2a.coordinator import create_coordinator_app

    search = StubSearch()
    fiqa = TestClient(create_app(memory=MemoryStore(), search_factory=search.factory, base_url=BASE))
    coord = TestClient(create_coordinator_app(BASE, base_url=BASE, http_factory=lambda tok: fiqa))
    alice = A2AClient(BASE, token_for("alice"), http=coord)
    assert alice.card["name"] == "FiQA coordinator" and alice.skills == ["delegate"]
    t = alice.send("what is a roth ira, and how do I buy my first home?", k=3)
    assert t["status"]["state"] == P.COMPLETED and len(doc_ids(t)) == 3
    assert search.queries == ["what is a roth ira", "how do I buy my first home"]   # one FiQA task per part
    with pytest.raises(A2AError) as e:
        A2AClient(BASE, token_for("bob"), http=coord).get(t["id"])
    assert e.value.code == P.TASK_NOT_FOUND
    assert len(A2AClient(BASE, token_for("alice"), http=fiqa).list()["tasks"]) == 2
    assert A2AClient(BASE, token_for("bob"), http=fiqa).list()["tasks"] == []
    with pytest.raises(A2AError) as e:                     # the coordinator only offers its own skill
        alice.send("what is a roth ira?", skill="retrieve")
    assert e.value.code == P.INVALID_PARAMS


def test_ui_page_and_demo_logins_are_off_unless_asked_for():
    tc, _ = make()
    assert "SendMessage" in tc.get("/ui").text
    assert tc.get("/ui/config").json()["demo_logins"] == []
    assert tc.post("/ui/login", json={"user": "alice"}).status_code == 403

    app = create_app(memory=MemoryStore(":memory:"), search_factory=StubSearch().factory, base_url=BASE,
                     demo_logins=("alice", "bob"))
    tc = TestClient(app)
    tok = tc.post("/ui/login", json={"user": "bob"}).json()["token"]
    assert tc.get("/memory", headers={"Authorization": f"Bearer {tok}"}).json()["tenant"] == "globex"
    assert tc.post("/ui/login", json={"user": "admin"}).status_code == 403   # never an admin token
