# 20 · Agents talking to agents (A2A), and agents that remember

Until now every agent in this repo was a Python function: `run_single_agent(question, llm, tool)`. Two things change
when agents become services that other agents use, and that people talk to more than once:

- another agent has to **find** this agent, learn what it can do, and **call** it in a standard way: that is the
  A2A protocol;
- a conversation has a **past**: "how are they taxed?" means nothing without the previous message, and a user who
  said "I live in the UK" yesterday expects the agent to know it today: that is agent memory.

How to run and use both, and every measured number, is in
[docs/design/A2A_AND_MEMORY.md](../design/A2A_AND_MEMORY.md). Code: `src/a2a/` and `src/agent/memory.py`.

## 1. A2A in four objects

A2A (Agent2Agent, version 1.0, https://a2a-protocol.org) is an open protocol for one agent to delegate work to
another without knowing how the other is built. Four objects carry it:

| object | what it is | here |
|---|---|---|
| agent card | a JSON document at `/.well-known/agent-card.json`: name, skills, the endpoint, the protocol version, how to authenticate | `agent_card()` in `src/a2a/server.py` |
| message | one turn: `role` (`ROLE_USER` / `ROLE_AGENT`) and `parts`; a part is text, a file or a JSON `data` object | the question, plus optional `{"remember": ...}` data |
| task | the unit of work: an id, a `contextId`, a `status.state`, `artifacts` (the output) and `history` | one search or agent run |
| artifact | the result, made of parts | doc ids, scores and snippets as JSON, plus a one-line text |

A task moves through states: `SUBMITTED` → `WORKING` → `COMPLETED`, or ends in `FAILED`, `CANCELED` or `REJECTED`;
`INPUT_REQUIRED` and `AUTH_REQUIRED` pause it. Tasks that belong to one conversation share a `contextId`.

How A2A relates to the other protocol people mention, MCP (Model Context Protocol): MCP connects a model to
**tools and data** (here, `search` would be an MCP tool). A2A connects an agent to **another agent**, which has its
own model, its own tools and its own judgement, and may take a while. An agent can use both: MCP for its tools,
A2A to hand a sub-problem to a specialist.

### Toy: a whole A2A exchange in-process

The server takes the search backend as a parameter, so a stub stands in for the router (no network, no LLM):

```python
from fastapi.testclient import TestClient
from src.a2a.server import create_app

def stub_search(token):                      # a fake retrieval service
    return lambda q, strategy, k: [{"doc_id": f"d{i}", "score": 1 / i, "text": f"passage {i} about {q}"} for i in range(1, k + 1)]

tc = TestClient(create_app(search_factory=stub_search, base_url="http://agent.example"))
req = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
       "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "what is a roth ira?"}],
                              "metadata": {"skill": "retrieve", "k": 2}}}}
task = tc.post("/a2a", json=req, headers={"A2A-Version": "1.0"}).json()["result"]["task"]
print(task["status"]["state"], task["artifacts"][0]["parts"][0]["data"]["doc_ids"])
print(tc.post("/a2a", json=req).json()["error"]["message"])
```

```
TASK_STATE_COMPLETED ['d1', 'd2']
A2A version 0.3 is not supported; send the header A2A-Version: 1.0
```

The second line is the spec at work: a request without the `A2A-Version` header means version 0.3, and a 1.0
server must say it doesn't speak it rather than guess.

### What JSON-RPC adds

JSON-RPC 2.0 is the envelope: every request is `{"jsonrpc": "2.0", "id": ..., "method": ..., "params": ...}`, and
the answer has the same `id` and either `result` or `error`. Errors carry numbers a program can act on: -32601 for an
unknown method, -32602 for bad parameters, and A2A's own range from -32001 ("task not found") to -32009 ("version
not supported"). The HTTP status stays 200 for all of them; only a transport problem such as a bad token is an HTTP
error (401).

### Delegation

A2A becomes useful when one agent hands part of a job to another. `scripts/a2a_demo.py` has a coordinator that gets
a compound question, splits it into its parts with a rule, sends each part as its own task in one `contextId`, and
merges the answers. On the 100 compound questions, Coverage@10 (both halves of the question found) went from 0.11
as one task to 0.34 with delegation, without an LLM anywhere. The single agent, whose LLM can search again for the
part its first search missed, reaches 0.40.

### Things the protocol makes you decide

- **Who may see a task.** Here a task belongs to the token's tenant and user. For anyone else it doesn't exist:
  `GetTask` answers "not found", not "forbidden", so a stranger can't even learn that the id is real.
- **Waiting.** `SendMessage` blocks until the task is done unless the client asks for `returnImmediately`; then it
  polls `GetTask`. Streaming (Server-Sent Events) and push notifications are the other two options; this server
  declares both off in its card, and answers the methods with "unsupported".
- **Cancel races.** A cancel can arrive while the agent is still working. The worker attaches its result and marks
  the task completed in one locked step, and only if the task is not already finished, so the cancel wins (a test
  covers this).

## 2. Memory

### Two kinds

| kind | lifetime | example | how it is used |
|---|---|---|---|
| short-term (conversation) | one `contextId` | "I hold some US stocks that pay dividends." then "how are they taxed?" | a follow-up is searched together with the last turn |
| long-term (user) | across conversations, until it expires or is deleted | "I live in the UK" | the country is added to searches that don't name one |

Both serve one purpose here: making the **search query** carry what the conversation already established.
Agent frameworks keep more (summaries of old conversations, preferences, tool results), but every kind raises the
same questions as below.

### Toy: memory step by step

```python
from src.agent.memory import MemoryStore, contextualize, extract_facts, needs_context

print(extract_facts("I'm based in the UK and I want to buy a house"))
print(needs_context("how are they taxed?"), needs_context("how are dividends taxed?"))

m = MemoryStore()                                  # SQLite in RAM
m.remember_fact("acme", "alice", "I live in the UK")
m.remember_turn("acme", "alice", "chat-1", "I hold some US stocks that pay dividends", [])
print(contextualize(m, "acme", "alice", "chat-1", "how are they taxed?")["query"])
print(contextualize(m, "acme", "bob", "chat-1", "how are they taxed?")["query"])        # bob: nothing of alice's
m.remember_turn("acme", "alice", "chat-1", "my email is jane.doe@example.com", [])
print([e["text"] for e in m.export("acme", "alice")])
print(m.forget("acme", "alice"), m.export("acme", "alice"))
```

```
[{'type': 'country', 'value': 'UK', 'text': 'I live in UK'}]
True False
I hold some US stocks that pay dividends how are they taxed? UK
how are they taxed?
['I live in UK', 'I hold some US stocks that pay dividends', 'my email is [EMAIL]']
3 []
```

Read the output line by line: the fact extractor found the country; "they" marks a follow-up, "dividends" doesn't;
alice's follow-up is searched with her last turn and her country; bob, in the same tenant and even with the same
conversation id, gets nothing; the email was redacted before it was stored; and `forget` removed all three entries.

### Three ways to resolve a follow-up

| approach | cost | on 100 generated follow-ups (R@10; the original question scores 0.491) |
|---|---|---|
| nothing (search the follow-up as typed) | 0 | 0.103 |
| rule: prepend the last turn when the follow-up points back | 0 | 0.439 |
| always prepend the last turn | 0 | 0.444 |
| LLM rewrite into one standalone query | 1 LLM call | 0.455 |

The LLM rewrite is the standard answer in conversational search ("query rewriting"), and it is the best here, but
by 0.016 recall. Prepending is free and gets most of the way, because a search engine doesn't mind a slightly
redundant query. The rule exists for the case where prepending hurts: a new, unrelated question in the same
conversation would drag the old topic in.

### Memory is personal data

An agent that remembers is a database of personal data, and the rules from the retrieval service apply to it too:

- **Attribution.** Only an identified user (a verified token) gets memory. Anonymous callers get none.
- **Minimisation.** PII is redacted before it is written. Memory never holds what the audit log may not hold.
- **Isolation.** Every read filters on tenant and user, and a test checks that another user and an anonymous
  caller see nothing.
- **Expiry.** Every entry has an end date (30 days by default) and is never returned after it.
- **Access and deletion.** The user can list everything kept about them and delete one entry or all of it.
- **Opt-out.** One message can turn memory off (`"memory": false`).

A wrong memory is worse than none: the agent would silently search for the wrong country. That is why a new country
replaces the old one instead of adding to it, and why the task metadata shows exactly which memories changed the
search (`memory.query_searched`, `facts_used`, `turn_used`).

## 3. Pitfalls

- A follow-up rule that fires on a new topic drags the old topic into the search. Measure both directions.
- A fact that is true but irrelevant still shifts results: adding "Canada" moved 54% of the top 10 to Canadian
  posts on average, but for "what retirement account should I open?" the top answers barely changed. A shift
  toward a country is not the same as better answers.
- Memory makes results depend on history, so the same question gives different results to different users. Log
  `query_searched` with every result, or bugs can't be reproduced.
- An agent card is a promise: if it says streaming is off, the streaming methods must refuse, not half-work.
