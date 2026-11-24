# 22 · Agents that remember

A conversation has a **past**: "how are they taxed?" means nothing without the previous message, and a user who said
"I live in the UK" yesterday expects the agent to know it today. This topic explains agent memory with a toy you can
run, the rules that apply because memory is personal data, how the A2A server ([21_a2a.md](21_a2a.md)) uses it, and
what it measured on follow-up questions and on a remembered country.

Code: `src/agent/memory.py`, the memory part of `src/a2a/server.py`, and `scripts/memory_eval.py`.

## 1. Two kinds of memory

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

## 2. Three ways to resolve a follow-up

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

## 3. Memory is personal data

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

## 4. Quick start

With the stack and the A2A server running (see [21_a2a.md](21_a2a.md#2-quick-start)):

| I want to | command |
|---|---|
| continue a conversation | add `--context <the contextId the first answer printed>` |
| tell it a fact about me | `python -m src.a2a.client --remember "I live in Canada" --token $(python -m scripts.mint_token alice)` |
| see / delete my memory | `curl -H "Authorization: Bearer $TOKEN" localhost:8300/memory`, `curl -X DELETE ...` |
| measure memory | `python scripts/memory_eval.py` (`--confirm-spend --llm-rewrite` for the LLM variant) |

| option / variable | default | what it sets |
|---|---|---|
| `--memory-db` / `MEMORY_DB` | `a2a_memory.sqlite` | the memory file; `off` disables memory, `:memory:` keeps it in RAM |
| `MEMORY_TTL_DAYS` | 30 | how long a memory entry lives |

## 5. Memory in the A2A server

`src/agent/memory.py` keeps two kinds of entries in one SQLite table, keyed by tenant and user:

| kind | what | written when | used for |
|---|---|---|---|
| `turn` | a question the user asked, with the doc ids it got, under its `contextId` | a task completes | a follow-up in the same conversation is searched together with the last turn |
| `fact` | something about the user, e.g. "I live in Canada" | the message says it ("I live in", "I'm from", "I moved to", ...), the client sends `remember`, or `POST /memory` | the country is added to every search that doesn't already name it |

**When is a question a follow-up?** `needs_context` looks for words that point back (it, that, they, their, one,
each, ...) or an opening like "what about". On the generated follow-ups below it flagged 97%. Only the last turn is
used, and only for the search: the LLM still sees the question as asked plus the search results.

**Privacy rules.**

- Only identified callers have memory. An anonymous caller has nobody to attribute memory to, and nobody who could
  ask to see or delete it.
- Text is PII-redacted before it is stored: "my email is jane.doe@example.com" is kept as "my email is [EMAIL]".
- Every read filters on (tenant, user); no query crosses users.
- Entries expire after `MEMORY_TTL_DAYS` (30) and are purged on the next write.
- A new country fact replaces the old one (people move).
- `memory: false` in a message's metadata turns memory off for that message.

**The memory API** (REST on the A2A server; A2A itself has no memory API). Each call needs the user's token and
only ever touches that user's entries:

| request | what it does |
|---|---|
| `GET /memory` | everything kept about me, oldest first |
| `POST /memory {"text": "I live in Canada"}` | remember a fact |
| `DELETE /memory/{id}` | forget one entry |
| `DELETE /memory` | forget everything |

## 6. Results

### Conversation memory: follow-up questions

`scripts/memory_eval.py` (`results/memory_eval.json`). Azure split each of the 100 plain dev questions into a first
message that sets the topic and a follow-up that only makes sense after it (`data/followups_plain.jsonl`), e.g.

> Q: Why is gold not a good investment?
> first: "I'm thinking about investing in gold."  follow-up: "Why might it not be a good investment?"

Together the two ask the original question, so its relevance labels still apply. Hybrid search, minilm-ft:

| what is searched | R@10 | nDCG@10 |
|---|---|---|
| the original question (upper bound) | 0.491 | 0.429 |
| the follow-up alone (no memory) | 0.103 | 0.089 |
| memory rule: last turn + follow-up when `needs_context` fires | 0.439 | 0.401 |
| always prepend the last turn | 0.444 | 0.407 |
| an LLM rewrites the follow-up into a standalone query (1 call) | 0.455 | 0.410 |

Without memory a follow-up loses four fifths of its recall. The free rule recovers most of it; the LLM rewrite
gains another 0.016 R@10 for one LLM call per follow-up. The rule's first version missed "one", "their" and "each"
and flagged 92% (R@10 0.421); adding them took it to 97% and 0.439.

### User memory: a country fact

Ten generic questions ("what retirement account should I open?", "how are capital gains taxed?", ...) for a user
whose memory says "I live in <country>". The measure is the share of the top 10 passages that mention the country
(its name or words such as RRSP, ISA, rupee, superannuation). It shows whether the results moved toward the
country, not whether they answer better; there are no labels for that.

| country | without the fact | with the fact |
|---|---|---|
| Canada | 8% | 54% |
| UK | 9% | 70% |
| India | 2% | 35% |
| Australia | 1% | 25% |

For "how do I file my income taxes?" with "I live in India", the top 3 change from US Form 1040 answers to "Since
you are living in India and earning income not from salary, you must file your tax return under ITR4..." and two
posts about Indian residents and US returns. It doesn't always help at the top: for "what retirement account should
I open?" with Canada, the top 4 hardly change, and the Canadian posts come in further down.

## 7. How it is checked

| what | where |
|---|---|
| memory: follow-up and fact change the search; bob and anonymous get nothing of alice's; per-message opt-out; export is redacted; forget one and all; expiry; a new country replaces the old; no identity, no memory | `tests/test_a2a.py`, doctests in `src/agent/memory.py` |
| follow-up recall and the country shift on the real corpus | `scripts/memory_eval.py` (`results/memory_eval.json`) |

## 8. Pitfalls and what I would do next

- A follow-up rule that fires on a new topic drags the old topic into the search. Measure both directions.
- A fact that is true but irrelevant still shifts results: adding "Canada" moved 54% of the top 10 to Canadian
  posts on average, but for "what retirement account should I open?" the top answers barely changed. A shift
  toward a country is not the same as better answers.
- Memory makes results depend on history, so the same question gives different results to different users. Log
  `query_searched` with every result, or bugs can't be reproduced.
- **The country rule is narrow.** Four countries and a regex. An LLM could extract facts of any kind (age, risk
  appetite, employer plan), with the same redaction and deletion rules; each new kind of fact needs its own check
  that it improves results.
- **Memory changes the search, not the question.** The LLM skills see the question as asked. Passing the facts
  into the prompt as well is easy, and is also where a stale or wrong memory would do the most harm.
