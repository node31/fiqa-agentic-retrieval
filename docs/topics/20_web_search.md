# 20 · Web search: answering what the corpus can't

FiQA's forum posts end around 2018 and cover personal finance only. When a question has no good answer in the
corpus, the router can also ask a web provider (Wikipedia by default) and return its results next to the corpus
results. The question leaves the system when that happens, so the feature is off by default, sits behind an admin
toggle like the ones in [17_security.md](17_security.md), and is audited. The cost side of the LLM work is in
[19_llm_cost.md](19_llm_cost.md).

Code: `src/service/websearch.py` (providers, `web_query`), the `web_fallback` function inside `create_app` in
`src/service/router.py`, the `web_fallback` toggle in `src/security/config.py`, and the cascade path of
`scripts/ask.py`, which asks for web results when even the strong tier found little.

## 1. The idea

FiQA's forum posts end around 2018, and the corpus has nothing on bicycles. A retrieval system can say so, or it can
ask a second retriever: here, Wikipedia. Three decisions shape it:

1. **When to ask.** Only when the corpus looks weak: the best dense score is under a threshold (about 1% of real
   questions fall below it), the search returned nothing, or an agent asks explicitly. Asking on every question
   would add 0.6 to 0.9 s and send every question to a third party.
2. **What to send.** The question leaves the system, so personal data must not. Redacting the email is not enough:

   ```python
   >>> from src.service.websearch import web_query
   >>> web_query("my email is [EMAIL], what is a roth ira")
   'what is a roth ira'
   ```

   Sent with the placeholder, "my email is [EMAIL], what is a roth ira" made Wikipedia return pages about the 2016
   election email hacks. Dropping the whole clause that held personal data fixed relevance and leaks less context.
3. **What to trust.** Web text is untrusted input, like a forum post, so it gets the same treatment as passages:
   instruction-like text is cut (prompt injection) and personal data is redacted. Web results are returned in their
   own list, never mixed into the corpus ranking: they have no doc id and no tenant, and no eval can score them.

A guardrail refusal always wins: if `guard_scope` blocked a question as off-topic, the router does not go and look it
up on the web instead.

## 2. Quick start

| I want to | command |
|---|---|
| ask through the cascade, with the web as a last resort | `python scripts/ask.py "how do I fix a flat bicycle tire" --strategy cascade --backend azure --confirm-spend --url http://localhost:8000` (toggle on first, next row) |
| ask the router directly | `curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "what is a 529 to Roth IRA rollover", "k": 3, "web": true}'` |
| try it in the browser | `http://localhost:8000/docs`, tour step 12 |
| switch the web fallback on | `curl -H "Authorization: Bearer $(python -m scripts.mint_token admin)" -X POST localhost:8000/admin/security -H "Content-Type: application/json" -d '{"web_fallback": "on"}'` |
| measure the web fallback on the stack | `python scripts/web_fallback_eval.py` (restores the toggles) |

| variable | default | what it sets |
|---|---|---|
| `WEB_SEARCH_PROVIDER` | `wikipedia` | `wikipedia` (free, no key) or `fake` (tests, no network) |
| `WEB_FALLBACK_MIN` | the `guard_scope` threshold of the active model (0.4421 for minilm-ft) | best dense score below which the corpus counts as "no good answer" |

## 3. Usage and results

### What it does

When the corpus has no good answer, the router (`/search`) also asks a web provider and returns its results in a
separate `web_results` list, never mixed into `results`. It runs when the admin toggle `web_fallback` is on and one
of these holds:

| reason | when |
|---|---|
| `requested` | the caller sent `"web": true` (the cascade CLI does this when even the strong tier picked fewer than 3 docs) |
| `no results` | the search returned nothing |
| `low confidence` | the best dense score is below `WEB_FALLBACK_MIN` (by default the `guard_scope` threshold: about 1% of real dev questions fall below it) |

It never runs for a request a guardrail blocked: `guard_scope=block` means "off-topic, don't answer", and the web
would undo that decision.

The question leaves the system, so: the toggle is off by default and admin-only; the question is PII-redacted
before it is sent; every snippet that comes back goes through the injection neutraliser and PII redaction, whatever
those toggles say; and the audit log records the reason and the URLs.

### How to use it

Switch it on (admin token), then search as usual or with `"web": true`:

```bash
TOKEN=$(python -m scripts.mint_token admin)
curl -H "Authorization: Bearer $TOKEN" -X POST localhost:8000/admin/security \
     -H "Content-Type: application/json" -d '{"web_fallback": "on"}'
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "my email is jane.doe@example.com, what is a roth ira", "k": 3, "web": true}'
```

The response adds (a real response; latency varies):

```json
"web_results": [
  {"title": "Roth IRA", "url": "https://en.wikipedia.org/wiki/Roth_IRA",
   "snippet": "A Roth IRA is an individual retirement account (IRA) under United States law that is generally not taxed upon distribution, ...",
   "source": "web"},
  {"title": "Roth 401(k)", ...},
  {"title": "Individual retirement account", ...}],
"web_fallback": {"reason": "requested", "provider": "wikipedia", "query_sent": "what is a roth ira", "latency_ms": 663.5}
```

`query_sent` is what left the system. The email is gone, and so is the clause around it: the first version sent
"my email is [EMAIL], what is a roth ira", and Wikipedia answered with three pages on the 2016 election email hacks,
because "email" outweighed the question. `web_query` in `src/service/websearch.py` now drops every clause that held
personal data. A low-confidence call also reports the score that triggered it, e.g. for "how do I fix a flat bicycle
tire": `"top_dense": 0.382508, "minimum": 0.4421`. A snippet that had instruction-like text cut out carries
`"injection_removed": <count>`, and a provider failure adds `"error"`.

In `/docs`, tour step 12 and the search examples *Ask the web too*, *Web, with an email* and *Low confidence* do
the same.

The Wikipedia provider (`src/service/websearch.py`) asks the MediaWiki API for the top pages and each page's first
two intro sentences, in one request. Asking for the search snippets instead returned the text around the match,
which on finance pages was often the reference list ("Archived from the original on July 5, 2017..."). Wikimedia
answers 403 to a request without a descriptive User-Agent. Another provider (Tavily, Brave, Bing grounding) is one
class with the same `search` method.

### Results

`scripts/web_fallback_eval.py` on the Docker stack (`results/web_fallback_eval.json`):

| set | questions | fallback ran | web returned results | web latency p50 / max |
|---|---|---|---|---|
| FiQA dev questions (the corpus should answer these) | 500 | 5 (1%) | 4 | 834 / 905 ms |
| off-topic questions from the guardrail red team | 30 | 21 (70%) | 21 | 639 / 909 ms |
| finance questions newer than the corpus, `"web": true` | 10 | 10 | 10 | 664 / 4,743 ms |

The dev rate matches the threshold's design (the 1st percentile of real questions). The 9 off-topic questions that
did not fire score above the threshold; `guard_scope` uses the same threshold, so it lets the same 9 through.
Latency is the router's wait for the provider, added to the search only when the fallback runs. The 4.7 s maximum
was one slow Wikipedia answer; the router gives up after 5 s and returns the corpus results with an `error`.

## 4. How it is checked

| what | where |
|---|---|
| web fallback: off by default, `requested` and `low confidence` triggers, no trigger without a dense score, PII redacted and its clause dropped in `query_sent`, PII and injections cut from snippets, a guardrail block wins, audit entry with URLs | `tests/test_security_service.py` (fake provider, real cluster); doctests of `web_query` in `src/service/websearch.py` |
| `/docs` examples (`web: true`, the toggle) are valid request bodies | `tests/test_api_docs.py` |
| on the live stack: how often it fires, what it returns, latency | `results/web_fallback_eval.json` |

## 5. Pitfalls and what I would do next

- **Web results are not ranked against the corpus.** They come back as a separate list, and nothing checks that the
  web text answers the question. An agent that writes answers would need to cite which source each claim came from.
