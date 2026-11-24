# 19 · Paying less for LLM calls, and answering what the corpus can't

Every agent question costs tokens. This note covers the ideas behind the four things I added to cut that cost (a
model cascade, a semantic cache, context trimming, provider prompt caching) and the web fallback for questions the
corpus has no answer to. Each idea comes with a toy example you can run. How to use them, and every measured number,
is in [docs/design/LLM_OPTIMIZATIONS.md](../design/LLM_OPTIMIZATIONS.md).

Code: `src/agent/cascade.py`, `src/agent/semantic_cache.py`, `src/service/websearch.py`, and the trimming
parameters of `run_single_agent` in `src/agent/single_agent.py`.

## 1. Where the money goes

An LLM call is priced per token, input and output separately. For our agents the input dominates: the single agent
reads the system prompt, the question and 10 search results of 200 characters each, then answers with a short JSON
object. On a plain question that is about 2,118 tokens over 1 to 3 calls. So there are three levers:

| lever | idea | here |
|---|---|---|
| fewer calls | don't call the LLM at all when you already know the answer | semantic cache |
| cheaper calls | call a cheaper configuration first, the expensive one only when needed | cascade |
| shorter calls | send fewer tokens per call, or get repeated tokens billed at a discount | trimming, prompt caching |

## 2. The cascade

Run a cheap tier first. If its answer looks reliable, stop; otherwise run the strong tier. (FrugalGPT, 2023, is the
paper usually cited for LLM cascades.) The expected cost per question is

```
cost = c_cheap + p_escalate × c_strong
```

so the cascade beats strong-only only while `p_escalate < 1 − c_cheap / c_strong`, the break-even escalation rate.
A cheap tier that costs almost as much as the strong one leaves almost no room to escalate.

```python
def cascade_cost(c_cheap, c_strong, escalate):
    return c_cheap + escalate * c_strong

for name, c_cheap, c_strong, esc in [("textbook", 0.0005, 0.0025, 0.30), ("our v1", 0.00223, 0.0025, 0.74),
                                     ("our v2, non-routed", 0.00068, 0.0025, 31/56)]:
    print(f"{name:20s} cascade ${cascade_cost(c_cheap, c_strong, esc):.5f} vs strong ${c_strong:.5f};"
          f" break-even escalation {1 - c_cheap / c_strong:.0%}, actual {esc:.0%}")
```

```
textbook             cascade $0.00125 vs strong $0.00250; break-even escalation 80%, actual 30%
our v1               cascade $0.00408 vs strong $0.00250; break-even escalation 11%, actual 74%
our v2, non-routed   cascade $0.00206 vs strong $0.00250; break-even escalation 73%, actual 55%
```

My first cascade is the middle line. Its cheap tier read less context but was allowed to search again, so it did,
and cost $0.00223 a question against the strong tier's $0.0025. That left room for 11% escalation, and 74%
escalated. The model predicts $0.00408; I measured $0.0044. Version 2 gave the cheap tier exactly one call, which
moved the break-even to 73%.

The cascade also needs a signal for "this answer looks unreliable". Ours, cheapest first:

| signal | costs | catches |
|---|---|---|
| route: the question looks compound | nothing (a word count and "and"/"?" check) | questions the cheap tier would almost surely fail |
| error, no finish | nothing | broken JSON, timeouts |
| low confidence | nothing extra (a field in the same answer) | the model's own doubt |
| too few picks (< 3 ids) | nothing | it found little that answers the question |

The toy below runs both tiers with the scripted fake LLM, which is confident on a one-topic question and not on a
two-topic one (no network, no cost):

```python
from src.agent.llm import FakeLLM
from src.agent.fake_policies import heuristic_responder
from src.agent.cascade import default_tiers, run_cascade

hits = [{"doc_id": str(i), "score": 1.0 / i, "text": f"passage {i}"} for i in range(1, 11)]
tool = lambda q, s="hybrid": hits                     # a search that always returns the same 10 passages
cheap, strong = default_tiers(FakeLLM(heuristic_responder), FakeLLM(heuristic_responder))
for q in ["what is a roth ira?", "what is a roth ira and should I pay off my mortgage?"]:
    r = run_cascade(q, cheap, strong, tool, route_compound=False)
    print(q)
    for t in r.trace:
        print(f"   {t['tier']:6s} calls={t['llm_calls']} in={t['input_tokens']} out={t['output_tokens']} escalate={t.get('escalate')}")
```

```
what is a roth ira?
   cheap  calls=1 in=276 out=20 escalate=None
what is a roth ira and should I pay off my mortgage?
   cheap  calls=1 in=292 out=19 escalate=low confidence
   strong calls=2 in=570 out=28 escalate=None
```

With `route_compound=True` (the default) the second question skips the cheap tier and goes straight to strong.

### Evaluating escalation rules without paying twice

To compare rules fairly, every rule must be scored on the same answers; two separate LLM runs differ by noise
alone. Shadow mode (`--cascade-shadow`) runs both tiers on every question, records the tier the rule skipped as
`"shadow": true`, and keeps it out of the totals. Any rule can then be replayed offline: "strong only", "cheap
only", "confidence", "picks only", and the **oracle**, which picks the better answer per question with hindsight.
No real rule can beat the oracle; the gap between your rule and the oracle is what a better signal could still win.
On plain questions the oracle needed the strong tier for only 7% of questions, while our rule sent 75%.

## 3. The semantic cache

An exact cache keys on the prompt string, so "how do I roll my 401k into an IRA?" and "how do I roll over my 401k to
an IRA?" are two misses. A semantic cache keys on the question's embedding: it returns a stored answer when the
cosine similarity to a stored question is at least a threshold. It is a nearest-neighbour search with a cut-off,
the same operation as dense retrieval, run over past questions instead of passages.

The threshold trades two errors against each other: too high and paraphrases miss (no saving), too low and a
*different* question gets someone else's answer. The toy uses made-up 4-dimensional vectors:

```python
import numpy as np
cached = {"how do I roll my 401k into an IRA?": np.array([0.90, 0.40, 0.10, 0.10])}
incoming = {"how do I roll over my 401k to an IRA?": np.array([0.86, 0.45, 0.10, 0.22]),
            "how do I roll my 401k into a Roth IRA?": np.array([0.88, 0.38, 0.30, 0.10]),
            "can I move my old 401(k) to an IRA?": np.array([0.55, 0.70, 0.10, 0.45]),
            "should I pay off my mortgage early?": np.array([0.05, 0.10, 0.20, 0.97])}
unit = lambda v: v / np.linalg.norm(v)
(cq, cv), = cached.items()
for q, v in incoming.items():
    s = float(unit(v) @ unit(cv))
    print(f"{s:.3f}  {q:40s}  at 0.85: {'hit ' if s >= 0.85 else 'miss'}  at 0.95: {'hit' if s >= 0.95 else 'miss'}")
```

```
0.991  how do I roll over my 401k to an IRA?     at 0.85: hit   at 0.95: hit
0.980  how do I roll my 401k into a Roth IRA?    at 0.85: hit   at 0.95: hit
0.832  can I move my old 401(k) to an IRA?       at 0.85: miss  at 0.95: miss
0.204  should I pay off my mortgage early?       at 0.85: miss  at 0.95: miss
```

The Roth question is a different question (a Roth rollover is taxed), yet it sits as close as the true rephrasing,
and the third line is the same question in other words and misses. No threshold fixes both. The real encoder
behaves the same way: minilm-ft gives 0.935, 0.928 and 0.657 for the first three. So the threshold has to be set
from data, on the error you care about most. I fixed the rule first (at most 1% of real dev questions may hit a
different question) and then measured: 0.85, where 65% of paraphrases still hit.

Two more rules every semantic cache needs:

- **Tenant isolation.** An answer built from acme's documents must never be served to globex, so the tenant is part
  of the key.
- **Staleness.** The cache doesn't know the corpus changed. Key on a corpus version, or expire entries by age.

## 4. Fewer tokens per call

**Trimming** is the obvious lever: show fewer and shorter passages (our cheap tier: 5 hits of 120 characters instead
of 10 of 200), and don't show a passage twice. It saves only if the model doesn't compensate. An agent that is
allowed to search again often will, when it sees less: that is how cascade v1 lost money.

**Provider prompt caching** is the less obvious lever. The model provider keeps the computed state (the attention
key/value cache) of a prompt's beginning, and when the next prompt starts with the same tokens it reuses that work
and bills those tokens at a discount. On Azure OpenAI it applies automatically once the identical prefix is at
least 1,024 tokens. Only the *prefix* counts: one changed token near the start makes everything after it new.

```python
from src.agent.prompts import SINGLE_AGENT

def shared_prefix_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n

p1 = SINGLE_AGENT + "\nQuestion: what is a roth ira?\n1. doc 214174: A Roth IRA is simply ..."
p2 = SINGLE_AGENT + "\nQuestion: should I pay off my mortgage early?\n1. doc 8126: ..."
print("system prompt words:", len(SINGLE_AGENT.split()), " shared prefix words:", shared_prefix_words(p1, p2))
```

```
system prompt words: 117  shared prefix words: 118
```

Two questions share only the system prompt, 117 words, far below 1,024 tokens. So every run reported 0 cached
tokens. The layout rule still holds for when it does pay: put everything that never changes (instructions,
examples, tool definitions) first, and the question and search results last.

## 5. When the corpus has no answer: the web fallback

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

## 6. Pitfalls

- A cascade with the same model in both tiers saves little: the cheap tier is cheaper only by reading less. The
  break-even formula in section 2 tells you before you build it.
- A self-reported confidence is a signal, not a fact. Check it against the oracle on your own data.
- A semantic-cache threshold belongs to one encoder. After a model migration, re-run the calibration.
- Measure latency on the first run: replays from a response cache report near-zero latency.
