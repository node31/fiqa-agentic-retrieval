# 19 · Paying less for LLM calls

Every agent question costs tokens. This note covers six levers on that cost. Three are hard caps that bound the worst
case: a dollar budget per run, a maximum number of LLM calls per question, and a maximum number of tokens. Three lower
the typical cost: a model cascade, a semantic cache, and fewer tokens per call (trimming and provider prompt caching).
Each section explains the idea with a toy example you can run; the second half is how to use it and what I measured.
Answering questions the corpus can't is a separate topic: [20_web_search.md](20_web_search.md).

Code: `Budget` and `BudgetExceeded` in `src/agent/llm.py`, `Spend` in `scripts/agent_eval.py`, `src/agent/cascade.py`,
`src/agent/semantic_cache.py`, and the trimming parameters of `run_single_agent` in `src/agent/single_agent.py`.

## 1. Where the money goes, and the six levers

An LLM call is priced per token, input and output separately. For our agents the input dominates: the single agent
reads the system prompt, the question and 10 search results of 200 characters each, then answers with a short JSON
object. On a plain question that is about 2,118 tokens over 1 to 3 calls ($0.0025 at the placeholder prices of
$1 / $4 per 1M input / output tokens).

| # | lever | what it bounds or saves | where | when it bites | measured |
|---|---|---|---|---|---|
| 1 | budget (dollars per run) | the total spend of an evaluation or batch | `Spend` and `--max-usd` (default $2) in `scripts/agent_eval.py` | the running estimate crosses the cap: the run stops with `ABORT: estimated spend ...` | never hit: the full Part C eval (600 runs) was estimated at $0.79 |
| 2 | max calls per question | a looping or confused agent | `Budget.max_calls`: 3 for the single agent (= its step cap), 3 for the orchestrator | the next call would be one too many: `BudgetExceeded("call cap 3 reached")` before anything is sent | binds often: 72 of 100 compound questions used all 3 single-agent calls |
| 3 | max tokens | the size of one question's conversation, and of each answer | `Budget.max_tokens` (12,000 single, 8,000 orchestrator); `max_output_tokens` per call (250, the planner 200) | input estimate + the worst-case output would cross the cap: `BudgetExceeded("token budget ...")` | never hit: compound single-agent questions averaged 3,380 tokens |
| 4 | cascade | the typical cost: a cheap tier first, the strong one only when needed | `src/agent/cascade.py` | the cheap tier is unsure, or the question looks compound | R@10 0.546 vs 0.547 at $0.00240 vs $0.00251; a 10x cheaper tier 1 would save 18% |
| 5 | semantic cache | repeat and rephrased questions: no LLM call at all | `src/agent/semantic_cache.py` | a cached question of the same tenant scores cosine >= 0.85 | 65% of paraphrases hit their original, 0.8% false hits |
| 6 | fewer tokens per call | every call's input | trimming in `run_single_agent`; provider prompt caching | always (trimming); prefixes of 1,024+ identical tokens (caching) | the cheap tier reads about 442 input tokens; 0 cached tokens, as our prompts are too short |

Levers 1 to 3 cap the worst case and never make a question cheaper; levers 4 to 6 make the typical question cheaper
and never cap anything. A production agent needs both kinds.

## 2. Hard caps: budget, max calls, max tokens

All three caps refuse **before** money is spent. `Budget.reserve` runs at the start of every `llm.complete`: it adds
one call and checks the input estimate (4 characters a token) plus the largest output the call may produce
(`max_output_tokens`), because the real output length is only known after paying for it. One `Budget` object is
shared by every LLM call of one question, including the orchestrator's parallel workers, so a lock makes the check
and the increment atomic.

```python
from src.agent.llm import Budget, BudgetExceeded, FakeLLM, cost_usd

llm = FakeLLM(lambda system, user: '{"action": "finish", "doc_ids": []}')   # scripted replies, no network
system, user = "ROLE: single_agent\nReply with JSON.", "Question: what is a roth ira?\n" + "passage text " * 300

# max calls: the 4th call is refused before anything is sent
b = Budget(max_calls=3, max_tokens=100_000)
for i in range(4):
    try:
        llm.complete(system, user, max_output_tokens=250, budget=b)
        print(f"call {i + 1}: ok, {b.calls} calls, {b.tokens} tokens so far")
    except BudgetExceeded as e:
        print(f"call {i + 1}: refused: {e}")

# max tokens: refused when input estimate + the WORST-CASE output would cross the cap
b = Budget(max_calls=10, max_tokens=2_000)
for i in range(3):
    try:
        llm.complete(system, user, max_output_tokens=250, budget=b)
        print(f"call {i + 1}: ok, {b.tokens} tokens so far")
    except BudgetExceeded as e:
        print(f"call {i + 1}: refused: {e}")

# the dollar ceiling these caps imply for ONE question (placeholder prices: $1 in, $4 out per 1M tokens)
calls, cap, out = 3, 12_000, 250
print(f"worst case per question: ${cost_usd(cap - calls * out, calls * out):.5f}")
```

```
call 1: ok, 1 calls, 999 tokens so far
call 2: ok, 2 calls, 1998 tokens so far
call 3: ok, 3 calls, 2997 tokens so far
call 4: refused: call cap 3 reached
call 1: ok, 999 tokens so far
call 2: refused: token budget 2000 would be exceeded
call 3: refused: token budget 2000 would be exceeded
worst case per question: $0.01425
```

The second block shows the worst-case rule: call 2 would really have ended at about 1,998 tokens, under the cap,
but 999 so far + about 990 input + up to 250 output could reach 2,239, so it is refused. A cap that only checked
after the call could overshoot by a whole answer.

What happens on a refusal: the agents catch `BudgetExceeded` and degrade instead of failing. The single agent stops
asking the model and fills the top 10 from the searches it already has; the orchestrator searches the
question as asked if the planner is refused, and keeps its lists if the verifier is.
The error is recorded on the result (`errors`), so an eval counts how often a cap fired.

The three caps answer different questions:

- **Max calls** bounds behaviour. An agent that keeps asking for "one more search" is stopped at 3. On compound
  questions this is the cap that binds most: 72% of single-agent runs used all three calls, which is also why the
  single agent's compound cost ($0.0039) is higher than its plain cost ($0.0025).
- **Max tokens** bounds size. Each step re-sends the whole history, so the prompt grows with every search; 12,000
  tokens is about four times what a compound question used, so it only fires on a runaway. `max_output_tokens`
  bounds each answer: the agent replies with a short JSON object, and 250 tokens is plenty.
- **The dollar budget** bounds a run, not a question. `scripts/agent_eval.py` adds every result's estimated cost to
  a shared `Spend` and aborts the whole run past `--max-usd`. Together with the per-question caps it gives a ceiling
  you can compute in advance: at most 12,000 tokens per question, at most 750 of them output (3 calls x 250),
  means at most $0.01425 per question at the placeholder prices, so 300 questions can never cost more than $4.28.

## 3. The cascade

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

## 4. The semantic cache

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

## 5. Fewer tokens per call

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

## 6. Quick start

| I want to | command |
|---|---|
| ask one question through the cascade | `python scripts/ask.py "what is a roth ira?" --strategy cascade --backend azure --confirm-spend` |
| evaluate the cascade (both tiers on every question) | `python scripts/agent_eval.py --backend azure --confirm-spend --systems cascade --cascade-shadow --out results/agent_eval_cascade.json` |
| compare escalation rules offline, no LLM calls | `python scripts/cascade_policies.py` |
| calibrate the semantic cache | `python scripts/semantic_cache_eval.py --confirm-spend` (the spend is 100 paraphrase calls, once) |
| try it in the browser | `http://localhost:8000/docs`, tour step 12 |

Environment variables:

| variable | default | what it sets |
|---|---|---|
| `CASCADE_CHEAP_BACKEND` | the `--backend` value | the cheap tier's LLM backend (`azure`, `copilot`, `fake`) |
| `CASCADE_CHEAP_DEPLOYMENT` | `AZURE_OPENAI_DEPLOYMENT` | the cheap tier's Azure deployment, e.g. a mini model |
| `CASCADE_CHEAP_PRICE_IN` / `_OUT` | $1 / $4 per 1M tokens | the cheap tier's prices, for the cost columns |
| `CASCADE_STRONG_PRICE_IN` / `_OUT` | $1 / $4 per 1M tokens | the strong tier's prices |
| `SEMANTIC_CACHE_MIN` | 0.85 | the cosine a cached question needs to count as a hit |

## 7. Model cascade: usage and results

### What it does

Most questions are easy, and paying the strong configuration's price for each of them wastes money. The cascade
(`src/agent/cascade.py`) runs up to two tiers:

| tier | what it is | LLM calls |
|---|---|---|
| cheap | one call, no extra searches. It reads the question's hybrid top 5 with 120-character snippets, and its prompt (`SINGLE_AGENT_CHEAP`) also asks for a confidence, `high` or `low` | 1 |
| strong | the single agent exactly as before: top 10, 200-character snippets, up to 3 steps. The prompt is byte-identical to the single agent's, so its answers replay from the LLM disk cache | 1 to 3 |

Before the cheap tier runs, a free check (`looks_simple`, the same one the orchestrator uses) sends questions that
look compound straight to the strong tier. After it runs, the cascade escalates on the cheapest signal that fires:
`error`, `no finish`, `low confidence`, or `too few picks` (fewer than 3 valid ids). A wrong "keep" loses quality and
a wrong "escalate" only costs one strong run, so every rule leans toward escalating.

### How to use it

```
$ python scripts/ask.py "how do I fix a flat bicycle tire" --strategy cascade --backend azure --confirm-spend --url http://localhost:8000

tier cheap  (gpt-6-sol): 1 LLM call(s), 435 in + 45 out tokens, $0.00061, escalated: low confidence
tier strong (gpt-6-sol): 2 LLM call(s), 1896 in + 116 out tokens, $0.00236

final top 10 (3 searches, 8.7 s, total $0.00298):
 1. doc 122234: Wayshopy - Tech Mech Mechanical Vehicle Jack has Small compact design ...
 ...
the corpus has little on this question; from the web (wikipedia, sent: 'how do I fix a flat bicycle tire'):
   - Bicycle tire: A bicycle tire (tyre in Commonwealth English) is a tire that fits on the wheel of a bicycle ...
```

The same run on "what is a roth ira?" stops after the cheap tier (`kept`, 1 call, $0.00069). On "what is a 529 to
Roth IRA rollover" the cheap tier says `low confidence`, and the strong tier finds 10 relevant passages, so the web
is not asked.

In code, `run_cascade(question, cheap, strong, tool, cache=None, tenant=None, route_compound=True, shadow=False)`
returns an `AgentResult` whose `trace` has one entry per tier: tokens, dollars, picked ids and the escalation reason.
`default_tiers(cheap_llm, strong_llm)` builds the two tiers and `cascade_llms(backend, confirm)` reads the
`CASCADE_CHEAP_*` variables. `--no-route` in `agent_eval.py` switches the compound routing off.

To make the cheap tier really cheap, point it at a cheaper deployment and give both tiers their prices:

```bash
CASCADE_CHEAP_DEPLOYMENT=<a mini deployment> CASCADE_CHEAP_PRICE_IN=0.15 CASCADE_CHEAP_PRICE_OUT=0.6 \
  python scripts/ask.py "what is a roth ira?" --strategy cascade --backend azure --confirm-spend
```

I only have one Azure deployment (gpt-6-sol), so every number below has the same model in both tiers. The cheap
tier is cheaper only because it reads fewer tokens and makes one call.

### Results

Plain is the 100-question dev sample, compound the 100 compound questions; the strong-only row is the single agent
from `results/agent_eval.json`. Prices are the placeholder $1 / $4 per 1M tokens.

| | plain R@10 | plain $/q | compound R@10 | compound Cov@10 | compound $/q |
|---|---|---|---|---|---|
| strong only (single agent) | 0.547 | 0.00251 | 0.400 | 0.40 | 0.00393 |
| cascade v1 | 0.532 | 0.0044 | 0.373 | 0.34 | 0.0058 |
| cascade v2 (shipped) | 0.546 | 0.00240 | 0.400 | 0.40 | 0.00390 |

**Version 1 cost more than no cascade.** Its cheap tier was the single agent with trimmed context, allowed to search
again. With less to read it did search again: on 80 of the 100 plain questions it made 3 calls, about 1,676 tokens a
question. Its self-reported confidence was strict, so 74% of plain and 76% of compound questions escalated anyway,
and those paid for both tiers. Latency went up too: p50 5.67 s on plain, against the single agent's 4.09 s.

Version 2 changed three things: the cheap tier gets one call and no searches (about 500 tokens), the confidence
wording is plainer ("high" if the picked passages answer the question, "low" if they are only loosely related), and
compound questions skip the cheap tier. On plain, 75% still escalate: 44 routed as compound, 21 low confidence, 10
too few picks. Quality matches the single agent and the cost is 4% lower.

### Comparing rules offline

`--cascade-shadow` runs the tier that the rule skipped as well, marks it `"shadow": true` and keeps it out of every
total. With both answers recorded for every question, `scripts/cascade_policies.py` scores any rule on the same
answers, without another LLM call (`results/cascade_policies.json`). The what-if column reprices the cheap tier at
a tenth of the strong tier's price, the gap between a mini and a full model.

| rule (plain) | R@10 | MRR@10 | $/q | $/q, cheap tier at 1/10 price | to strong |
|---|---|---|---|---|---|
| strong only | 0.547 | 0.642 | 0.00251 | 0.00251 | 100% |
| cheap only | 0.502 | 0.525 | 0.00068 | 0.00007 | 0% |
| confidence (no routing) | 0.546 | 0.614 | 0.00214 | 0.00153 | 48% |
| picks only (ignore confidence) | 0.506 | 0.525 | 0.00088 | 0.00027 | 11% |
| route only | 0.517 | 0.594 | 0.00153 | 0.00120 | 44% |
| route + confidence (shipped) | 0.546 | 0.641 | 0.00240 | 0.00206 | 75% |
| oracle (better of the two, by recall) | 0.547 | 0.575 | 0.00094 | 0.00033 | 7% |

| rule (compound) | R@10 | Cov@10 | $/q | to strong |
|---|---|---|---|---|
| strong only | 0.400 | 0.40 | 0.00393 | 100% |
| cheap only | 0.204 | 0.10 | 0.00072 | 0% |
| confidence (no routing) | 0.384 | 0.38 | 0.00435 | 90% |
| route + confidence (shipped) | 0.400 | 0.40 | 0.00393 | 100% |

What the tables say:

- The self-reported confidence carries real signal. Ignoring it ("picks only") keeps 89% of plain questions on the
  cheap tier and loses 0.04 recall; using it recovers the strong tier's recall.
- The cheap tier finds the right documents nearly as often as the strong one (plain R@10 0.502 vs 0.547) but orders
  them worse (MRR 0.525 vs 0.642): one call can pick, but can't search again to check.
- Routing helps compound questions (the cheap tier alone halves their recall) and costs money on plain ones,
  because `looks_simple` calls 44% of plain questions compound. Without routing, plain costs $0.00214 instead of
  $0.00240 at the same recall, but compound then costs more than strong-only ($0.00435). Averaged over both sets
  the shipped rule is cheaper ($0.00317 vs $0.00325), so routing stays on.
- The oracle shows the ceiling: the strong tier is only needed for 7% of plain questions. A better escalation check
  is worth more than any other change here.
- With a genuinely cheaper tier-1 model, the same rules save much more: confidence-only would cost 39% less than
  strong-only on plain, at the same recall.

## 8. Semantic cache: usage and results

### What it does

The LLM disk cache in `src/agent/llm.py` only hits on a byte-identical prompt. `SemanticCache`
(`src/agent/semantic_cache.py`) keys on the question's embedding instead: a new question whose cosine to a stored
one is at least `SEMANTIC_CACHE_MIN` gets the stored answer, with no LLM call and no search. It uses the query
encoder the search already runs. Entries carry the caller's tenant and only match the same tenant. It is in
memory, thread-safe and first-in-first-out once full.

### How to use it

```python
from src.agent.semantic_cache import SemanticCache
from src.retrieval.dense import get_embedder

emb = get_embedder("minilm-ft")
cache = SemanticCache(lambda q: emb.encode_queries([q])[0])        # threshold 0.85
cache.store("how do I roll my 401k into an IRA?", ["125973", "292937"], tenant="acme")
cache.lookup("how do I roll over my 401k to an IRA?", tenant="acme")  # {'value': [...], 'similarity': 0.935, ...}
cache.lookup("how do I roll over my 401k to an IRA?", tenant="globex")  # None: other tenant
cache.stats()                                                       # entries, hits, misses, hit_rate, threshold
```

Pass it to the cascade with `run_cascade(..., cache=cache, tenant="acme")`. A hit shows up as a `cache` entry in
the trace with 0 calls and $0; a fresh answer without errors is stored.

The cosine is only as good as the encoder. Against "how do I roll my 401k into an IRA?":

| question | cosine | at 0.85 |
|---|---|---|
| how do I roll over my 401k to an IRA? | 0.935 | hit, right |
| how do I roll my 401k into a Roth IRA? | 0.928 | hit, **wrong**: a Roth rollover is taxed differently |
| can I move my old 401(k) to an IRA? | 0.657 | miss, although it is the same question |
| should I pay off my mortgage early? | 0.108 | miss |

One changed word can change the answer and barely move the vector, while a real rephrasing can move it a lot. That
is why the threshold comes from a measurement and not from intuition.

### Choosing the threshold

`scripts/semantic_cache_eval.py` measures both sides (`results/semantic_cache_eval.json`):

- benefit: Azure rewrote the 100 plain questions in other words (`data/paraphrases_plain.jsonl`); with the 6,000
  train and dev questions in the cache, how many paraphrases find their original;
- harm: for every dev question, the nearest *different* question; at a threshold above that cosine it would be
  served someone else's answer. For those cases both result lists are scored against the asker's own labels.

| threshold | paraphrases that hit their original | paraphrases that hit a wrong question | dev questions with a false hit |
|---|---|---|---|
| 0.80 | 84% | 0% | 2.4% |
| **0.85** | **65%** | 0% | **0.8%** (4 of 500) |
| 0.88 | 48% | 0% | 0.4% |
| 0.90 | 40% | 0% | 0.2% |
| 0.95 | 14% | 0% | 0% |

I chose the rule before looking: the lowest threshold where at most 1% of dev questions get a false hit. That is
0.85. The 4 false hits there are near-duplicate FiQA questions, and the served results scored a little better than
the askers' own (nDCG@10 0.255 vs 0.211), so at this threshold the false hits did no measurable harm. My first guess,
0.95, would have kept only 14% of the benefit.

Fed the plain-set cascade answers, the cache at 0.85 answered 65 of the 100 paraphrases, saving 131 LLM calls and
$0.145 of the $0.240 the plain set cost, with the served answers at the originals' recall (0.596 on those 65).
That is an upper bound: every paraphrase here has its original in the cache, which real traffic won't.

## 9. Context trimming and prompt caching: usage and results

**Trimming.** `run_single_agent` takes `max_hits`, `snippet_chars` and `dedupe`. The strong tier keeps the old
values (10 hits, 200 characters); the cheap tier uses 5 hits, 120 characters, and with `dedupe` a later search shows
only passages it hasn't shown yet. The cheap tier's one call reads about 442 input tokens a question; the strong
tier, like the single agent, uses about 2,118 tokens on a plain question over its calls. Trimming only saves money
if the model doesn't make up for it by searching more, which is what went wrong in cascade v1.

**Provider prompt caching.** Azure OpenAI bills a repeated prompt prefix at a discount once it is at least 1,024
tokens long. `AzureLLM` now reads `usage.input_tokens_details.cached_tokens`, `AgentResult.cached_input_tokens`
carries it, and `agent_eval.py` reports `avg_provider_cached_input_tokens`. In every run it was 0: the system prompt
plus the first search results are under 1,024 tokens, so no call had a long enough repeated prefix. It would start
to pay with a longer system prompt (few-shot examples, tool schemas) placed first and kept identical, with the
question and results after it.

## 10. How it is checked

| what | where |
|---|---|
| the call cap and the token cap refuse before spending (worst-case output counted); agents degrade instead of failing | `tests/test_agents.py` (`Budget`, fake LLM) |
| escalation reasons, routing, shadow entries outside the totals, prices per tier, the cheap tier reading less while the strong prompt stays byte-identical, provider cached tokens recorded | `tests/test_cascade.py` (fake LLM, no network) |
| the cascade as a strategy next to single and multi | `tests/test_agents.py` |
| semantic cache: same tenant only, a miss below the threshold; a hit skips every LLM call | doctests in `src/agent/semantic_cache.py`; `tests/test_cascade.py` |
| quality and cost on real questions | `results/agent_eval_cascade.json`, `results/cascade_policies.json`, `results/semantic_cache_eval.json` |

## 11. Pitfalls and what I would do next

- **One model in both tiers.** The cascade's real saving needs a cheaper model in tier 1. Next: run it with a mini
  deployment and check the confidence calibration again, since a different model reports confidence differently.
- **Self-reported confidence.** It helped here, but a model can be confidently wrong. Two cheaper checks I'd try:
  agreement between the cheap tier's picks and the plain hybrid top 5, and the top dense score.
- **`looks_simple` over-routes.** 44% of plain questions look compound to it. A better check (the orchestrator's
  planner, or a tiny classifier) would cut the plain cost to the no-routing $0.00214 without hurting compound.
- **Semantic cache staleness.** A cached answer doesn't notice that the corpus changed. I would key entries on a
  corpus version that every write bumps, or expire them by age. The threshold also belongs to the encoder: re-run
  the eval after a model migration.
- **Latency from cached runs.** Replays from the LLM disk cache report near-zero latency; take latency from the
  first run.
- **Prices are placeholders** ($1 / $4 per 1M tokens), so read the dollar columns as ratios between rows.
