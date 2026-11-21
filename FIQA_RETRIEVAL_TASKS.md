# Take-Home: Distributed, Agentic Retrieval System

**Submission:** Link or zip of a git repo with commit history (no single-commit dumps).
**Follow-up:** 60-minute technical debrief. We walk through your code, re-run parts of it live, and extend it together.

---

## What we're evaluating

- **Empirical rigor:** are your numbers real, reproducible and honestly reported?
- **Distributed systems design:** does the service stay correct under sharding, live updates and model changes?
- **Agentic judgment:** can you build single- and multi-agent systems yourself, and measure when each is worth its cost?
- **ML judgment:** do your choices make empirical sense on *this* data under *these* constraints?
- **Design for change:** can the system evolve (new data, new model, more scale) without rework or downtime?
- **Code quality:** can another engineer read this in 10 minutes?
- **Communication:** are trade-offs stated specifically, not in generalities?
- **Self-assessment:** do you recognize your own system's weak spots?

| Area | Weight |
|---|---|
| Part A: Retrieval quality and evaluation | 25% |
| Part B: Distributed retrieval service | 30% |
| Part C: Agentic and multi-agent retrieval | 25% |
| Design doc, trade-offs and self-assessment | 20% |

---

## The problem

Build a retrieval system over a public Q&A corpus. Turn it into a sharded service that takes live updates and can switch embedding models with no downtime. Then build a single agent and a multi-agent system on top, and decide with data when each should be used.

### Corpus

[BEIR / FiQA](https://github.com/beir-cellar/beir): financial Q&A, about 57K passages, with labeled train, dev and test queries. Use the standard BEIR download.

- **Tune on dev only.** Report the test split **once**, for your final chosen configuration, and say so in DESIGN.md.
- An exact (flat) vector index is fine at this corpus size.

### Binding constraints

Your chosen **non-agentic** operating point must satisfy **all** of these:

- **CPU only:** no GPU at index or serve time.
- **Latency:** end-to-end p95 ≤ 50 ms per query through the router, measured after warmup.
- **Memory:** ≤ 2 GB RAM per shard process at serve time.

Configurations that miss a constraint are still valid to include in the comparison, just not as the chosen point. The agentic path (Part C) is exempt from the latency limit, but you must report its latency and cost.

---

## Part A: Retrieval quality

### A1. Retrieval strategies

Implement **three** strategies:

- **BM25** (required baseline)
- **Dense** (required): use **two** of `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5`. Pick one as your primary model; you will migrate to the other in Part B3.
- **Hybrid** (required): BM25 + dense, combined by a principled method (RRF, linear interpolation, learned weights, etc.).

Expose a CLI that queries the running service:

```
python search.py --query "what is short selling?" --top-k 10 [--strategy hybrid]
```

Output ranked passages with scores.

### A2. Evaluation harness

Evaluate on the FiQA dev set. Report:

**Overall metrics**
- nDCG@10, Recall@10, MRR, p50/p95 latency, peak RAM

**Stratified metrics**
- Recall@10 by query length: short (<5 tokens) / medium (5–15) / long (>15)
- Recall@10 on queries whose gold passage is in the top 10% longest docs vs. the rest

**Latency**
- Cold-start p50/p95 (first 20 queries) *and* warm p50/p95 (after a 100-query warmup). Explain any delta.

Commit the raw output to `results/bench.json`.

### A3. Ablations

Run at least **one real ablation** beyond the strategies above: either **cross-encoder reranking** (with a top-k cutoff sweep) or **hybrid weighting**. More ablations are welcome (dense model choice, truncation length, flat vs. approximate index). Report the observed deltas and your explanation.

### A4. Failure analysis: `results/failures.md`

Pick **3 specific failure cases** from your own eval run. For each, include:

- The exact query text
- Top-5 retrieved passages with scores
- The gold passage, with its rank and score in your system
- Your diagnosis of why your system missed it
- A concrete, specific fix you'd try next

---

## Part B: Distributed retrieval service

Everything runs on one laptop with `docker compose up`. Simulate the cluster with separate processes or containers. Replicas, caching and chaos testing are **not** required; we'll discuss them in the debrief.

### B1. Sharding

- Split the corpus across **at least 3 shards**, each a separate process with its own lexical and dense index.
- A **router** sends each query to the shards, merges results and returns the global top-k.
- **Correctness requirement:** the sharded service's quality metrics must match your single-node results within **1% relative** for every strategy. If they don't match at first, explain why and what you changed.
- Set a timeout on calls to shards. Decide and document what happens when a shard is slow or down: return partial results (and how the caller knows), or fail the query.
- Report p50/p95 for **1 shard vs. your N shards**.

### B2. Live updates

Expose an ingestion API for **upsert** and **delete** of passages.

- Writes are **idempotent**: replaying the same write (same doc ID and version) has no further effect, and an older version never overwrites a newer one.
- The lexical and dense indexes stay **consistent** with each other. State what a reader can see if a write fails halfway, and how the system recovers.
- State your **freshness guarantee** (for example "a write is searchable within N seconds"), and include a test that proves it.

### B3. Zero-downtime embedding model migration

Migrate the dense index from your primary embedding model to your second one **while queries are running**:

- Build the new index in the background, including passages written during the migration.
- Cut over with a flag, then show a working **rollback** to the old model.
- Ship this as runnable steps (`make migrate`, `make cutover`, `make rollback`) with a script that sends queries throughout and reports failed queries. The target is zero failed queries.

---

## Part C: Agentic and multi-agent retrieval

Build LLM agents that use your service's search as a tool, then decide with data when a multi-agent setup is worth its cost.

**Write all agent loops yourself** with the model provider's SDK. Agent frameworks (LangGraph, CrewAI, AutoGen, LlamaIndex agents, etc.) are not allowed for the loops or the orchestration.

### C1. Compound query set

Plain FiQA questions mostly need one search. To test multi-step retrieval, build a compound query set:

- With a fixed seed, sample 200 dev queries and pair them into **100 compound queries** (the two halves should be on different topics).
- Join each pair into one natural question. You may use an LLM to rewrite it; hand-check 10 and fix any that read badly.
- Gold labels are the union of both halves' qrels. **Keep track of which half each gold passage belongs to.**
- Commit the set as `data/compound_dev.jsonl` with the script that builds it.

### C2. Single agent (baseline)

- One agent with one tool, `search(query, strategy)`. It may rewrite the query and search again, then stops with a final top-10 list.
- Cap it at **3 steps**.

### C3. Multi-agent system

Build an **orchestrator** that coordinates **at least 2 sub-agents with distinct roles**. You choose the roles and justify them in DESIGN.md. Examples (not required): a planner that decomposes the question, retrieval workers that run sub-queries in parallel, a verifier that checks relevance and requests a re-search.

Requirements:

- **One final top-10 list.** Decide how results from different sub-agents share those 10 slots, and explain why.
- **Guardrails:** a total step/LLM-call cap and a per-query token budget across all agents.
- **Partial failure:** if one sub-agent fails or times out, the query still returns a result. Show this with a test.
- **Don't over-orchestrate:** the orchestrator should be able to skip decomposition when a question is simple.

### C4. Evaluation and decision

Evaluate on two sets: a **fixed 100-query sample** of plain dev queries (state the seed) and your **100 compound queries**. Compare your best non-agentic configuration, the single agent and the multi-agent system on:

- **Recall@10** and **MRR**
- **Coverage@10** (compound set only): the share of queries with at least one gold passage from **each** half in the top 10
- **Cost per query**, **latency** (p50/p95) and **average LLM calls**

Then, in DESIGN.md:

- Propose a **routing policy**: which queries should go to which path, and how the system decides.
- Make a **ship / don't ship** call for each agentic path. "It costs more and doesn't help" is a fine answer if the numbers support it.

We will also run your system on a held-out compound set built from the test split during the debrief.

### Example complex queries

These show the *style* of question a single search tends to half-answer. They are illustrative, not from your compound set:

1. *Two independent parts:* "How does short selling actually work, and is it smarter to pay off my mortgage early or invest the extra money?"
2. *Comparison:* "For someone in a high tax bracket who plans to retire early, is a Roth IRA or a traditional 401(k) the better choice?"
3. *Dependent parts:* "What happens to my unvested stock options if my company gets acquired, and how would they be taxed in that case?"
4. *Several factors:* "Is buying a house better than renting once you account for the mortgage interest deduction, opportunity cost and maintenance?"
5. *Personal plan:* "I'm 25 with student loans at 6% and an employer 401(k) match. What should I prioritise, and how big should my emergency fund be?"
6. *Sequential decision:* "If I'm self-employed, should I open a SEP IRA or a Solo 401(k), and can I still contribute to a Roth IRA on top of it?"

**LLM access:** we provide an API key with a hard spending cap. You may use a local model instead if you prefer. Document which model you used and your total spend.

---

## Design doc: `DESIGN.md` (2–3 pages)

Must include:

1. **Chosen operating point:** the configuration, and why.
2. **Benchmark table:** at least 5 configurations (BM25, two dense models, hybrid, one ablation point) with all metrics from A2, plus the single test-split run.
3. **Cold vs. warm latency:** what causes the delta, if any.
4. **Architecture:** a diagram of router, shards and ingestion path, plus the request path for a query and for a write.
5. **Consistency and freshness:** the guarantees you give, what can go wrong, and how you recover.
6. **Migration and rollback:** how B3 works and when you would roll back.
7. **Agent design and decision:** your sub-agent roles and why, the routing policy, and your ship / don't ship calls from Part C, with their numbers.
8. **One counterintuitive finding:** what you expected, what you observed, and your best hypothesis for why.
9. **Approaches that didn't pan out:** at least two. For each: what you tried, what you expected, what you observed, and why you think it failed.
10. **Scaling to 100M passages:** memory and cost estimates (show the arithmetic), index type, and what in your current design would break first.
11. **Production concerns:** 3–5 bullets on availability (replicas), caching, cost and monitoring.

---

## Repo layout

```
├── README.md               # setup + run
├── DESIGN.md               # design doc (sections above)
├── Dockerfile              # reproducible runtime
├── docker-compose.yml      # router, shards, ingestion
├── Makefile                # index, eval, bench, up, migrate,
│                           # cutover, rollback, agent-eval
├── src/
│   ├── search.py           # CLI entry point
│   ├── index/              # indexing and sharding
│   ├── retrieval/          # BM25, dense, hybrid
│   ├── service/            # router, shard server, ingestion
│   ├── agent/              # single agent, orchestrator, sub-agents
│   └── eval.py             # evaluation harness
├── data/
│   └── compound_dev.jsonl  # compound query set (C1) + build script
├── scripts/                # migration and query-load scripts
├── tests/                  # unit and integration tests
├── results/
│   ├── bench.json          # reproducible benchmark output
│   ├── agent_eval.json     # agent quality, cost, latency
│   └── failures.md         # failure analysis
└── requirements.txt
```

---

## Reproducibility contract

Ship a `Dockerfile` and `docker-compose.yml`. A reviewer with Docker must be able to bring up the service and run the eval end to end without installing Python locally.

**Quality metrics** (nDCG@10, Recall@10, MRR, ablation deltas) are hardware-independent. When we run your containers, they must match DESIGN.md within a **5% relative tolerance**. Submissions whose quality numbers don't reproduce are marked non-reproducible, which weighs heavily in grading.

**Latency** is hardware-dependent. Document the machine you measured on (CPU model, core count, RAM). We'll trust your reported numbers as measured on that hardware and sanity-check them against the spec; we won't re-measure.

**Agent results** depend on the LLM and are not deterministic. Report the model and temperature.

If indexing is long, ship a pre-built index artifact *and* the script that regenerates it.

---

## Ground rules

- **Python** for the retrieval and agent code. Open-source libraries are fine (for example rank_bm25/pyserini, sentence-transformers, faiss, FastAPI).
- Do **not** build the solution by wrapping a managed RAG platform or a vector database that hides sharding or updates from you. You may use LlamaIndex/LangChain-style libraries for glue, not as the core.
- Total code under about **2,000 lines**, excluding tests and data loaders.
- Tests:
  - Unit tests for at least **2** non-trivial functions (for example fusion scoring, an eval metric, merging results from shards).
  - Integration tests for idempotent writes and the freshness guarantee.
- Commit history should reflect real iteration. Don't squash to one commit.

---

## Assistance policy

Use AI assistants (Claude, ChatGPT, Copilot), documentation or web search **as you would at work**. We care about *your* judgment on top of that help.

In the debrief we'll probe why you made specific choices, what you observed in your runs, and what you'd change. "The assistant suggested it" is a fine part of an answer, paired with your own reasoning for keeping, changing or rejecting the suggestion. Expect to modify your own code live.

---

## Submission

Send a repo link (or zip) within 7 days. We'll review, then schedule the 60-minute debrief.

If you run out of time, **ship what you have with a short note on what's missing and what you'd do next.** Suggested priority if time is short: A1–A2, B1–B2, C, then the rest. A partial, honest submission with a clear write-up is stronger than a padded one.
