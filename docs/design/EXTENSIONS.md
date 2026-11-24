# What else I'd add, as the engineering lead

[DESIGN.md](../../DESIGN.md) describes what I built and what it measured. This page is my list of what I'd add next,
grouped by the part of the system it touches. Unless a line says otherwise (items 2, 15 and 20 to 25 are built), nothing here is built yet; the numbers
quoted are the ones DESIGN.md measured, and each item says how I'd check it. Security and LLM guardrails (tenant
identity, authorisation, tenant filtering, PII redaction, audit log, harmful-intent, injection, scope and rate-limit
guards) are built and have their own page: [SECURITY.md](../topics/17_security.md). So are the LLM cost items 20 to 23 below,
on [19_llm_cost.md](../topics/19_llm_cost.md) and [20_web_search.md](../topics/20_web_search.md), and A2A and agent memory (items 24 and 25), on
[A2A_AND_MEMORY.md](A2A_AND_MEMORY.md).

## Contents

- [Retrieval quality (A1)](#retrieval-quality-a1)
- [Evaluation (A2)](#evaluation-a2)
- [Failures and their fixes (A4)](#failures-and-their-fixes-a4)
- [Scaling the service (B1)](#scaling-the-service-b1)
- [Writes (B2)](#writes-b2)
- [Agents and LLMs (Part C)](#agents-and-llms-part-c)

## Retrieval quality (A1)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 1 | **Tune the parameters: α, and BM25's k1 and b** | α = 0.7 was swept on dev (0.4190 nDCG@10); k1 = 1.2 and b = 0.75 are textbook defaults I never tuned, and FiQA's long forum answers may want a different length penalty `b` | grid on dev, one test run at the end |
| 2 | **Fine-tune the embedding model** on the 5,500 FiQA train questions (**done**) | in-domain training is usually the biggest single gain, and the train labels were unused | done: dev hybrid nDCG@10 0.4190 → 0.4469 (95% CI of the gain +0.016 to +0.040), test 0.3968 → 0.4358; α stayed at 0.7. See [17_fine_tuning.md](../topics/18_fine_tuning.md) |
| 3 | **Try different datasets** | to show the choices aren't FiQA-specific: other BEIR sets, finance QA sets, multi-hop sets for Part C | the same harness on each dataset |
| 4 | **Pick models from the MTEB leaderboard** | MTEB is the standard leaderboard for embedding models; it's where I'd shortlist a replacement for MiniLM | shortlist by MTEB retrieval score and size, then run our own dev eval, because leaderboard rank doesn't always carry over to one domain |
| 5 | **Chunk the passages** | the encoder reads only the first 256 tokens, so long gold passages are found less often (Recall@10 0.464 vs 0.508) | index overlapping chunks, score a doc by its best chunk; check the long-passage slice |

## Evaluation (A2)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 6 | **Report precision** (the fraction of returned results that are actually good) | recall says how much of the gold we found; precision says how much of what we show is useful, which is what a user sees | Precision@10 next to Recall@10; FiQA labels are sparse, so it's a lower bound |
| 7 | **Judge the LLM's answers, not only the retrieval.** An answer should be: **grounded** (it comes from the retrieved passages, as in RAG), **correct** (confirmed by an independent check), **complete** (no missing information) and **relevant** (it answers the question asked) | once agents write answers, Recall@10 says nothing about whether the answer is right | an LLM judge per criterion, checked against human ratings on a sample (RAGAS and TruLens use the same criteria) |
| 8 | **Use task-specific benchmarks to verify a model**, e.g. **SWE-bench** for coding tasks (reading of my notes: "coding task → SWE-bench") | a general leaderboard doesn't tell you whether a model is good at your task; pick the benchmark that matches the task | for our agents, the equivalent is the agent eval in `results/agent_eval.json` |

## Failures and their fixes (A4)

The failure analysis found two kinds I'd fix first:

| failure | example | fix |
|---|---|---|
| **long passages** | qid 4029: the answering part of the gold passage is past the 256-token limit | **chunking** (item 5) |
| **rare terms** | qid 4746, "EFU and NAFA": BM25 ranks the gold #1, dense misses it, and α = 0.7 drowns BM25 out | **query-adaptive α**: more BM25 weight when the query has a very rare term (my one quick rule moved nDCG only 0.4190 → 0.4206; a learned rule might do more) |

## Scaling the service (B1)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 9 | **Router replicas** | one router is a single point of failure | kill a router under load; no failed queries |
| 10 | **Shard replicas** | a dead shard means partial results today | kill one replica under load; no `partial: true` |
| 11 | **Cache the global DF table** at the router | removes the statistics round trip on every query (about 15 ms of the router's time) | refresh it periodically; nDCG must stay equal to single node |
| 12 | **Consistent hashing** instead of `crc32(doc_id) % 3` | with `% N`, changing the shard count moves almost every doc | reshard 3 → 4 and count the moved docs |
| 13 | **Hedging**: send the same query to more than one replica and take the first answer | waiting for the slowest shard is what drives p95 | p95 with and without hedging, at a capped extra load |
| 14 | **ANN indexes** (HNSW, IVF-PQ) instead of exact search | exact search is fine at 19K docs per shard, not at tens of millions | recall against exact search at the same latency |

## Writes (B2)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 15 | **Sync vs async, both behind the WAL** (built) | sync: searchable on the next query, 57.8 writes/s. Async: searchable within 1 s, 94.2 writes/s. Both fsync the WAL before acking, so an acked write survives a crash | the freshness tests in DESIGN.md §6.6 |
| 16 | **Live writes from a source database through change data capture (CDC)** | in production, documents change in a database; CDC turns each committed change into an upsert or delete carrying the database's version | kill the consumer mid-stream, restart, compare counts with the source |

## Agents and LLMs (Part C)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 17 | **Break complex queries into simpler ones** | a compound question searched as one query finds one half (baseline Coverage@10 0.10); splitting it is what the agents do (single 0.40, multi 0.32) | Coverage@10 on the compound set |
| 18 | **Closed models**: improve them through **retrieval and tools** at question time, and **distillation** into a small open model (reading of my notes) | you can't fine-tune a closed model's weights; you can give it better context, or train an open model on its outputs | the distilled model must match the big one on the agent-eval table |
| 19 | **Check the LLM's answer quality** (item 7) | the four criteria: grounded, correct, complete, relevant | as in item 7 |
| 20 | **Model cascade** (built) | most questions don't need the strong configuration | built: plain R@10 0.546 vs 0.547 at 4% lower cost with one model in both tiers; the shadow replay prices a 10x cheaper tier-1 model at 18% lower. [19_llm_cost.md](../topics/19_llm_cost.md#2-the-cascade) |
| 21 | **Semantic cache** (built) | users rephrase; an exact prompt cache misses every rephrasing | built: at cosine 0.85, 65% of paraphrases hit, 0.8% of dev questions hit a different question |
| 22 | **Web search fallback** (built, toggle `web_fallback`) | the corpus ends around 2018 and is finance only | built: fires on 1% of dev and 70% of off-topic questions; the question is sent PII-free |
| 23 | **Context trimming and provider prompt caching** (built) | fewer input tokens per call; repeated prefixes billed at a discount | built: the cheap tier reads about 442 input tokens; cached tokens are reported, and were 0 because our prompts are under the 1,024-token minimum |
| 24 | **A2A (Agent2Agent) service** (built) | other agents can discover the FiQA agents from a card and delegate to them | built: delegation of compound questions over A2A, Coverage@10 0.11 → 0.34 without an LLM. [A2A_AND_MEMORY.md](A2A_AND_MEMORY.md) |
| 25 | **Agent memory** (built) | follow-ups and user facts ("I live in Canada") | built: follow-up R@10 0.103 → 0.439 (rule) / 0.455 (LLM rewrite); PII-redacted, per user, expiring, deletable |
