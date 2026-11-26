# What else I'd add, as the engineering lead

[DESIGN.md](../../DESIGN.md) describes what I built and what it measured. This page is my list of what I'd add next,
grouped by the part of the system it touches. Unless a line says otherwise (items 2, 15 and 20 to 25 are built, and so is everything in the first section), nothing here is built yet; the numbers
quoted are the ones DESIGN.md measured, and each item says how I'd check it. Security and LLM guardrails (tenant
identity, authorisation, tenant filtering, PII redaction, audit log, harmful-intent, injection, scope and rate-limit
guards) are built and have their own page: [17_security.md](../topics/17_security.md). So are the LLM cost items 20 to 23 below,
on [19_llm_cost.md](../topics/19_llm_cost.md) and [20_web_search.md](../topics/20_web_search.md), and A2A and agent memory (items 24 and 25), on
[21_a2a.md](../topics/21_a2a.md) and [22_agent_memory.md](../topics/22_agent_memory.md).

## Contents

- [Built beyond the task](#built-beyond-the-task)
- [Retrieval quality (A1)](#retrieval-quality-a1)
- [Evaluation (A2)](#evaluation-a2)
- [Failures and their fixes (A4)](#failures-and-their-fixes-a4)
- [Scaling the service (B1)](#scaling-the-service-b1)
- [Writes (B2)](#writes-b2)
- [Agents and LLMs (Part C)](#agents-and-llms-part-c)

## Built beyond the task

The assignment asked for hybrid retrieval, a sharded service with live writes and a model migration, and agents on
top. These are the pieces I built after that, to learn how a retrieval service behaves once real users, real money
and other agents are involved. Each is off by default or opt-in, so the assignment's results don't change.

Each item belongs to one of five categories, used for colour in the deck and the replays: **Distributed** (how the
cluster behaves under writes), **Agentic Security** (who is calling, what they may do, abuse of the LLM),
**Agentic Privacy** (personal data and accountability), **Agentic Cost Optimization** (what each answer costs)
and **Agentic Extensions** (agents that talk to agents, remember, and look outside the corpus).

| # | category | what | where in the code | what I measured | topic |
|---|---|---|---|---|---|
| 1 | Distributed | **Sync vs async writes.** Sync applies the write before it answers; async answers once the WAL is on disk and a background applier makes it searchable | `src/service/store.py`, `wal.py`, `APPLY_MODE` | sync 57.8 writes/s, searchable on the next query; async 94.2 writes/s, searchable within 1 s. Both fsync the WAL before the ack, so an acked write survives a crash | [09](../topics/09_live_updates_consistency.md) |
| 2 | Agentic Cost Optimization | **Fine-tuning the embedding model** on the 5,500 FiQA train questions | `scripts/finetune.py` | hybrid nDCG@10 on dev 0.4190 → 0.4469, on test 0.3968 → 0.4358, at MiniLM's size and speed | [18](../topics/18_fine_tuning.md) |
| 3 | Agentic Cost Optimization | **LLM cost and budgets.** A hard cap per run (`Budget`: max calls, max tokens; `--max-usd` in `scripts/agent_eval.py`), a cascade, a semantic cache and trimming | `src/agent/llm.py`, `cascade.py`, `semantic_cache.py` | the cascade keeps plain R@10 (0.546 vs 0.547) at $0.00240 vs $0.00251 a question; a 10x cheaper tier-1 model would save 18%. The semantic cache at cosine 0.85: 65% of paraphrases hit, 0.8% false hits | [19](../topics/19_llm_cost.md) |
| 4 | Agentic Extensions | **Web search** when the corpus has no good answer (toggle `web_fallback`) | `src/service/websearch.py`, `web_fallback` in `src/service/router.py` | fires on 1% of dev questions and 70% of off-topic ones; adds 834 ms at p50 when it fires; the question leaves PII-free | [20](../topics/20_web_search.md) |
| 5 | Agentic Security | **Authentication and authorisation.** Signed tokens, one tenant per user, a user reads only their tenant's docs and can't overwrite another tenant's | `src/security/tokens.py`, `policy.py`; the owner map and the tenant filter in `src/service/store.py` | a filtered search returns exactly what a tenant-only index returns (same ids, order and scores); filtering after ranking instead left alice 5 of 10 results; bob's overwrite of alice's doc is a 403. Router p50 31.9 ms → 32.3 ms | [17 §4.1 to 4.4](../topics/17_security.md) |
| 6 | Agentic Privacy | **Privacy: PII redaction** of emails, card numbers (Luhn), IBANs (mod 97), SSNs, phones and IPs | `src/security/pii.py` | 237 of 57,638 FiQA passages hold PII (182 phones, 165 emails, 17 card numbers); the index keeps the original, so search quality is unchanged | [17 §4.5](../topics/17_security.md) |
| 7 | Agentic Privacy | **Audit log**: one line per request, hash-chained | `src/security/audit.py`, `GET /admin/audit/verify` | an edited, deleted or reordered line is reported at the first bad line (tests); questions are stored redacted even when redaction of results is off | [17 §4.6](../topics/17_security.md) |
| 8 | Agentic Security | **Guardrails: harmful intent and prompt injection** (criminal how-to questions, injected questions, poisoned passages) | `src/security/guardrails.py` | held-out set: 23 of 25 how-to fraud questions and 10 of 10 injections caught, 0 of 25 legitimate questions refused; 0 of all 6,648 FiQA questions refused; about 43 µs a question | [17 §4.7](../topics/17_security.md) |
| 9 | Agentic Security | **LLM abuse: topic scope.** A question far from everything in the corpus gets no results, so nobody uses the finance assistant as a free general chatbot | `guard_scope` in `src/security/guardrails.py` | threshold 0.4421 (the 1st percentile of dev questions' best score): 1% of dev questions flagged, 70% of off-topic questions caught | [17 §4.8](../topics/17_security.md#48-topic-scope) |
| 10 | Agentic Extensions | **A2A protocol 1.0.** The agents as a service other agents discover from a card and call over JSON-RPC | `src/a2a/` | a coordinator that splits compound questions and delegates each part over A2A: Coverage@10 0.11 → 0.34 with no LLM; 314 tasks in 21.1 s. A second agent (`src/a2a/coordinator.py`, :8310) delegates to the first with the caller's token, recorded as a sequence-diagram replay; a browser chat page at `/ui` | [21](../topics/21_a2a.md) |
| 11 | Agentic Cost Optimization | **LLM query rewrite.** The verifier rewrites a sub-query whose results look off topic; a follow-up is rewritten into a standalone question | `verify` in `src/agent/orchestrator.py`; `--llm-rewrite` in `scripts/memory_eval.py` | the verifier rewrote a sub-query in 16 of the 100 compound questions; follow-up R@10 0.455 with the LLM rewrite vs 0.439 with the free rule (the full question scores 0.491) | [13](../topics/13_multi_agent_orchestration.md), [22](../topics/22_agent_memory.md) |
| 12 | Agentic Extensions | **Agent memory**: the conversation's last turn and facts about the user, per user, redacted, expiring, deletable | `src/agent/memory.py`, `/memory` on the A2A server | follow-up R@10 0.103 → 0.439; "I live in Canada" moves the share of top-10 passages about Canada from 8% to 54%. Six duties (attribution, isolation, expiry, access, deletion, opt-out), each enforced in one place and shown in a recorded replay; deleted text is overwritten in the file (`secure_delete`) | [22](../topics/22_agent_memory.md) |
| 13 | Agentic Cost Optimization | **Rate limiting**: a token bucket per user (per IP when anonymous) | `src/security/ratelimit.py` | 120 a minute with bursts of 20: bob's 30 quick searches got 22 answers, then 429s, while alice kept getting 200 | [17 §4.9](../topics/17_security.md#49-rate-limiting) |

## Retrieval quality (A1)

| # | idea | why | how I'd check it |
|---|---|---|---|
| 1 | **Tune the parameters: α, and BM25's k1 and b** | α = 0.7 was swept on dev (0.4190 nDCG@10); k1 = 1.2 and b = 0.75 are textbook defaults I never tuned, and FiQA's long forum answers may want a different length penalty `b` | grid on dev, one test run at the end |
| 2 | **Fine-tune the embedding model** on the 5,500 FiQA train questions (**done**) | in-domain training is usually the biggest single gain, and the train labels were unused | done: dev hybrid nDCG@10 0.4190 → 0.4469 (95% CI of the gain +0.016 to +0.040), test 0.3968 → 0.4358; α stayed at 0.7. See [18_fine_tuning.md](../topics/18_fine_tuning.md) |
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
| 20 | **Model cascade** (built) | most questions don't need the strong configuration | built: plain R@10 0.546 vs 0.547 at 4% lower cost with one model in both tiers; the shadow replay prices a 10x cheaper tier-1 model at 18% lower. [19_llm_cost.md](../topics/19_llm_cost.md#3-the-cascade) |
| 21 | **Semantic cache** (built) | users rephrase; an exact prompt cache misses every rephrasing | built: at cosine 0.85, 65% of paraphrases hit, 0.8% of dev questions hit a different question |
| 22 | **Web search fallback** (built, toggle `web_fallback`) | the corpus ends around 2018 and is finance only | built: fires on 1% of dev and 70% of off-topic questions; the question is sent PII-free |
| 23 | **Context trimming and provider prompt caching** (built) | fewer input tokens per call; repeated prefixes billed at a discount | built: the cheap tier reads about 442 input tokens; cached tokens are reported, and were 0 because our prompts are under the 1,024-token minimum |
| 24 | **A2A (Agent2Agent) service** (built) | other agents can discover the FiQA agents from a card and delegate to them | built: delegation of compound questions over A2A, Coverage@10 0.11 → 0.34 without an LLM. [21_a2a.md](../topics/21_a2a.md) |
| 25 | **Agent memory** (built) | follow-ups and user facts ("I live in Canada") | built: follow-up R@10 0.103 → 0.439 (rule) / 0.455 (LLM rewrite); PII-redacted, per user, expiring, deletable. [22_agent_memory.md](../topics/22_agent_memory.md) |
