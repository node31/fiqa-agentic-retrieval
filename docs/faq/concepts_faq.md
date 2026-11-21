# Concepts FAQ: general questions, answered from this project

General questions on the topics this project covers. For each one I give the general answer first, then what I did
and measured here. Numbers come from `results/` and DESIGN.md (dev split, 500 queries, unless I say test). Questions
about this specific repo are in the [project FAQ](project_faq.md).

## Information retrieval

### How does BM25 work?
BM25 scores a document by summing, over the query terms it contains, the term's IDF times a saturating function of its
frequency in the document, normalised by document length. `k1` controls how fast repeated terms stop adding score and
`b` how strongly long documents are penalised. It needs no training and is very good at rare exact terms such as names,
tickers and acronyms. In my project: I wrote it myself (`src/retrieval/bm25.py`, k1 = 1.2, b = 0.75) so I could control
corpus statistics across shards. It scores 0.2642 nDCG@10 on FiQA dev at 1.2 ms p50, the weakest quality and the
fastest by far.

### How does dense retrieval differ from BM25?
A bi-encoder embeds query and passage separately into vectors, and relevance is their cosine similarity, so it can
match paraphrases that share no words. Its weak spots are rare entities the model never learned and anything past its
token limit. In my project: all-MiniLM-L6-v2 reaches 0.3881 nDCG@10 vs BM25's 0.2642, and bge-small 0.3981. One of my
failure cases (`results/failures.md`) is the opposite: "EFU" and "NAFA" are Pakistani fund houses that MiniLM splits
into meaningless subwords, so dense never finds the passage BM25 ranks first.

### Why combine BM25 and dense, and how should you fuse them?
They fail on different queries, so a fused list usually beats both. Reciprocal rank fusion only uses ranks, so it needs
no score calibration, but it gives every list an equal vote by default. Linear fusion normalises each list's scores and
weights them, which lets you lean on the stronger retriever. In my project: plain RRF scored 0.3733, below dense alone,
because BM25 is 12 points weaker. Linear min-max fusion with 0.7 on the dense side reached 0.4190, and that became the
shipped default.

### When does a cross-encoder reranker help?
A cross-encoder reads query and passage together, so it is more accurate per pair than a bi-encoder, but it costs one
model pass per candidate and can only rerank a short list. It helps when the first stage has the right documents in
its top-k but in the wrong order, and when the reranker was trained on similar data. In my project it hurt: the
ms-marco MiniLM cross-encoder gave 0.4058 at k = 10 and 0.3771 at k = 50, vs 0.4190 without it, at 338.6 ms p50 for
k = 10. I put it down to domain shift, 256-token truncation of long forum answers, and a strong first stage.

### Explain nDCG, MRR and Recall@k. Which would you optimise?
Recall@k is the share of relevant documents in the top k. MRR is one over the rank of the first relevant hit. nDCG
gives each relevant hit a log-discounted gain and divides by the best possible ordering, so it rewards putting good
documents near the top. For RAG I care most about Recall@k at the context size the LLM reads, with nDCG as the ranking
check. In my project: I report all three plus Coverage@10 for two-part questions, the share of queries with at least
one gold passage from each half in the top 10. Ideal DCG uses all judged-relevant docs, not just the retrieved ones,
which is a common bug (`src/metrics.py`).

### How do you keep evaluation honest with dev and test splits?
Tune every choice on dev, freeze it, and run test once, because each look at test is a little more tuning. Report
group sizes for stratified numbers, since small groups are noise. In my project: FiQA has 5,500 train, 500 dev and
648 test queries. I tuned α and every other choice on dev and ran test exactly once for the final configuration:
0.3968 nDCG@10 vs 0.4190 on dev (`results/runs/FINAL-test-hybrid-linear-minilm.json`). The warmup queries come from
train so they don't touch dev.

### Exact (flat) search or approximate nearest neighbours?
Flat search compares the query with every vector: exact, simple, and linear in N. ANN indexes such as HNSW or IVF-PQ
skip most vectors and trade a little recall for much lower latency or memory, so they win once flat search stops being
small next to everything else. In my project flat was the right call: over a 19K-vector shard it takes 1.4 ms and over
all 57.6K vectors 5.3 ms (`results/microbench.json`), next to an 8 ms query encoder. At 100M vectors flat would be
153.6 GB and seconds per query, so I'd use IVF-PQ there.

### How do you choose an embedding model?
Leaderboards are a starting point, but the right model is the one that wins on your data within your latency and
memory budget, measured inside the full pipeline. In my project bge-small beat MiniLM dense-only (0.3981 vs 0.3881) but
added about 12 ms per query on CPU, and in hybrid it was slightly worse (0.4155 vs 0.4190), because BM25 already
supplied what it added. I kept MiniLM as primary and used bge-small as the migration target.

### What do you do when the labels themselves are imperfect?
Sparse qrels count unjudged relevant documents as misses, so every system looks worse than it is, and some failures
are label problems. I look at failures by hand before "fixing" them. In my project 161 of 500 dev queries have
Recall@10 = 0. In one case the query is a forum title ("Clarify on some Stocks Terminology") and the actual question
was in the post body, which the benchmark drops. I report that class separately and don't tune for it.

## Distributed systems

### How would you shard a search index?
Partition documents across shards by a stable hash of the doc id, give each shard a complete local index, and have a
router fan out every query and merge. Hash partitioning balances well but makes resharding expensive; consistent
hashing or a shard map fixes that. In my project: `crc32(doc_id) % 3` gives 19,167 / 19,254 / 19,217 docs. I used crc32
and not Python's `hash()`, because `hash()` is salted per process and writes would land on the wrong shard. Going to 4
shards would move about three quarters of the docs.

### What does scatter-gather need to stay correct?
Each shard must return enough candidates (its local top-D for a global top-D), scores must be comparable across shards,
and anything that depends on the whole list, like fusion, must happen after the global merge. In my project all three
rules were built in from the start, and the sharded service matches single node to 4 decimals on BM25, dense and
hybrid. `tests/test_store.py::test_sharded_router_equals_single_node` guards it.

### Why does tail latency grow with fan-out?
A query is as slow as its slowest shard. If each shard is slow 1% of the time, a 100-shard query is slow
1 − 0.99^100 ≈ 63% of the time. Fixes are hedged or backup requests, replicas, and timeouts with partial results. In my
project, going from 1 to 3 shards moved p50 from 23.4 to 28.6 ms but p95 from 26.3 to 40.5 ms: the tail grew much
more than the median.

### Global vs local IDF: does it matter?
BM25's IDF and average length come from corpus statistics, so with shard-local statistics the same document scores
differently depending on its shard, and merged scores aren't comparable. Elasticsearch's default uses local stats; its
DFS mode gathers global stats first. In my project local IDF gave BM25 0.2653 vs 0.2642 with global stats, inside
noise, and was about 4 ms faster, because random hashing makes every shard's df ratios nearly equal. I kept the DFS
round trip for exact equivalence and because skewed shards (by topic or time) would break local IDF.

### How do you make writes idempotent?
Give every write an identity and a version, apply it only if the version is newer than what is stored, and keep
tombstones for deletes so a late, older write can't resurrect a document. Then clients and routers can retry on
timeout without risk. In my project each write is `(doc_id, version)`: equal is a replay, older is stale, both no-ops.
The router retries twice with backoff (`Router.write`). In async mode I had to compare against accepted versions, not
applied ones, or a queued v2 could be overwritten by v1.

### What is a write-ahead log and what can go wrong with it?
Append each change to a log and fsync it before acknowledging, then rebuild state on restart by replaying the log. The
classic edge case is a crash mid-append that leaves a torn record. In my project the WAL lives in
`src/service/wal.py`, sync-mode records carry the embedding vectors so replay needs no model, and replay uses the same
version check, so replaying twice is harmless. I also shipped a torn-tail bug and fixed it (see the last section). The
log is never truncated yet; production needs snapshots.

### What consistency and freshness guarantees would you give a search index?
State them precisely: what a reader can see during a write, and how soon after the ack a write is searchable. Most
engines give per-document atomicity and near-real-time visibility, not cross-document transactions. In my project a
per-shard RW lock means a reader sees a document before or after a write, never with BM25 and dense out of sync. Sync
mode promises the first search after the ack finds the write; async mode promises within 1 s. Both have integration
tests over real processes.

### Synchronous or asynchronous apply?
Sync applies before the ack: simple read-your-writes, slower acks. Async acks once the write is durable and applies in
the background: higher throughput and faster acks, but a visibility lag you must bound. In my project, under 8 writers
and 2 readers (`results/write_load.json`), async acked 94.2 writes/s vs 57.8, ack p50 78 vs 128 ms, with visible lag p95
466 ms and max 583 ms. Searches under that load were slower (p50 184 vs 131 ms). I kept sync as the default.

### What is back-pressure and why do you need it?
When producers outrun consumers, a queue grows without bound and any latency promise silently breaks. Back-pressure
pushes the cost back to the producer, by blocking, rejecting or slowing it, before the queue gets too long. In my
project async mode delays the ack when the predicted apply time for a new write exceeds 0.5 s, which keeps the
"visible within 1 s" promise. My first version only looked at the oldest write's age and missed fast bursts;
`test_async_backpressure_keeps_visible_lag_under_one_second` caught it.

### How do you migrate an index to a new embedding model with no downtime?
Blue/green: build the new index next to the old one, dual-write new changes to both, backfill existing documents, cut
over with a flag once the new index is complete everywhere, and keep the old one for rollback. Dual-write must start
atomically with the backfill snapshot, and the backfill must never overwrite a newer vector. In my project
`make migrate`, `make cutover` and `make rollback` do this across 3 shards. Under 4 query threads and a writer, 2,045
queries ran with 0 failures, and all 116 docs written during the migration were found by both models
(`results/migration_demo.json`).

### When and how do you roll back?
Roll back on a quality regression, a latency SLO breach or new errors, and make rollback cheap by keeping the old path
live and current until the new one has survived a full traffic cycle. In my project rollback is a flag flip, instant
because dual-write kept the old index current; it stops being instant only after `make finalize` drops the old index.
The router persists the flag, so a router restart keeps the decision.

### A shard is down. Do you fail the query or return partial results?
It depends on the caller. A search box usually prefers most of the answer, clearly flagged, while an audit or an eval
wants an error. In my project every shard call has a 0.5 s timeout. The default policy returns what arrived with
`partial: true, missing_shards: [i]`, and `FAILURE_POLICY=fail` returns 503. An integration test hard-kills a shard,
checks the flag, restarts it and finds the acknowledged write again from the WAL.

## LLM agents

### How does tool calling work under the hood?
The model only emits text. You describe the tools and an exact output format, parse its reply, run the tool yourself,
append the result to the next prompt, and loop until it says it's done. Native function calling is the same loop with
the schema enforced by the provider. In my project I used a plain JSON action protocol so one code path works for a
fake LLM, the Copilot CLI and Azure, and every step lands in a trace. I never trust tool arguments: unknown strategies
fall back to hybrid, and returned doc ids are kept only if the agent actually saw them.

### Single agent or multi-agent?
A single agent with a loop is simpler, cheaper to debug, and often enough. Splitting into roles pays off when a task
decomposes into independent parts that can run in parallel, or when a role is better done in code. In my project the
multi-agent system is a planner LLM, parallel search workers with no LLM, a verifier LLM, and an orchestrator in plain
code. On compound questions with gpt-6-sol, the single agent was more accurate (R@10 0.400 vs 0.355, Coverage@10 0.40
vs 0.32), and multi-agent was cheaper and had a shorter tail (925 vs 3,380 tokens, p95 4.75 vs 9.28 s).

### What guardrails does an agent need?
A hard cap on steps and LLM calls, a token budget per user request, timeouts on every tool call, and validation of
everything the model returns, with a defined fallback for each failure. In my project one `Budget` per user query is
shared by all LLM roles and refuses a call before spending if the worst case could exceed the cap (3 calls and 8K
tokens for the orchestrator, 3 calls and 12K for the single agent). A planner error degrades to "simple", a failed
worker's list is dropped, and if every worker fails the orchestrator falls back to one direct search.

### How do you evaluate an agent?
Measure it against the cheap non-agentic baseline on the query types it's meant for, with quality, latency and cost
together, and look at per-query wins and losses, not just averages. At n = 100, bootstrap the differences. In my
project I compared baseline, single and multi on 100 plain dev queries (seed 42) and 100 compound queries. Single minus
multi on compound was +0.045 R@10 with a 95% CI of [+0.007, +0.085]: real, but small.

### How do you estimate and control LLM cost?
Count input and output tokens per query, multiply by prices, and remember that input usually dominates because each
step re-sends context. Do a dry run before spending, cap spend, and cache responses. In my project the final run used
about 655K tokens, roughly 610K of them input, for $0.79 at placeholder prices ($1 / $4 per 1M input / output tokens).
The dry-run estimate was $0.47 because it replays a fake policy that makes fewer, shorter calls, so I treat dry runs as
a lower bound. Every Azure call is gated by `--confirm-spend` and `--max-usd` and cached on disk.

### How would you route queries between cheap and agentic paths?
Use a free classifier first, spend an LLM call only when the cheap signal is unsure, and send each class to the path
that wins for it on measured quality per cost and latency. In my project a regex and length gate (`looks_simple()`)
keeps short single-clause questions on the 30 ms non-agentic path; otherwise one planner call decides. Plain questions
stay non-agentic, the single agent is an opt-in "deep search", and compound questions go to the single agent for
quality or to multi-agent under a latency or cost budget.

### What about prompt injection through retrieved content?
Any retrieved text lands in the prompt and can contain instructions. The main defence is limiting what the model can do:
delimit tool output and never let it trigger tools with side effects. In my project the agents' only tool is read-only
search, so the worst an injected passage can do is distort one ranking.

## System design

### Design a search service for 100M documents.
Shard the corpus, replicate each shard, put a router in front, and size memory first. In my project's arithmetic
(DESIGN.md §9): flat 384-d float32 vectors take 100M × 1,536 B = 153.6 GB, so IVF-PQ with 48-byte codes (4.8 GB) plus a
rerank of the top-100 from full vectors on disk. BM25 has about 6.5B postings, roughly 52 GB uncompressed, so
compressed postings scored with WAND or block-max. Passage text (about 77 GB) goes to disk or object storage, fetched
only for the final top-k. Re-embedding takes about 11 days on my 32-core box, so migrations need GPUs.

### What breaks first when you scale?
Find it with arithmetic before building. In my design the order is: flat float32 vectors per shard hit the 2 GB budget
at about 1M vectors; Python BM25 allocates an O(N) score array per query; router fan-out tail latency grows past 50
shards; WAL replay time grows without snapshots; and modulo sharding moves most docs on every resize.

### Where would you add caching?
Cache where repeated work is expensive and invalidation is cheap. In my project I'd add a router-side LRU on
(query, strategy, model) invalidated by writes to returned docs, a query-embedding cache (the encoder is the largest
CPU cost at about 8 ms per query), and a global df cache that replaces the DFS round trip, about 4 ms. None of these
is built.

### How would you make it highly available?
At least two replicas per shard, a router that load-balances and hedges slow reads, writes to a primary replicated from
its WAL, and health checks that eject bad replicas. In my project there's no replication: one dead shard means
partial results until it restarts, which the router reports per query. Replicas are the first thing I'd add.

### What would you monitor?
Latency per phase (embed, stats, scatter, merge) at p50, p95 and p99, the partial-result rate per shard, WAL size and
apply lag, index sizes per model during a migration, an offline quality canary after every deploy, and LLM tokens and
dollars per query. In my project every `/search` response carries `timings_ms`, and `/admin/status` reports doc counts,
per-model readiness, apply mode, queue stats and RSS per shard.

### How do you spend a 50 ms p95 budget?
Measure each phase first, then spend on what moves quality. In my project the query encoder is about 8 ms warm, flat
search 1.4 ms per shard and BM25 about 1 ms, and most of the rest is HTTP round trips. The chosen config lands at
40.5 ms p95 through the router, so a reranker (338.6 ms) or bge-small (about 12 ms more) didn't fit.

## Behavioural and judgment

### Tell me about a counterintuitive finding.
I expected sharding to cut latency, since each shard scans a third of the corpus. It made it worse: 1 shard was
23.4 / 26.3 ms p50 / p95 and 3 shards 28.6 / 40.5 ms, with identical quality. Per-shard compute at 19K docs is a few
milliseconds, so there was little to parallelise, and sharding added a second fan-out, three times the HTTP calls, and a
wait for the slowest shard. It changed how I describe sharding here: it buys capacity and isolation, not speed.

### Tell me about an approach that failed.
Cross-encoder reranking. I expected a gain of a few nDCG points on top of hybrid. It lowered nDCG@10 at every depth
(0.4058, 0.3964, 0.3771 for k = 10, 20, 50 vs 0.4190) and cost 338.6 ms p50 at k = 10. I think the model's web-search
training data doesn't match long forum answers, and truncation cuts the long answers that are already the hardest.
I kept the ablation in the results and left reranking off.

### Tell me about a time your first version was worse than what it replaced.
My first async write mode was supposed to beat sync on throughput and acked 47.1 writes/s vs sync's 57.8, with lag up
to 1.3 s. I had assumed batching makes embedding cheaper; on this CPU eight passages per call cost 31.5 ms each vs
25.6 ms one at a time, and one applier thread embedded serially while sync embeds in parallel request threads. I
measured, switched to 8 parallel single-doc embed workers and a per-write wait prediction, and got 94.2 writes/s.

### Tell me about a bug you found and fixed.
A crash mid-append leaves a torn last line in the WAL, and my first version just skipped it during replay. But the
reopened file kept the fragment, so the next acked write was appended onto it and became one unparseable line, and that
write vanished on the following restart. I fixed it by truncating to the last newline on open
(`WAL._truncate_torn_tail`) and added a test that crashes, recovers, writes and restarts again
(`test_wal_write_after_torn_tail_survives_next_restart`). Now I test recovery twice in a row.

### Tell me about a time the data changed your decision.
On the free dev model (claude-haiku-4.5) single and multi-agent were close on compound questions (R@10 0.327 vs
0.335) and multi used 2.9 times fewer tokens, so I planned to ship only the multi-agent path. On gpt-6-sol the single
agent became the best system: plain MRR went from 0.485 to 0.642, where haiku had lowered it to 0.432. Its last step
is an LLM rerank over snippets, which a stronger model does well, while the multi-agent LLM never ranks. I changed the
routing policy and wrote down that the eval must be re-run whenever the model changes.

### Tell me about a trade-off where you chose the slower option.
Shard-local IDF gave the same BM25 quality (0.2653 vs 0.2642) and was about 4 ms faster than my DFS round trip. I kept
DFS because the requirement was single-node equivalence, and because local IDF fails on skewed shards, which a
production system could easily have. I stated the cost in DESIGN.md and noted that a cached global df would recover
most of the 4 ms.

### What would you do differently?
I'd run the agent eval on the model I intend to ship much earlier, since the haiku results pointed me toward the wrong
routing policy for a while. I'd also put WAL snapshots and replicas ahead of the async mode, because a production
deployment needs them first.
