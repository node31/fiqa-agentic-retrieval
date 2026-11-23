# What else I'd add, as the engineering lead

[DESIGN.md](../../DESIGN.md) describes what I built for each task and what it measured. This doc is the other half: for
each task, the things I didn't build but would put on the roadmap if this became a team's product. Other parameters,
other models and datasets, other ways to scale, more checks, other kinds of updates, other failure points, other LLMs,
and the non-LLM options for the agent layer.

Nothing in this doc is built or measured. Every number quoted as "today" comes from DESIGN.md; everything else is a
proposal, and the "how I'd test it" column says what would have to be true before I'd ship it.

How to read each section:
- **Today**: one line on what exists, with the measured number that matters.
- **Ideas**: a table of `idea | why | cost or trade-off | how I'd test it`.
- **First**: the one I'd start with, and why.

## Contents

- [A1 Retrieval](#a1-retrieval)
- [A2 Evaluation harness](#a2-evaluation-harness)
- [A3 Ablations](#a3-ablations)
- [A4 Failure analysis](#a4-failure-analysis)
- [B1 Sharding and the router](#b1-sharding-and-the-router)
- [B2 Live writes](#b2-live-writes)
- [B3 Model migration](#b3-model-migration)
- [C1 Compound question set](#c1-compound-question-set)
- [C2 Single agent](#c2-single-agent)
- [C3 Multi-agent system](#c3-multi-agent-system)
- [C4 Agent evaluation and routing](#c4-agent-evaluation-and-routing)
- [Cross-cutting: ops, security, cost, data, team](#cross-cutting-ops-security-cost-data-team)
- [My top 10](#my-top-10)

## A1 Retrieval

**Today**: BM25 + all-MiniLM-L6-v2, linear fusion at α = 0.7, depth 100, exact faiss search. Dev nDCG@10 0.4190 at
20.7 ms p50 single node.

**Other parameters I never tuned**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| BM25 `k1` and `b` | I used the textbook defaults (1.2 and 0.75); FiQA answers are long forum posts, so the length penalty `b` may be wrong for them | none at query time; a small sweep | grid `k1` 0.6 to 2.0, `b` 0.3 to 1.0 on dev, BM25 alone and in hybrid |
| tokenizer: stemming, stop words, numbers, tickers | finance text has `401k`, `S&P`, `$AAPL`; my tokenizer may split them badly | a tokenizer change means a BM25 rebuild (a migration, see B3) | dev nDCG plus a list of queries with symbols and numbers |
| fusion depth (100 today) | depth 50 might be as good and faster; depth 200 might find more gold | deeper = more data over the wire from 3 shards | sweep 25 / 50 / 100 / 200, quality and p95 |
| `max_seq_length` (256 tokens) | long gold passages lose 0.044 Recall@10 in hybrid (0.464 vs 0.508) | 512 tokens roughly doubles encode time for long docs | re-embed at 512, compare the stratified "long gold" slice |
| query-adaptive α | case 1 in A4 needed more BM25 weight for a rare-acronym query | I tried one rule: 0.4190 → 0.4206, too small to adopt; a learned version might do more | learn α from query features on train, measure on dev |

**Other models and retrieval methods**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| bigger general embedders (bge-base, e5-base, gte-base, 768 dims) | usually a few points of nDCG on BEIR | about 3x the encode cost on CPU, 2x the vector memory; puts p95 over 50 ms unless the encoder moves to a GPU | dev nDCG and p95 through the router |
| fine-tune MiniLM on FiQA train (5,500 questions with labels) | in-domain training is usually the biggest single gain; the train split is unused today | a training pipeline, a model registry, and the risk of overfitting to FiQA | train on train, tune on dev, test once; also check it didn't get worse on another BEIR set |
| chunking: 150-word overlapping chunks, best chunk per doc | fixes A4 case 2 (one gold doc's cosine rose 0.464 → 0.548 in my spot check) | about 3x more vectors per doc; dedupe by doc at merge | stratified "long gold" slice; memory per shard |
| learned sparse (SPLADE) | lexical matching with learned term weights; often beats BM25 + dense | a transformer at index and query time, bigger postings | dev nDCG in place of BM25 inside the hybrid |
| late interaction (ColBERT) | token-level matching; strong on long docs | many vectors per doc: memory grows by an order of magnitude | small prototype on one shard |
| query expansion: RM3 / pseudo-relevance feedback, or doc2query at index time | finds gold passages that share no words with the query (58 of 161 failures had no gold in either top-100) | RM3 is a second BM25 pass; doc2query is an LLM or T5 pass over every doc at index time | how many of the 58 "no gold anywhere" queries get a gold hit in the top 100 |
| a reranker trained on FiQA-like data | the MS MARCO cross-encoder made things worse (0.4058 at depth 10) because it truncates long posts | 339 ms at depth 10 on CPU: needs a GPU or a distilled model | must beat 0.4190 and stay inside the latency budget |

**Other datasets to prove it generalises**

| dataset | what it would show |
|---|---|
| other BEIR sets (SciFact, NFCorpus, TREC-COVID, HotpotQA, NQ) | that α = 0.7 and MiniLM aren't FiQA-specific; BEIR is the standard way to report this |
| finance QA sets (FinQA, TAT-QA, ConvFinQA, FinanceBench) | tables and numbers, which forum posts barely have |
| multi-hop sets (HotpotQA, MuSiQue, 2WikiMultiHopQA) | real compound questions with human labels, for Part C |
| our own query logs, once there are users | the only dataset that matches real traffic |

**First**: fine-tune the encoder on FiQA train. The labels are already there, unused, and it attacks the biggest
weakness (dense quality) without changing the serving architecture.

## A2 Evaluation harness

**Today**: nDCG@10, Recall@10, MRR@10, p50/p95 cold and warm, peak RSS, stratified recall by query length and doc length;
my own metric code checked against pytrec_eval.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| significance on every comparison (paired bootstrap, or a paired t-test per query) | many of my deltas are small (bge-small hybrid 0.4155 vs 0.4190); without a CI I can't say if they're real | seconds of CPU | I already did this once for C4 (single vs multi R@10 +0.045, CI +0.007 to +0.085); make it a column in every table |
| more metrics: Recall@100, MAP, nDCG@1/3/5 | Recall@100 says what a reranker or agent could still find; nDCG@3 is what a user sees on one screen | none | a column in bench.json |
| "judged@10" (share of top-10 hits that have any label) | FiQA labels are sparse; an unlabelled hit may be relevant, which makes every number a lower bound | none to compute; fixing it needs labels | report it next to nDCG; if it's low, sample unlabelled hits for review |
| LLM-as-judge for unlabelled hits | cheap extra labels | the judge can be wrong or biased toward its own family's outputs | agree with human labels on a sample before trusting it |
| per-query regression report | an average can hide 30 queries that got worse | none | list queries whose nDCG moved by more than 0.2 between two runs |
| eval in CI on a small fixed slice | catches quality regressions before merge | a few minutes per CI run | 100 dev questions, fail the build if nDCG drops by more than a threshold |
| throughput (QPS at the p95 budget), not just latency | latency at one client hides queueing under load | a load generator | QPS where p95 crosses 50 ms, per configuration |
| more strata: topic, has-number, question vs statement | tells where to invest next | some labelling | per-slice recall |
| an online metric plan (click-through, reformulation rate, dwell time) | offline labels don't capture real user value | needs product and logging | A/B test (see C4) |

**First**: significance and per-query regressions in every report. They're cheap and they stop me from shipping noise.

## A3 Ablations

**Today**: fusion method × α sweep, rerank depth sweep, MiniLM vs bge-small, global vs shard-local IDF. One factor at a
time, on dev.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| BM25 `k1`/`b`, depth, chunk size, `max_seq_length` | the parameters listed in A1 that I never varied | CPU time | same one-at-a-time protocol |
| interactions (a small factorial: α × depth × model) | one-at-a-time misses interactions, e.g. the best α may change with the model | runs multiply | 2-level factorial, or Bayesian search with Optuna |
| systems knobs: torch threads, batch size, float16 or int8 vectors, int8 encoder | latency and memory, not just quality | quantisation can cost a little quality | p95 and RSS per setting, nDCG must stay within 0.005 |
| run-to-run variance | timing numbers move between runs; I report one run | 3 to 5 repeats | report mean ± spread for latency rows |
| ablation hygiene: a held-out "dev2" slice | after many sweeps on the same 500 questions, dev is partly overfit | 500 fewer questions to tune on, or carve them from train | the chosen point must hold on dev2 before the one test run |

**First**: the interaction check between α and the dense model. If I ever switch models (B3), α should be re-tuned, and
I want evidence for that rule.

## A4 Failure analysis

**Today**: 161 of 500 dev queries have Recall@10 = 0; I analysed 3 by hand. 58 of the 161 have no gold in either
retriever's top 100.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| automatic failure buckets for all 161 | three cases don't tell me how big each problem is | a script per bucket | buckets like: gold beyond depth 100 in both; gold in BM25 only; gold in dense only; gold is long; query is a title only |
| a label-noise audit | some "failures" are probably relevant passages that were never labelled | an hour of reading per 50 queries | read the top 3 of 50 failures and count how many are actually answers |
| a failure dashboard per release | so a fix for one bucket doesn't quietly break another | a small HTML report, like the tour report | bucket sizes before and after each change |
| map each bucket to a fix with an owner | turns analysis into a roadmap | none | the bucket shrinks after the fix |

**First**: automatic buckets. They turn "three interesting stories" into "fix X helps N queries", which is what a lead
needs to prioritise.

## B1 Sharding and the router

**Today**: one router, 3 shards, `crc32(doc_id) % 3`, global BM25 statistics, 0.5 s timeout with partial results, 0.0%
quality gap vs single node, 28.6 / 40.5 ms p50 / p95.

**Other ways to scale**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| replicas per shard, with hedged reads | today a dead shard means partial results; a slow one sets p95 | 2x to 3x the machines | kill one replica under load: no `partial: true`, p95 within budget |
| consistent hashing or range partitions | `% N` moves almost every doc when N changes | a ring or a partition map to maintain | reshard 3 → 4 and count moved docs |
| more router replicas behind a load balancer | the router is a single point of failure | routers must agree on `active_model` (move it to a small shared store) | kill a router under load |
| cache the global df table at the router | removes the DFS round trip (~15 ms of the router's time in my timing breakdown) | the cache can drift from the shards after writes | refresh it periodically; nDCG must stay equal to single node |
| query result cache for popular queries | real traffic repeats queries | must be invalidated by writes | hit rate on a replayed log |
| gRPC instead of JSON over HTTP | smaller messages, less parsing for top-100 lists | another protocol to maintain | p50 and p95 through the router |
| an ANN index (HNSW, IVF-PQ) per shard | flat search stops being cheap at tens of millions of docs per shard | recall loss, build time | recall vs exact at matched latency (DESIGN.md §9 has the sizing) |
| autoscaling on QPS and p95 | traffic changes during the day | needs replicas first | load test ramp |

**Other failure points to plan for**

| failure | today | what I'd add |
|---|---|---|
| router crashes | everything is down until it restarts | replicas (above) |
| network partition between router and one shard | looks like a dead shard: partial results | alert on partial rate; replicas |
| a shard is slow, not dead (GC pause, noisy neighbour) | waits up to 0.5 s, then partial | hedged read to a replica after the shard's p95 |
| out of memory on a shard | Docker kills it; partial results until restart | memory alerts at 80% of `mem_limit`; refuse writes before OOM |
| a bad deploy on all shards at once | all shards break together | roll out one shard (or replica) at a time with a canary |
| clock or config drift between shards (different `N_SHARDS`) | docs routed to the wrong shard, silently | the shard checks its own id and count on every request and refuses on mismatch |
| overload | requests queue without bound | load shedding: reject with 429 above a queue length |

**First**: replicas with hedged reads. They fix both the single biggest p95 cost (waiting for the slowest shard) and
the partial-results problem.

## B2 Live writes

**Today**: idempotent versioned upsert and delete, tombstones, WAL with fsync, sync mode (searchable on the next query,
57.8 writes/s) and async mode (within 1 s, 94.2 writes/s, lag p95 466 ms).

**Other kinds of updates the system will see**

| update | what it needs that I don't have | how I'd test it |
|---|---|---|
| bulk import (millions of docs) | a batch path that builds index segments offline and swaps them in, instead of one write at a time | import 1M docs while serving queries: p95 within budget, all docs searchable at the end |
| partial update (metadata only: title, tags, date) | fields besides text, and filters at search time; an update that doesn't re-embed | update metadata, check vectors weren't recomputed |
| time-to-live / expiry | a background sweeper that deletes expired docs with a new version | docs disappear on time, tombstones stay |
| right-to-be-forgotten delete | a delete that also purges the text from the WAL and backups, not just a tombstone | grep every on-disk file for the deleted text |
| multi-doc transaction (replace a thread of posts at once) | atomic apply across docs, possibly across shards | readers never see half a thread |
| schema change (new field, new tokenizer) | versioned schemas and a migration like B3 | old and new docs both searchable during the change |
| change data capture from a source database | a consumer that turns DB changes into upserts with the DB's version | kill the consumer mid-stream, restart, compare counts with the source |

**More checks and guarantees**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| server-assigned versions or compare-and-set | today correctness depends on clients sending increasing versions | a round trip to read the current version | two writers racing; exactly one wins |
| WAL checksums per record | detect corruption, not just a torn last line | a few bytes per record | flip a byte, restart: the shard refuses or truncates loudly |
| WAL snapshots and truncation | replay time grows forever today | snapshot logic | restart time stays flat as writes grow |
| tombstone cleanup and BM25 compaction | memory and score time grow with deletes and new docs | a background job | memory after 1M upserts and deletes |
| an invariant checker | BM25 doc count = dense count = doc store count, per shard, per model | a cheap periodic check | inject a bug that skips the dense index: the checker fires |
| a replay-determinism test | WAL replay must give exactly the same index | test time | write, crash, replay, compare search results to a never-crashed twin |
| a chaos test for writes | kill a shard mid-write, again and again | CI time | no acked write is ever lost, no unacked write half-appears |

**First**: WAL snapshots with truncation, then the invariant checker. Without them a long-running shard gets slower to
restart every day, and a silent index mismatch would only show up as worse search.

## B3 Model migration

**Today**: dual-write + throttled backfill, cutover by a router flag when every shard is ready, instant rollback,
finalize. 2,045 queries during the demo, 0 failed.

**Other migrations the same machinery should handle**

| migration | what's different |
|---|---|
| a new BM25 tokenizer or `k1`/`b` | the lexical index needs the same build-next-to-it-and-switch treatment; today only dense models have it |
| a new fusion α or strategy | no index change: a config flag, but it still needs a canary and rollback |
| a different vector dimension (768) | memory doubles during dual-write; check headroom first |
| a new index type (flat → HNSW) | quality can drop slightly; needs a recall check before cutover |
| a shard count change (3 → 4) | the hardest one: docs move between shards; needs consistent hashing (B1) and dual routing during the move |

**Safer rollouts**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| shadow traffic | send real queries to the new model too, compare, don't serve | double query cost during the shadow | overlap@10 and latency of old vs new on the same queries |
| canary percentage | serve 1%, then 10%, then 100% | the router needs per-request model choice | metrics per arm |
| automatic rollback | today a person decides; a quality or latency alarm should roll back on its own | false alarms roll back good changes | inject a slow model: rollback within a minute |
| a quality canary | a fixed set of 100 labelled queries run every few minutes against the live model | small | nDCG on the canary set is a dashboard line with an alert |
| versioned config in one place | today `active_model` lives in `router_state.json` | needs a shared store once routers have replicas | two routers always agree |

**First**: the quality canary. It's the signal the rollback decision needs, and every later automation depends on it.

## C1 Compound question set

**Today**: 100 compound questions built by pairing unrelated dev questions (cosine < 0.25) and rewriting each pair with
an LLM; gold = union of both halves' labels; 10 hand-checked, all good.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| more question shapes | today every question is "A and also B"; real users ask comparisons ("Roth or traditional?"), conditionals and follow-ups | each shape needs its own gold logic | per-shape recall for each system |
| human-written questions | an LLM rewrite has a recognisable style an LLM planner may find easy | paid annotators or time | do the agents score differently on human vs LLM questions? |
| mine real compound questions from logs | the true distribution | needs users and privacy review | share of traffic that is compound |
| use an existing multi-hop set (HotpotQA, MuSiQue) | human labels and real dependency between halves ("who founded the company that…") | a different corpus | run the same three systems |
| check all 100 rewrites, not 10 | 10% is a spot check | an hour or two | count rewrites that lost a half |
| a bigger set | n = 100 gives wide CIs (single vs multi Coverage CI was +0.01 to +0.15) | LLM cost to build, still cheap | CI width roughly halves at n = 400 |

**First**: a bigger set. Every Part C decision I made rests on n = 100, and the CIs say that's barely enough.

## C2 Single agent

**Today**: one free hybrid search, then up to 3 LLM steps with a JSON protocol, 200-character snippets, a budget of
3 calls and 12K tokens. With `gpt-6-sol`: compound R@10 0.400, p95 9.3 s, 3,380 tokens per question.

**Other LLMs**

| kind | example families | what changes | how I'd test it |
|---|---|---|---|
| large hosted reasoning models | the frontier model families from the big providers | best ranking judgement; slowest and most expensive; reasoning effort is a latency knob | the C4 table, per model |
| small hosted models | the "mini" / "haiku" / "flash" tiers | 3x to 10x cheaper and faster; my run with claude-haiku-4.5 showed the final ranking step can hurt (plain MRR 0.485 → 0.432) | same table; watch MRR on plain questions |
| open-weight models served locally | Llama, Qwen, Mistral, Phi, Gemma families, quantised (llama.cpp, vLLM) | no data leaves the machine, fixed cost; needs a GPU for useful speed, and quality varies a lot | same table plus tokens/s on our hardware |
| a model fine-tuned for this task | a small open model trained on traces from the big one | cheap at serve time; needs a training pipeline and a trace dataset | must match the big model on the C4 table |

The lesson from my two runs: the best architecture changed with the model. So the model is a parameter of the routing
policy, not a fixed choice.

**Other knobs**

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| step limit (3 today), snippet length (200 chars) | more steps may help compound questions; longer snippets help judgement | tokens and latency grow | sweep both, plot R@10 against tokens |
| prompting: few-shot examples, a ReAct-style "thought" field | can steady small models | more input tokens | same table |
| structured output / native tool calling everywhere | fewer malformed-JSON failures | backend-specific code | count parse failures |
| let the agent set `alpha` and `k` | the router's `/search` already accepts them; the agent's tool only exposes `query` and `strategy` | more ways to go wrong | traces: does it use them sensibly? |
| a streaming answer | users see something before 9 s | UI work | time to first result |

**Instead of an LLM**

| alternative | what it would do | where it falls short |
|---|---|---|
| rule-based splitting on "and", "also", "?" | split compound questions for free | misses implicit compounds; splits phrases like "stocks and bonds" |
| a seq2seq query rewriter (a small T5 fine-tuned on question → sub-questions) | decomposition in ~10 ms on CPU | needs training pairs (the LLM could generate them once) |
| pseudo-relevance feedback (RM3) | a second search informed by the first, like the agent's step 2 | no judgement about which results answer the question |
| a cross-encoder or learned reranker as the final step | what the agent's `finish` step really does, at a fraction of the cost | the MS MARCO one didn't help here; would need a FiQA-trained one |

**First**: try an open-weight model served locally. If it gets close to `gpt-6-sol` on the C4 table, the cost and
data-privacy questions mostly go away.

## C3 Multi-agent system

**Today**: a free gate, an LLM planner (up to 3 sub-queries), workers that search without an LLM, one batched LLM
verifier, round-robin slot interleave. Compound R@10 0.355 at p95 4.7 s and 925 tokens; 7 plain questions wrongly split.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| a stricter split rule | the 7 false splits cost the whole plain-set loss (their recall fell 0.824 → 0.514) | may miss a few real compounds | require two independent needs and sub-query results that overlap by less than half; re-run both sets |
| a final rerank call by the strong model | the single agent wins because its last step ranks; multi never ranks | one more LLM call, ~2 s | does multi + rerank reach the single agent's R@10 at a lower p95? |
| reserve ~3 slots for the full question when a worker fails | today the survivor fills all 10 slots | slightly fewer slots per half in the normal case | the partial-failure test plus the compound set |
| other topologies: debate (two agents argue over the ranking), map-reduce (many workers, one reducer), hierarchical (sub-planners) | different cost / quality points | more calls, more ways to fail | same table |
| per-role budgets and models | a cheap model for planning, a strong one for verification | more configuration | cost per question at equal R@10 |
| cache plans by question | repeated questions skip the planner | cache invalidation when prompts change | hit rate on replayed traffic |
| adaptive slot split by sub-query confidence | 5 + 5 is fair, not optimal | needs a confidence signal | R@10 vs fixed split |

**Instead of an LLM planner**

| alternative | trade-off |
|---|---|
| a dependency-parse splitter (spaCy) on coordinating conjunctions | free and fast; brittle on informal forum English |
| a small classifier "compound or not" + the T5 rewriter from C2 | two small models instead of one LLM call; needs training data |
| embedding clustering of query parts | splits by topic without understanding; works for "A and also B" shapes only |

**First**: the final rerank call. It's one change that tests my main hypothesis about why single beats multi.

## C4 Agent evaluation and routing

**Today**: baseline, single and multi on 100 plain and 100 compound questions; a hand-written routing policy (gate,
then planner, then single or multi by budget); placeholder prices of $1 / $4 per 1M tokens.

| idea | why | cost or trade-off | how I'd test it |
|---|---|---|---|
| a learned router | today's gate is hand rules; a small classifier on query features (length, conjunctions, embedding) can choose the path | needs labels: which path won per question (I have them from the eval) | train on half the eval, test on the other; compare with the rule gate |
| cost-aware routing with real prices | my dollar figures are placeholders | none once prices are known | cost per 1,000 questions per policy |
| answer quality, not just retrieval | users want an answer; R@10 is a proxy | an answer-generation step and a judge | human-rated sample, then an LLM judge that agrees with it |
| online A/B test | offline labels aren't user value | needs users and logging | click-through, reformulation rate, time to answer |
| drift monitoring | the planner's behaviour changes when the provider updates the model | small | weekly re-run of the eval on cached-off calls; alert on change |
| more agent metrics: tool-call success rate, steps used, budget exhaustion rate, parse failures | explain *why* a system is slow or expensive | logging | per-system dashboard |
| eval the agents on more than one LLM in CI | the haiku and gpt-6-sol runs led to different decisions | LLM cost | the routing decision table per model |

**First**: the learned router. The data to train it already exists in `results/agent_eval.json`, and it replaces the
part of the policy I'm least sure about.

## Cross-cutting: ops, security, cost, data, team

**Operations**

| idea | why |
|---|---|
| SLOs: search p95 ≤ 50 ms, partial rate < 0.1%, write ack p95, async lag p95 < 1 s | turn "it works" into numbers someone is paged for |
| tracing across router → shards → LLM (OpenTelemetry) | `timings_ms` already exists per request; traces connect it to agent steps |
| alerts: partial rate, WAL size, memory at 80%, canary nDCG, LLM error rate, budget exhaustion | each maps to a failure in B1, B2, B3 or C |
| runbooks for: shard down, migration rollback, LLM provider outage | the Docker tour already scripts the first two; turn them into runbooks |
| backups of the WAL and state, with a tested restore | today the WAL is the only copy |

**Security**

| idea | why |
|---|---|
| authentication on the API, admin endpoints separated | today anyone who can reach port 8000 can call `/admin/cutover` |
| rate limits per client | one client can saturate the shards |
| prompt-injection defence | passages are user-written forum posts that go into LLM prompts; a post saying "ignore previous instructions" is an attack vector. Keep tools read-only, treat snippets as data, and test with planted posts |
| secrets in a vault, not `.env` | the LLM keys |
| dependency and container scanning in CI | standard hygiene |

**Cost**

| idea | why |
|---|---|
| a cost model per query path | baseline costs ~0; agents cost tokens and seconds; the routing policy is really a cost policy |
| CPU vs GPU for encoding | the corpus took 567 s to embed on 32 CPU cores; a GPU changes both indexing and model choice (A1) |
| LLM response caching in production | already on disk for evals; repeated questions are free |

**Data and privacy**

| idea | why |
|---|---|
| dataset licences recorded per source | FiQA and any new dataset (A1) |
| PII scan on ingested text | forum posts can contain names and account details |
| what goes to the LLM provider | snippets leave the machine; a local model (C2) avoids it |
| retention and deletion policy | right-to-be-forgotten needs the B2 purge |

**Team and process**

| idea | why |
|---|---|
| owners per area: retrieval quality, serving, agents | each has its own metrics and on-call |
| CI gates: unit tests, the quick eval slice (A2), the Docker tour | the tour already checks the whole stack end to end |
| design reviews for anything that changes a guarantee (freshness, idempotency, partial results) | those are the promises other teams build on |
| a quarterly "test split" discipline | the test set is touched once per decision, by one person |

## My top 10

In the order I'd do them, with the reason in one line:

1. **Significance + per-query regressions in every eval** (A2): stop shipping noise; cheap.
2. **Quality canary** (B3): the signal every rollback and alert needs.
3. **Replicas with hedged reads** (B1): fixes p95 and partial results together.
4. **WAL snapshots, truncation and an invariant checker** (B2): keeps restarts fast and catches silent index drift.
5. **Fine-tune the encoder on FiQA train** (A1): the largest likely quality gain, with labels already in hand.
6. **Chunked indexing for long passages** (A1): fixes a measured gap (0.464 vs 0.508 Recall@10).
7. **Automatic failure buckets** (A4): turns anecdotes into a sized roadmap.
8. **A bigger compound set** (C1): every Part C decision rests on n = 100.
9. **A learned router** (C4): replaces hand rules with data I already have.
10. **Auth on the API and prompt-injection tests** (security): needed before anyone outside the team uses it.
