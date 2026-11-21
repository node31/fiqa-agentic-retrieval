# Glossary

Terms used in this repo, in plain words, with the doc or code that covers each one.

| Term | Meaning | Where |
|---|---|---|
| ANN | Approximate nearest neighbour search: faster than exact search, may miss some true neighbours | [03](03_vector_indexes.md) |
| Async apply | Write mode that acks after the WAL and applies (embeds and indexes) in the background. Searchable within 1 s, not instantly. The opposite is sync apply (apply, then ack) | [09](09_live_updates_consistency.md), [ADR 0013](adr/0013-async-apply-mode.md) |
| avgdl | Average document length in tokens across the corpus; BM25 normalises by it | [01](01_lexical_retrieval_bm25.md) |
| Back-pressure | Slowing producers when the consumer falls behind. Here the async ack waits while the predicted apply time is over 0.5 s | [09](09_live_updates_consistency.md), `src/service/async_apply.py` |
| Backfill | Filling a newly created index with all existing documents in the background | [10](10_model_migration.md) |
| BEIR | A benchmark suite of retrieval datasets; FiQA is one of them | [00](00_rag_primer.md) |
| Bi-encoder | Embeds query and passage separately, so passage vectors can be computed ahead of time | [02](02_dense_retrieval.md) |
| Blue/green | Run old (blue) and new (green) side by side, switch traffic with a flag, keep blue for rollback | [10](10_model_migration.md) |
| BM25 | Classic lexical ranking function: IDF × saturated, length-normalised term frequency | [01](01_lexical_retrieval_bm25.md), `src/retrieval/bm25.py` |
| Budget (agent) | Per-query cap on LLM calls and tokens, shared by all agents | [13](13_multi_agent_orchestration.md), `src/agent/llm.py` |
| Cold start | The first requests after a process starts, which pay one-off costs (model load, caches) | [07](07_benchmarking_latency_memory.md) |
| Compaction | Merging segments and dropping deleted entries to reclaim space. Not built here | [ADR 0001](adr/0001-own-bm25.md), `src/retrieval/bm25.py` |
| Consistent hashing | Shard assignment on a hash ring; adding a shard moves only about 1/N of the keys. The repo uses plain hash modulo N instead, which moves about (N-1)/N | [08](08_sharding_scatter_gather.md), `src/index/sharding.py` |
| Cosine similarity | Cosine of the angle between two vectors; equals the dot product for unit vectors | [02](02_dense_retrieval.md) |
| Coverage@10 | 1 if the top 10 contains gold from every half of a compound question | [04](04_ir_metrics.md), [14](14_agent_evaluation_cost.md) |
| Cross-encoder | Reads query and passage together; accurate, but needs one pass per pair (used for reranking) | [06](06_reranking.md) |
| CSC / CSR | Compressed sparse column / row matrix formats. Columns are postings lists, rows are a doc's terms | [01](01_lexical_retrieval_bm25.md), `src/retrieval/bm25.py` |
| Cutover | The moment traffic switches to the new version | [10](10_model_migration.md) |
| Dense retrieval | Retrieval by nearest neighbours in embedding space | [02](02_dense_retrieval.md) |
| df | Document frequency: the number of docs that contain a term | [01](01_lexical_retrieval_bm25.md) |
| DFS phase | "Distributed frequency statistics": a round trip that gathers global df and N before scoring | [08](08_sharding_scatter_gather.md), [ADR 0004](adr/0004-dfs-global-stats.md) |
| Dual-write | Writing every update to both the old and new indexes during a migration | [10](10_model_migration.md) |
| Embedding | A fixed-length vector that represents a text's meaning | [02](02_dense_retrieval.md) |
| Freshness | How soon after a write it becomes visible to searches | [09](09_live_updates_consistency.md) |
| Fusion (RRF / linear) | Combining several ranked lists into one | [05](05_hybrid_fusion.md), [16](16_ablations.md) |
| Gold / qrels | The judged-relevant passages for a query (query relevance judgements) | [04](04_ir_metrics.md) |
| Hybrid retrieval | BM25 and dense results, fused | [05](05_hybrid_fusion.md) |
| Idempotent | Applying an operation twice has the same effect as applying it once | [09](09_live_updates_consistency.md) |
| IDF | Inverse document frequency: rare terms get a higher weight | [01](01_lexical_retrieval_bm25.md) |
| Interleave | Round-robin merging of lists (A1, B1, A2, B2, ...) | [13](13_multi_agent_orchestration.md), [ADR 0010](adr/0010-slot-allocation.md) |
| Inverted index | A map from each term to the docs that contain it (postings) | [01](01_lexical_retrieval_bm25.md) |
| k1, b | BM25 settings: tf saturation and length-normalisation strength | [01](01_lexical_retrieval_bm25.md) |
| Merge (scatter-gather) | Combining per-shard top-k lists into a global top-k | [08](08_sharding_scatter_gather.md) |
| Min-max normalisation | Rescaling scores to [0, 1] within one list | [05](05_hybrid_fusion.md) |
| MRR | Mean reciprocal rank of the first relevant result | [04](04_ir_metrics.md) |
| nDCG | Normalised discounted cumulative gain: rewards relevant docs near the top | [04](04_ir_metrics.md) |
| p50 / p95 | Latency percentiles: 50% / 95% of requests are faster than this | [07](07_benchmarking_latency_memory.md) |
| Partial result | A response that is missing some shards' contributions and is flagged as such | [08](08_sharding_scatter_gather.md), [ADR 0006](adr/0006-partial-results.md) |
| Planner | The agent that decides whether and how to split a question | [13](13_multi_agent_orchestration.md) |
| Postings list | The docs (and term frequencies) for one term | [01](01_lexical_retrieval_bm25.md) |
| PQ (product quantisation) | Compressing vectors into a few bytes for huge indexes | [15](15_scaling_to_100M.md) |
| RAG | Retrieval-augmented generation | [00](00_rag_primer.md) |
| Read-your-writes | A client always sees its own completed write in its next read. Sync mode gives it; async mode gives it after `POST /admin/flush` | [09](09_live_updates_consistency.md) |
| Recall@k | Fraction of relevant docs found in the top k | [04](04_ir_metrics.md) |
| Reranking | Rescoring a short candidate list with a stronger, slower model | [06](06_reranking.md) |
| Rollback | Switching back to the previous version | [10](10_model_migration.md) |
| Router | The process that fans queries out to shards and merges the results | [08](08_sharding_scatter_gather.md), `src/service/router.py` |
| RSS | Resident set size: the RAM a process is using right now | [07](07_benchmarking_latency_memory.md) |
| RW lock | Readers-writer lock: many readers or one writer | `src/service/wal.py` |
| Scatter-gather | Sending a request to all shards, then collecting and merging the responses | [08](08_sharding_scatter_gather.md) |
| Shard | A process that holds a slice of the corpus and its own indexes | [08](08_sharding_scatter_gather.md) |
| Stemming | Reducing words to a common stem ("investing" becomes "invest") | [01](01_lexical_retrieval_bm25.md) |
| Tail latency / amplification | Slow outliers. With fan-out, the slowest shard sets the latency | [07](07_benchmarking_latency_memory.md), [08](08_sharding_scatter_gather.md) |
| Time split (`timings_ms`) | Per-query breakdown in every `/search` response (stats, embed wait, shard compute, network, merge, fuse, ...); the parts add up to `latency_ms` | [08](08_sharding_scatter_gather.md), [scenarios](scenarios.md) |
| Tombstone | A marker recording that a doc was deleted, at some version | [09](09_live_updates_consistency.md) |
| Token | The unit of text an LLM bills by (about 4 English characters) | [14](14_agent_evaluation_cost.md) |
| Tool calling | An LLM asks for a function call through structured output, and code runs it | [11](11_llm_tool_calling.md), [ADR 0008](adr/0008-json-action-protocol.md) |
| Vector database | Software around stored embeddings: a nearest-neighbour index, inserts and deletes, durability, sharding and an API (Qdrant, Milvus, pgvector). The router and shards in this repo are a small hand-made one | [00](00_rag_primer.md) |
| Verifier | The agent that checks the retrieved results address each sub-question | [13](13_multi_agent_orchestration.md) |
| Visible lag | Time from a write's ack until a search finds it: 0 in sync mode, under 1 s promised in async (measured p95 466 ms under load) | [09](09_live_updates_consistency.md) |
| WAL | Write-ahead log: a durable record of a write, made before applying it | [09](09_live_updates_consistency.md), [ADR 0007](adr/0007-sync-apply-wal.md) |
| Warmup | Requests sent before measuring, to exclude cold-start effects | [07](07_benchmarking_latency_memory.md) |
