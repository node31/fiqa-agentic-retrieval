# Glossary

Every term used in this repo, in plain words, with where to see it.

| Term | Meaning | Where |
|---|---|---|
| **ANN** | Approximate nearest neighbour search: faster than exact, may miss some true neighbours | [03](03_vector_indexes.md), toy 06 |
| **Async apply** | Write mode that acks after the WAL and applies (embeds + indexes) in the background; searchable within 1 s, not instantly. Opposite: *sync apply* (apply, then ack) | [09](09_live_updates_consistency.md), ADR 0013 |
| **avgdl** | Average document length (in tokens) across the corpus; BM25 normalises by it | [01](01_lexical_retrieval_bm25.md) |
| **Back-pressure** | Slowing producers down when the consumer falls behind; here the async ack waits while the predicted apply time is > 0.5 s | [09](09_live_updates_consistency.md), `src/service/async_apply.py` |
| **Backfill** | Filling a newly created index with all existing documents, in the background | [10](10_model_migration.md) |
| **BEIR** | A benchmark suite of retrieval datasets; FiQA is one of them | [00](00_rag_primer.md) |
| **Bi-encoder** | Embeds query and passage *separately*; enables pre-computed passage vectors | [02](02_dense_retrieval.md), toy 09 |
| **Blue/green** | Run old ("blue") and new ("green") side by side, switch traffic with a flag, keep blue for rollback | [10](10_model_migration.md), toy 19 |
| **BM25** | Classic lexical ranking function: IDF × saturated, length-normalised term frequency | [01](01_lexical_retrieval_bm25.md), toy 02 |
| **Budget (agent)** | Per-query cap on LLM calls and tokens shared by all agents | [13](13_multi_agent_orchestration.md), toy 23 |
| **Cold start** | The first requests after a process starts, paying one-off costs (model load, caches) | [07](07_benchmarking_latency_memory.md), toy 10 |
| **Compaction** | Merging segments and dropping deleted entries to reclaim space | toy 03, drill 5 |
| **Consistent hashing** | Shard assignment on a hash ring; adding a shard moves only ~1/N of the keys | toy 15 |
| **Cosine similarity** | Cosine of the angle between two vectors; equals the dot product for unit vectors | [02](02_dense_retrieval.md), toy 04 |
| **Coverage@10** | 1 if the top 10 contains gold from *every* half of a compound question | [04](04_ir_metrics.md), [14](14_agent_evaluation_cost.md) |
| **Cross-encoder** | Reads query + passage *together*; accurate but needs one pass per pair (reranking) | [06](06_reranking.md), toy 09 |
| **CSC / CSR** | Compressed sparse column/row matrix formats; columns = postings lists, rows = a doc's terms | toy 03 |
| **Cutover** | The moment traffic switches to the new version | [10](10_model_migration.md) |
| **Dense retrieval** | Retrieval by nearest neighbours in embedding space | [02](02_dense_retrieval.md) |
| **df** | Document frequency: number of docs containing a term | toy 01 |
| **DFS phase** | "Distributed frequency statistics": a round trip to gather global df/N before scoring | [08](08_sharding_scatter_gather.md), toy 12 |
| **Dual-write** | Writing every update to both old and new indexes during a migration | [10](10_model_migration.md) |
| **Embedding** | A fixed-length vector representing a text's meaning | toy 04 |
| **Freshness** | How soon after a write it becomes visible to searches | [09](09_live_updates_consistency.md), toy 18 |
| **Fusion (RRF / linear)** | Combining several ranked lists into one | [05](05_hybrid_fusion.md), toy 08 |
| **Gold / qrels** | The judged-relevant passages for a query (query relevance judgements) | [04](04_ir_metrics.md) |
| **Hybrid retrieval** | BM25 + dense, fused | [05](05_hybrid_fusion.md) |
| **Idempotent** | Applying an operation twice has the same effect as once | [09](09_live_updates_consistency.md), toy 16 |
| **IDF** | Inverse document frequency: rare terms get higher weight | toy 01 |
| **Interleave** | Round-robin merging of lists (A1, B1, A2, B2, …) | toy 25 |
| **Inverted index** | Map term → list of docs containing it (postings) | toy 03 |
| **k1, b** | BM25 knobs: tf saturation and length normalisation strength | toy 02 |
| **Merge (scatter-gather)** | Combining per-shard top-k lists into a global top-k | [08](08_sharding_scatter_gather.md) |
| **Min-max normalisation** | Rescale scores to [0, 1] within one list | toy 08 |
| **MRR** | Mean reciprocal rank of the first relevant result | toy 07 |
| **nDCG** | Normalised discounted cumulative gain: rewards relevant docs near the top | toy 07 |
| **p50 / p95** | Latency percentiles: 50% / 95% of requests are faster than this | toy 10 |
| **Partial result** | A response missing some shards' contributions, flagged as such | [08](08_sharding_scatter_gather.md), toy 14 |
| **Planner** | Agent that decides whether and how to split a question | [13](13_multi_agent_orchestration.md), toy 22 |
| **Postings list** | The docs (and term frequencies) for one term | toy 03 |
| **PQ (product quantisation)** | Compress vectors into a few bytes for huge indexes | [15](15_scaling_to_100M.md) |
| **RAG** | Retrieval-augmented generation | [00](00_rag_primer.md) |
| **Read-your-writes** | A client always sees its own completed write in its next read. Sync mode gives it; async mode gives it after `POST /admin/flush` | [09](09_live_updates_consistency.md) |
| **Recall@k** | Fraction of relevant docs found in the top k | toy 07 |
| **Reranking** | Rescoring a short candidate list with a stronger, slower model | [06](06_reranking.md) |
| **Rollback** | Switching back to the previous version | [10](10_model_migration.md) |
| **Router** | The process that fans out queries to shards and merges results | [08](08_sharding_scatter_gather.md) |
| **RSS** | Resident set size: RAM currently used by a process | toy 11 |
| **RW lock** | Readers–writer lock: many readers or one writer | `src/service/wal.py` |
| **Scatter-gather** | Send a request to all shards, collect and merge the responses | toy 14 |
| **Shard** | A process holding a slice of the corpus and its own indexes | [08](08_sharding_scatter_gather.md) |
| **Stemming** | Reducing words to a common stem ("investing" → "invest") | toy 00 |
| **Tail latency / amplification** | Slow outliers; with fan-out, the slowest shard dominates | toy 10 |
| **Time split (`timings_ms`)** | Per-query breakdown in every `/search` response (stats, embed wait, shard compute, network, merge, fuse, …); parts add up to `latency_ms` | [08](08_sharding_scatter_gather.md), [scenarios](scenarios.md) §4.1 |
| **Tombstone** | A marker recording that a doc was deleted (at some version) | toy 16 |
| **Token** | Unit of text an LLM bills by (~4 English characters) | toy 26 |
| **Tool calling** | An LLM requesting a function call via structured output; code executes it | [11](11_llm_tool_calling.md), toy 20 |
| **Vector database** | Software around stored embeddings: nearest-neighbour index + inserts/deletes + durability + sharding + API (e.g. Qdrant, Milvus, pgvector). Our router + shards is a small hand-made one | [00](00_rag_primer.md) §4.1 |
| **Verifier** | Agent that checks retrieved results address each sub-question | [13](13_multi_agent_orchestration.md) |
| **Visible lag** | Time from a write's ack until a search finds it: 0 in sync mode, < 1 s promised in async (measured p95 466 ms under load) | [09](09_live_updates_consistency.md) |
| **WAL** | Write-ahead log: durable record of a write, made before applying it | [09](09_live_updates_consistency.md), toy 17 |
| **Warmup** | Requests sent before measuring, to exclude cold-start effects | [07](07_benchmarking_latency_memory.md) |
