# 15 · Scaling to 100M passages: the arithmetic

Related: [03 vector indexes](03_vector_indexes.md), [08 sharding](08_sharding_scatter_gather.md), DESIGN.md §9

At 57K passages everything fits in one process. At 100M (1,735× more) my assumptions break in order: memory, then
flat-search latency, then the embedding job, then router fan-out. None of the 100M setup is built; this is the arithmetic.

## The arithmetic

| item | calculation | result |
|---|---|---|
| flat vectors (384-d float32, 1,536 B) | 100M × 1,536 B | 153.6 GB, 77+ shards at my 2 GB per shard |
| IVF-PQ (m = 48 × 8 bits) | 100M × 48 B | 4.8 GB (+ ~0.4 GB ids, + centroids), 32× smaller |
| HNSW over float32 | 153.6 GB + graph (M = 32, ~256 B/vector ≈ 25.6 GB) | ≈ 180 GB, big-memory nodes |
| BM25 postings | 3.77M postings / 57K docs ≈ 65 per doc → 6.5B × 8 B | ≈ 52 GB raw, ~10 to 17 GB with delta + varint |
| text | measured 768 B/passage × 100M | ≈ 77 GB, on disk or object storage |

Flat search: I measured 1.4 ms for a 19.2K shard and 5.3 ms for 57.6K vectors (4 threads, `results/microbench.json`),
about 11K vectors per ms, bound by memory bandwidth. 100M flat is about 9 s per query, still ~140 ms per shard over 64
shards. IVF with nlist = 65,536 and nprobe = 32 scans about 0.05% of the codes (~50K), about 1 to 3 ms.

Embedding: 57,638 passages took 567 s with MiniLM on 32 cores (≈ 102 docs/s; bge-small took 1,063 s). 100M would take
about 11.3 days on this box, ~3 to 6 hours on an A100 (≈ 5 to 10K docs/s for MiniLM), under 1 hour on 8 GPUs. A model
migration re-embeds everything, so the dual-write of [10](10_model_migration.md) would last days on CPU.

Cost, as orders of magnitude: 4.8 GB PQ + ~15 GB BM25 with 2× replicas is ~8 to 16 memory-optimised nodes (≈ 64 GB
each), about $3 to 6K per month. A re-embed on 8 GPUs for ~1 h is about $30 to 100.

## What breaks first in my design

| what breaks | why | what I would do |
|---|---|---|
| flat float32 vectors per shard | the 2 GB budget runs out at ~1M vectors per shard | IVF-PQ, then rerank the top-100 with full vectors from disk or a cross-encoder |
| Python BM25 scoring | the `np.zeros(N)` score array is O(N): 8 MB per query at 2M docs per shard | score only the postings (WAND, block-max) |
| router fan-out | 64+ shards amplify tail latency ([07](07_benchmarking_latency_memory.md)) | replicas and hedged requests, or a two-level router |
| the DFS round trip | one extra fan-out per query | cache global df and refresh it periodically |
| WAL without snapshots | replay time grows without bound | snapshots and log truncation |
| modulo sharding | adding shards moves almost every doc | consistent hashing or a shard map |

References: Jégou et al., *Product Quantization for Nearest Neighbor Search* (2011); Ding and Suel, *Faster top-k
document retrieval using block-max indexes* (2011).
