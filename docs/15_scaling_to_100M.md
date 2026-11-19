# 15 · Scaling to 100M passages: the arithmetic

**Related:** [03 vector indexes](03_vector_indexes.md), [08 sharding](08_sharding_scatter_gather.md), DESIGN.md §10

## 1. Intuition

At 57K passages everything fits in one laptop process with room to spare. At 100M (1,735× more),
the assumptions break one by one: memory first, then flat-search latency, then the embedding job,
then the router fan-out. Back-of-envelope arithmetic tells you which wall you hit first.

## 2. The arithmetic

**Dense vectors (384-d float32 = 1,536 bytes each)**
- 100M × 1,536 B = **153.6 GB** of raw vectors. Doesn't fit our 2 GB/shard budget: that would be 77+ shards for vectors alone.
- IVF-PQ with 48-byte codes (m = 48 sub-quantisers × 8 bits): 100M × 48 B = **4.8 GB** (+ ~0.4 GB ids, + centroids). That is 32× smaller.
- HNSW over float32: 153.6 GB + graph (M = 32 → ~256 B/vector ≈ 25.6 GB) ≈ **180 GB**: needs big-memory nodes.

**Flat search latency**: our flat scan does ~19K vectors/ms/shard (≈1 ms for a shard). 100M vectors
flat = ~5,200 ms of single-thread compute per query. Even split over 64 shards, that's ~80 ms per
shard. **Flat is out.** IVF (nlist = 65,536, nprobe = 32) scans ~0.05% of the codes, ~50K codes per query: ~1–3 ms.

**BM25 postings**: we have 3.77M postings for 57K docs (≈ 65 per doc). At 100M docs that's ≈ 6.5B postings
× (4 B doc id + 4 B tf) ≈ **52 GB** uncompressed. Lucene-style delta + varint compression gets ~3–5×, so ~10–17 GB.

**Text store**: FiQA averages ~760 bytes/passage, so 100M ≈ **76 GB**. It belongs on disk/object storage, not in RAM. Fetch only the top-k.

**Offline embedding**: we measured 57,638 passages in 567 s (MiniLM) on 32 cores ≈ 102 docs/s.
100M / 102 ≈ **11.3 days** on this box. On a GPU (A100 ≈ 5–10K docs/s for MiniLM) it's ~3–6 hours; 8 GPUs, under 1 hour.
**A model migration = re-embed everything**, so the B3 design must assume days of dual-write.

**Cost sketch** (cloud list prices change, so treat these as orders of magnitude):
- Serving: 4.8 GB PQ + ~15 GB BM25 + replicas ×2 → ~8–16 memory-optimised nodes (≈ 64 GB each) ≈ **$3–6K/month**.
- Re-embedding 100M with 8 GPUs for ~1 h ≈ $30–100 per run. Cheap compared with engineering time.

## 3. What breaks first in *our* design

1. **Memory (flat float32 vectors in each shard process)**: hits the 2 GB budget at ~1M vectors per shard.
   Fix: IVF-PQ (+ rerank the top-100 with full vectors from disk, or with a cross-encoder).
2. **Python BM25 scoring**: the dense `np.zeros(N)` score array per query is O(N). At 2M docs/shard
   that's 8 MB zeroed and scanned per query. Fix: score only the postings (sparse accumulators / WAND / block-max).
3. **Router fan-out**: 64+ shards → tail amplification ([07](07_benchmarking_latency_memory.md)). Fix: replicas + hedged requests,
   or a two-level router tree.
4. **The DFS round trip**: one extra fan-out per query. At scale, cache global df (it changes slowly) and refresh periodically.
5. **WAL without snapshots**: replay time grows without bound. Fix: periodic snapshots + log truncation.
6. **Modulo sharding**: adding shards moves ~all docs. Fix: consistent hashing or a shard map (toy 15).

## 4. Exercises

1. Redo the vector-memory arithmetic for 768-d models (bge-base). How many 2 GB shards are needed for PQ-48 vs flat?
2. At what corpus size does flat search per shard (19K vectors/ms) exceed half the 50 ms budget?
3. Estimate re-embedding time for 100M docs with bge-small on our CPU (we measured 1,063 s for 57,638).

Solutions: [solutions/15_scaling.md](solutions/15_scaling.md)

## 5. Further reading

- Jégou et al., *Product Quantization for Nearest Neighbor Search* (2011)
- faiss wiki: "Indexing 1G vectors"
- Ding & Suel, *Faster top-k retrieval using block-max indexes* (2011)
