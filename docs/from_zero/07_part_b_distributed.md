# Chapter 7: Part B: shards, router, live updates, model migration, deployment

> Goal: you understand how the single-machine search engine becomes a **service** of 4 processes (a router + 3 shards)
> that returns *exactly* the same results, accepts writes while serving, survives crashes, and swaps its AI model
> without a single failed query. And you can build and deploy it yourself.

> ⚙️ **Running the commands:** open a terminal in the repo root (the folder with `src/`) and use the venv's Python.
> Seeing `No module named 'src'`? You're in the wrong folder. See "Before you run ANY command" in the [guide README](README.md).

---

## 7.1 Why split the data at all?

One machine runs out of something eventually: **memory** (vectors for 100M docs = 154 GB), **CPU** (queries per second),
or **safety** (one machine dies → everything is down). The standard answer is to **shard**: split the corpus into
pieces, put each piece in its own process/machine, and put a **router** in front.

```mermaid
flowchart LR
    C["Client<br/>(eval.py, search.py, agents)"] -->|"HTTP POST /search"| R["ROUTER :8000<br/>the only door"]
    R --> S0["SHARD 0 :8101<br/>19,167 docs<br/>BM25 index + vector index"]
    R --> S1["SHARD 1 :8102<br/>19,254 docs<br/>BM25 index + vector index"]
    R --> S2["SHARD 2 :8103<br/>19,217 docs<br/>BM25 index + vector index"]
```

Every shard is the same program holding **both** indexes (BM25 and faiss vectors) for **its own third** of the
docs. A search goes to **all** shards. A write goes only to the **one** shard that owns that doc (§7.2).

At 57K docs we don't *need* this (one process does it in 21 ms). The assignment wants us to build it correctly
anyway, because the hard part is correctness, not size.

## 7.2 Which doc goes to which shard? Hash partitioning

> 🎯 **Why we're doing this:** Every read and write must agree on where a document lives. A stable, table-free rule makes that true across processes and restarts.

`shard = crc32(doc_id) % 3`: a **hash** turns the id into a big number (deterministically), and `% 3` picks 0, 1 or 2.

```
doc_id "12345" → crc32 → 3,421,846,044 → % 3 → 0 → shard 0
doc_id "12346" → crc32 → 1,392,274,342 → % 3 → 1 → shard 1
```
- Every process computes the same answer with no lookup table: the router knows where to send a write.
- It balances well: 19,167 / 19,254 / 19,217 docs (sum 57,638).
- **Pitfall:** Python's built-in `hash("12345")` is **randomised per process** (for security). If the router and a shard each
  used `hash()`, they'd disagree about where documents live. Use `crc32`.
- **Pitfall:** changing 3 → 4 shards with `% N` moves ~75% of docs. *Consistent hashing* moves only ~1/N (toy 15).

## 7.3 A sharded search: "scatter-gather" in two phases

> 🎯 **Why we're doing this:** The assignment requires sharded results within 1% of single node. The two phases (global stats, then merge-then-fuse) are exactly what makes them *identical*. Get them wrong and quality silently drifts.

The router **scatters** the query to all shards in parallel, then **gathers** and merges the answers:

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant R as Router
    participant S0 as Shard 0
    participant S1 as Shard 1
    participant S2 as Shard 2
    C->>R: POST /search {query, k=10, hybrid}
    Note over R: tokenize query → [what, short, sell]<br/>embed query ONCE → q (in a thread, overlapped with phase 1)
    par Phase 1 "DFS": collect statistics
        R->>S0: /stats {terms}
        R->>S1: /stats {terms}
        R->>S2: /stats {terms}
    end
    S0-->>R: N=19,167, total_len, df(what)=5,345...
    S1-->>R: N=19,254, df(what)=5,152...
    S2-->>R: N=19,217, df(what)=5,222...
    Note over R: GLOBAL stats = sum: N=57,638, df(what)=15,719 ...
    par Phase 2: search with global stats
        R->>S0: /search {terms, global stats, vector q, model, depth 100}
        R->>S1: /search {...}
        R->>S2: /search {...}
    end
    S0-->>R: BM25 top-100 + dense top-100
    S1-->>R: BM25 top-100 + dense top-100
    S2-->>R: BM25 top-100 + dense top-100
    Note over R: MERGE: 300 BM25 → global top-100,<br/>300 dense → global top-100,<br/>THEN fuse (linear α=0.7) → top 10
    R-->>C: {results, partial: false, missing_shards: []}
```

(The df(what) split across shards above is illustrative; the global sum is the real value.)

Each shard call has a **0.5 s timeout**. Embedding the query **once at the router**, instead of on every shard, saves 2×8 ms of CPU per query.

### Correctness rule 1: global IDF (the "DFS phase")

BM25's IDF needs `N` and `df`. If each shard used its **own** (local) numbers, the same word gets a different weight on
each shard, so **identical documents get different scores depending on where they were hashed**. That's the classic
**local-IDF bug**:

```
Toy 12: docs a5 (shard A) and b1 (shard B) have IDENTICAL text.
   local stats:  a5 = 1.367   b1 = 2.392    ← 'dividend' is common on A, rare on B
   global stats: a5 = 1.800   b1 = 1.800    ← equal, as they must be
```
Fix: phase 1 asks every shard for its `N`, total length and `df` for the query terms; the router **sums** them
(`CorpusStats.merge`) and sends the global numbers back in phase 2. The name comes from Elasticsearch's "DFS query then fetch".
Cost: one extra round trip (~4 ms). Dense search doesn't need this: a cosine doesn't depend on other documents.

#### Where does the "global N" live, and what happens to it after a write?

> 🎯 **Why this matters:** it's natural to picture a global counter somewhere ("total docs = 57,638") that every write
> must update. There isn't one. That's what makes IDF both correct and fresh without any coordination.

**Nothing global is stored.** Each shard keeps only *its own* counts, updated on every write. The router adds them up
**fresh on every query** in phase 1, then forgets them. So the "global N" is never out of date and never needs updating.

| | Kept where | Contains | Lifetime |
|---|---|---|---|
| local stats | each shard's `BM25Index`: `len(row_of)` (N), `total_len`, `df[term]` | counts for **that shard's live docs only** | permanent, updated by every write |
| global stats | the router, inside one `search()` call | the sum over all shards, for **this query's terms only** | one query, then thrown away |

The formula always uses the global numbers (default `bm25_stats="global"`):
`idf(term) = ln(1 + (N − df + 0.5) / (df + 0.5))` with **N = docs on all shards** and **df = docs on all shards
containing the term**. For example, for the term *what*:

| | shard 0 | shard 1 | shard 2 | global (router sums) |
|---|---|---|---|---|
| N | 19,167 | 19,254 | 19,217 | **57,638** |
| df(what) | 5,345 | 5,152 | 5,222 | **15,719** |

Only the **per-document** parts of BM25 (term frequency `tf`, the doc's length `len(d)`) come from the shard's own data.
(`--bm25-stats local` exists only to reproduce the bug for the experiment above.)

**What a write does to the shard's counts** (`BM25Index.add()` / `delete()` in `src/retrieval/bm25.py`). The counts are
updated *inside* the write, under the write lock, **before** the ack:

| write | shard's N | shard's df | shard's `total_len` |
|---|---|---|---|
| upsert a **new** doc | +1 | +1 for each distinct term in the doc | + its length |
| upsert an **existing** doc (an edit) | unchanged: old version removed, new one added | −1 for the old terms, +1 for the new terms | − old length + new length |
| delete | −1 | −1 for each of its terms (a term whose df reaches 0 is dropped) | − its length |

On delete, the old postings stay in the matrix but are masked out (a *tombstone*, §7.4). The **counts**, however,
forget the doc immediately, so IDF and avgdl always describe live documents only.

**Worked example:** upsert a new doc containing *what*; `crc32` sends it to shard 1.

```
before:      shard 1: N = 19,254   df(what) = 5,152      next query sums → N = 57,638   df(what) = 15,719
write+ack:   shard 1: N = 19,255   df(what) = 5,153      (updated under the write lock, then acked)
next query:  phase 1 sums fresh   → N = 57,639   df(what) = 15,720   ← already includes the write
```

So the freshness guarantee (§7.4: searchable as soon as acked) holds for **IDF too**, not just for the doc itself.

**Caveat (accepted by design):** if a write lands *between* one query's phase 1 and phase 2, that one query scores with
counts that are one write out of date (1 doc in 57K: an invisible change to IDF). The next query is exact again.
Making it perfect would need a snapshot across all shards, which costs far more than it's worth.

### Correctness rule 2: each shard returns its top-D, and the router merges *then* fuses

- **Why top-D per shard is enough:** the global top-100 must be made of docs that are in *their own* shard's top-100.
  (If a doc isn't in its shard's top 100, 100 docs on that shard beat it, so it can't be in the global top 100.)
- **Why fuse after merging:** min-max normalisation makes each *shard's* best doc = 1.0. Fusing per shard would inflate
  every shard's local winner:

```
Toy 13: shard0 has an excellent doc 'a' (BM25 10), shard1's best is mediocre 'c' (BM25 2).
   merge, then fuse:  a 0.750, b 0.688, c 0.556, d 0.000      ← correct
   fuse per shard:    a 1.000, c 1.000, b 0.000, d 0.000      ← 'c' ties 'a'. Wrong!
```

**Result:** sharded quality is **identical** to single node, to 4 decimals, for BM25, dense and hybrid (0.4190 = 0.4190).
The assignment allowed 1% difference; exact equality proves the merge logic is right.

### Timeouts and partial results

> 🎯 **Why we're doing this:** In a distributed system some machine is always slow or dead. Without timeouts one bad shard freezes every query; without the `partial` flag clients can't tell they got incomplete results.

If a shard is slow or dead, the router doesn't hang. After the timeout it returns what it has, **marked**
`partial: true, missing_shards: [2]` (policy `partial`), or an HTTP 503 (policy `fail`). A slightly worse answer usually beats
an error for a search box; for compliance search you'd choose `fail`.

```mermaid
flowchart LR
    R["Router waits ≤ 0.5 s"] --> A{"all 3 shards<br/>answered?"}
    A -- yes --> OK["normal response"]
    A -- "no, policy=partial" --> P["response with<br/>partial=true,<br/>missing_shards=[i]"]
    A -- "no, policy=fail" --> F["HTTP 503"]
```

### What sharding cost in latency (slower at this scale, and that's expected)

> 🎯 **Why we're doing this:** Sharding is not a free speed-up. Measuring it (instead of assuming) is what the
> assignment's "counter-intuitive finding" question is about. The result only *looks* counter-intuitive next to the naive
> belief "more shards = faster". Once you do the arithmetic below, it's exactly what you should expect at 57K docs.

| setup | p50 | p95 |
|---|---|---|
| single process, no HTTP | 20.7 ms | 24.5 ms |
| router + **1** shard (HTTP) | 23.4 ms | 26.3 ms |
| router + **3** shards (HTTP) | 28.6 ms | 40.5 ms |

#### Where the time goes: search vs merge vs network (measured)

The table above only gives end-to-end totals. To split them, `scripts/latency_breakdown.py` times every step **inside
one process (no HTTP at all)**, using the real BM25 / faiss / merge / fuse code and the 500 dev queries:
- one index over all 57,638 docs, versus
- the 3 crc32 shards, each with its own BM25 and faiss index.

Whatever end-to-end time is left over after subtracting that compute is the **network and coordination** cost. All
numbers are p50 in ms, from the same session (`results/latency_breakdown.json` plus fresh end-to-end runs). This session
ran about 10% slower than the table above (e.g. encode 11.2 vs 8.2 ms), so compare the rows **with each other**.

| setup | encode query | search | DFS stats | merge + fuse | network & coordination | other in-process | **end-to-end** |
|---|---|---|---|---|---|---|---|
| single process (no HTTP) | 11.2 | **7.5** (BM25 1.3 + faiss 6.2) | 0 (not needed) | 0.24 (fuse only) | **0** | 3.7 | **22.6** |
| router + 1 shard (HTTP) | 11.2 | 7.5 | 0 (skipped for 1 shard) | 0.24 | **3.8** | 3.7 | **26.4** |
| router + 3 shards (HTTP) | 11.2 | **3.1** (slowest shard: BM25 ≤ 0.7 + faiss ≤ 2.4) | 0.04 | **0.58** | **13.8** | 3.7 | **32.4** |

How each column was obtained:
- **encode, search, DFS stats, merge + fuse:** measured directly by the script. For 3 shards, "search" is the
  **slowest of the 3** shards, because 3 processes run at the same time and the router waits for the last one.
- **other in-process:** the single-process total minus its compute (22.6 − 11.2 − 7.5 − 0.24). It covers the router's
  async/thread hand-offs, fetching the top-10 texts and building the response. It's assumed to be the same for every
  setup.
- **network & coordination:** what's left of the end-to-end time: HTTP + JSON for every call, 2 round trips for
  3 shards (DFS, then search) instead of 1, 3× the result payload (100 BM25 + 100 dense hits per shard), and waiting for
  the slowest shard.
- **Caveats:**
  - The script runs the 3 shard searches one after another and takes the slowest as the "parallel" time. That assumes
    3 real processes don't slow each other down, which is reasonable here (3 × 4 threads on 32 cores).
  - The router overlaps the query encode with the DFS round trip, so the columns are not a perfect sum. Treat the
    network column as ≈.

**Does "split search + merge" beat "search everything"? Yes, measured:**

| | search | + DFS stats | + merge / fuse | **compute total** |
|---|---|---|---|---|
| one index, 57,638 docs | 7.5 | 0 | 0.24 | **7.8 ms** |
| 3 shards **in parallel** (3 processes) | 3.1 | 0.04 | 0.58 | **3.7 ms**: 2.1× faster |
| 3 shards **one after another** (one process) | 8.9 | 0.04 | 0.58 | **9.5 ms**: slower than one index! |

So splitting the search does save about **4 ms**, and the extra merge costs only about 0.3 ms more than a plain fuse.
Two lessons:
- It only works **if the shards really run in parallel**. Searching 3 small indexes one after another is *slower* than one
  big one, because each search call has fixed costs and small faiss searches use threads less efficiently.
- The 3 shards aren't 3× faster (3.1 ms, not 2.5 ms), because the router waits for the **slowest** shard and each
  search has a fixed cost.

**Why the end-to-end result is still slower.** 3 shards vs 1 shard (both over HTTP):
- **saved:** 7.5 − 3.1 ≈ **4.4 ms** of search
- **paid:** ≈ **10.4 ms** extra. That's 13.8 − 3.8 = 10.0 ms of extra network/coordination, plus 0.34 ms of extra merging
  (0.58 vs 0.24) and 0.04 ms for DFS.
- **net:** ≈ +6.0 ms, which matches the measured 32.4 − 26.4 = 6.0 ms.

The query encode (11.2 ms here) happens **once at the router** either way, so sharding can't save any of it.

**Where it would flip (rough estimate).** The network cost is roughly **fixed** per query: each shard always returns
100 + 100 hits, whatever the corpus size. The search saving, though, **grows with the data**, because flat faiss and
BM25 posting lists are O(N). Measured, the 3-shard search takes about 41% of the single-index search (3.1 / 7.5), so it
saves about 59% of W, where W is the single-index search time. Sharding wins once 0.59 × W is larger than the fixed
extra cost:

| compared with | fixed extra cost | break-even search time W (today 7.5 ms) | ≈ corpus size (flat search) |
|---|---|---|---|
| router + 1 shard (HTTP) | ≈ 10.4 ms | ≈ 18 ms (2.3× today) | ≈ **135K docs** |
| single process (no HTTP) | ≈ 14.2 ms (13.8 network + 0.38 merge/DFS) | ≈ 24 ms (3.2× today) | ≈ **185K docs** |

So roughly **2–3× today's corpus**. Below that, sharding is expected to be slower. At 10M docs, a flat index would
spend roughly a second per query on one node, and splitting it would obviously pay. (The estimate assumes search time
grows linearly with the number of docs and the network cost stays fixed. It's a ballpark, not a promise.)

**Two refinements for large scale:**
1. **At 10M+ docs you'd switch to an approximate index (HNSW / IVF, `docs/03_vector_indexes.md` and `docs/15_scaling_to_100M.md`).**
   Its search time grows only slowly with N, so a 10M-doc single node can still answer in a few ms. At that size,
   sharding is mainly about **memory and throughput**: 10M × 384 floats ≈ 15 GB, and 100M ≈ 154 GB doesn't fit on one
   machine. Lower latency is a side effect. BM25 keeps benefiting from the split, because its posting lists grow with N.
2. **The slowest-shard wait never goes away; it gets worse with more shards.** You can already see it in p95: 40.5 vs
   26.3 ms. Big systems fight it with replicas and *hedged requests*: send a backup copy of a slow request to another
   replica and take whichever answers first (DESIGN.md, "Availability").

**Takeaway:** at 57K docs, sharding buys **capacity and fault isolation, not speed**. The speed benefit appears only once
per-shard search work is larger than the fixed coordination overhead.

## 7.4 Live updates: changing documents while serving

> 🎯 **Why we're doing this:** Real corpora change. The assignment requires writes during serving, searchable fast, safe under retries and crashes. Each sub-mechanism below exists to prevent one specific way of losing or corrupting data.

The API: `POST /upsert {doc_id, version, text}` and `POST /delete {doc_id, version}`. The router sends each write to the
**owner** shard (`crc32 % 3`).

### Versions make writes safe to retry (idempotency)

> 🎯 **Why we're doing this:** Networks lose acknowledgements, so clients *will* retry and messages *will* arrive out of order. Versions turn those from data-corruption bugs into harmless no-ops.
Networks lose acknowledgements, so clients retry. Messages arrive out of order. Rule: **apply a write only if its version
is newer than what we have.**

```
upsert(x, v1)          → applied
upsert(x, v1) [retry]  → no-op (same version)       ← retrying is safe = "idempotent"
upsert(x, v3)          → applied
upsert(x, v2) [late]   → no-op (older)              ← old data never overwrites new
delete(x, v4)          → applied, leaves a TOMBSTONE (x: version 4, deleted)
upsert(x, v3) [late]   → no-op                       ← without the tombstone, x would come back to life!
```

### The write path, step by step

> 🎯 **Why we're doing this:** The order of steps *is* the correctness argument: what happens under the lock, what goes to disk first, and when the client is told "done".

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant R as Router
    participant S as Owner shard (crc32 % 3)
    participant W as WAL file (disk)
    C->>R: POST /upsert {doc_id, version, text}
    R->>S: /upsert (retries on timeout: safe, idempotent)
    S->>S: version newer? if not → "noop", stop
    S->>S: embed text with EVERY active model (OUTSIDE the lock, ~8 ms)
    S->>S: take WRITE lock (readers wait briefly)
    S->>S: re-check version (another writer may have won)
    S->>W: append JSON line {op, doc_id, version, text, vectors} + fsync
    S->>S: apply: doc store, versions, BM25 delta, faiss (every model)
    S->>S: release lock
    S-->>R: {applied: true}
    R-->>C: ack, and the doc is ALREADY searchable
```

- **One lock for both indexes:** a reader can never see the new text in BM25 but the old vector in faiss (that would be inconsistent).
- **Freshness guarantee:** the write is applied *before* the ack, so **the first search after the ack finds it**. Measured
  and proven by a test across 3 real processes. (Elasticsearch, by default, makes writes visible only after a ~1 s "refresh".)

### Crash safety: the write-ahead log (WAL)

> 🎯 **Why we're doing this:** RAM is lost on a crash. Without a log, any acknowledged write since start-up would vanish, breaking the promise you made to the client.

Everything above lives in RAM, so a crash would lose it. The fix is the oldest trick in databases: **write the change to
an append-only log file on disk *before* applying it in memory** (then `fsync` = force it onto the physical disk).

```mermaid
flowchart TB
    subgraph normal["Normal write"]
      A1["1. append to WAL + fsync"] --> A2["2. apply in memory"] --> A3["3. ack the client"]
    end
    subgraph crash["After a crash"]
      B1["4. restart: load base corpus<br/>+ embeddings cache"] --> B2["5. REPLAY the WAL<br/>top to bottom<br/>(versions make replay idempotent)"]
      B2 --> B3["6. state = exactly what was acked"]
    end
```
- Crash **before** step 1 → the write never happened, and the client got no ack, so it retries.
- Crash **after** step 1 → the replay re-applies it.
- Crash **during** step 1 (half a line written) → that line was never acked → drop it.
  **A real bug we found and fixed:** at first we only *skipped* the half line during replay, but left it in the file, so the
  next write got glued onto it and was lost on the *next* restart. Now the WAL truncates the torn tail when it opens.
  Lesson: test recovery **twice in a row**.
- The WAL stores the **vectors** too, so replay doesn't need to run the model: fast and deterministic.

## 7.5 Model migration with zero downtime ("blue/green")

> 🎯 **Why we're doing this:** Upgrading the embedding model is routine in real systems, and the naive way (stop, re-embed for hours, restart) means downtime. The assignment requires zero failed queries during the switch, plus a way back.

**Goal:** switch from MiniLM to bge-small while queries and writes keep flowing, and be able to undo it instantly.
**The catch:** a query vector from model A compared with document vectors from model B is **silent garbage** (no error,
just nonsense rankings). So every search must use *one* model end to end.

```mermaid
stateDiagram-v2
    [*] --> Serving_A: normal
    Serving_A --> Building_B: 1. migrate: each shard creates an empty B index<br/>+ turns ON dual-write (new writes → A and B)
    Building_B --> Ready_B: 2. backfill: embed all old docs with B in the background<br/>(throttled; skips docs changed since the snapshot)
    Ready_B --> Serving_B: 3. cutover: router flips active_model = B<br/>(only if EVERY shard reports B ready)
    Serving_B --> Serving_A: 4. rollback: flip back (instant; A is still up to date)
    Serving_B --> Final_B: 5. finalize: drop A (no more rollback)
```

- **Dual-write** keeps both indexes current during the build, so neither goes stale.
- **The backfill race:** the backfill reads doc d3 at version 0 while a user updates d3 to version 1 (dual-written to B).
  The backfill must not overwrite the newer vector. Check: "only write if the doc's version hasn't changed since my snapshot".
- **In-flight queries** read `active_model` once at the start, so a query never mixes models.

Our demo (`scripts/migration_demo.py` → `results/migration_demo.json`): 4 query threads + 1 writer for 77 s across
migrate → backfill (17 s) → cutover → rollback. **2,045 queries, 0 failed**; all 116 docs written during the run were findable with both models.

## 7.6 Deployment: processes or Docker

> 🎯 **Why we're doing this:** "It works on my machine" isn't a deliverable. Docker makes the 4-process system start the same way anywhere, enforces the 2 GB limit, and proves the results don't depend on your laptop.

Two ways to run the 4 processes:

```mermaid
flowchart TB
    subgraph P["Option 1: plain processes (src/service/cluster.py)"]
      P1["python -m src.service.shard_server (×3, ports 8101-8103)"]
      P2["python -m src.service.router (port 8000)"]
    end
    subgraph D["Option 2: docker compose (docker-compose.yml)"]
      I["ONE image fiqa-retrieval:latest<br/>(python 3.11 + CPU torch + models baked in)"]
      I --> DS0["container shard0<br/>mem_limit 2g"]
      I --> DS1["container shard1"]
      I --> DS2["container shard2"]
      I --> DR["container router :8000<br/>starts after shards are healthy"]
      V[("bind mounts:<br/>./data/raw, ./artifacts")] --- DS0 & DS1 & DS2 & DR
    end
```

- Each shard reads `SHARD_ID`, `N_SHARDS`, `SHARD_MODELS` and `STATE_DIR` from its environment, loads *only its own* docs
  (`crc32 % 3 == SHARD_ID`) plus their vectors from the shared embeddings cache, then replays its WAL.
- The router reads `SHARD_URLS`. Health checks (`GET /health`) gate start-up order.
- Measured in Docker: stack up in ~60 s; dev nDCG **0.4190**, identical to native; warm p50/p95 **30.4 / 34.1 ms** (native: 28.6 / 40.5);
  ~505 MiB per shard container (2 GiB cap).

---

## 🧸 A. The toys

| toy | shows |
|---|---|
| `12_sharded_idf_bug.py` | the local-IDF bug (1.367 vs 2.392) and the global-stats fix (1.800 = 1.800) |
| `13_merge_then_fuse.py` | why fusing per shard is wrong (the 'c' ties 'a' example) |
| `14_scatter_gather_timeouts.py` | asyncio fan-out with one slow and one dead shard; partial vs fail policies |
| `15_hash_partitioning.py` | crc32 % 3 balance; 75% movement when growing to 4; consistent hashing; why not `hash()` |
| `16_idempotent_versioned_writes.py` | retries, out-of-order delivery, tombstones on a real `ShardStore` |
| `17_wal_crash_recovery.py` | writes, a simulated crash mid-append, restart + replay (twice) |
| `18_freshness.py` | sync apply (searchable at ack) vs async refresh (searchable ~200 ms later) |
| `19_blue_green_migration.py` | dual-write, backfill race, cutover, rollback |

Run any with `python examples/NN_name.py`. Read the `->` lines (the explanation), then do `TRY THIS`. Toys 16, 17 and 19 use the **real**
`ShardStore` with a fake "hash" embedder, so they run in a second and teach the actual code.

## 🏭 B. The real thing in this repo

| file | role |
|---|---|
| `src/index/sharding.py` | `shard_of(doc_id, n) = crc32 % n` |
| `src/service/store.py` | `ShardStore`: one shard's data: BM25, dense indexes per model, docs, versions, WAL, migration build |
| `src/service/wal.py` | `WAL` (append + fsync, replay, torn-tail truncation), `RWLock` (many readers / one writer) |
| `src/service/shard_server.py` | FastAPI app: `GET /health`, `POST /{op}` for stats / search / docs / upsert / delete / build_model / drop_model |
| `src/service/router.py` | `Router`: the two-phase search, merge, fuse, fetch texts, write routing + retries, migration admin; `create_app()` exposes `/search /upsert /delete /health /admin/{status,migrate,cutover,rollback,finalize}` |
| `src/service/cluster.py` | starts 3 shards + router as local processes (tests and benchmarks use it) |
| `scripts/migration.py`, `query_load.py`, `migration_demo.py` | migration CLI, load generator, scripted demo |
| `tests/test_store.py`, `tests/test_integration_service.py` | sharded = single node, idempotency, freshness, crash + WAL recovery, partial results |

Try it locally (processes):
```powershell
.\.venv\Scripts\python -m src.service.cluster --shards 3        # terminal 1: starts shards + router
.\.venv\Scripts\python src/search.py --query "what is short selling?"   # terminal 2
.\.venv\Scripts\python -m src.eval --mode service --strategy hybrid --fusion linear --alpha 0.7 --name svc-test
```
Or with Docker: `docker compose build`, `docker compose up -d`, `docker compose run --rm eval`.

## ✍️ C. The toy, by you

Build a **2-shard system inside one Python process** (`exercises/ch07_shards/mini_shards.py`, no HTTP yet; check: `python -m pytest exercises\ch07_shards`):
1. 10 docs; assign each with `zlib.crc32(id.encode()) % 2`; give each shard its own BM25 (your Ch. 3 code).
2. Write `router_search(query)`: (a) collect each shard's N / total_len / df, sum them; (b) each shard scores with the global
   stats and returns its top-5; (c) merge by score → top-5.
3. **Check:** the results equal one BM25 over all 10 docs. Then switch to local stats and find a query where they differ (toy 12's trick:
   put identical texts on both shards).
4. Add `upsert(id, version, text)` / `delete(id, version)` with version checks and tombstones; replay toy 16's sequence and get the same
   applied/no-op pattern.
5. Add a WAL: append a JSON line + `os.fsync` before applying. Simulate a crash by writing half a line, then "restart" (rebuild from the file)
   **twice**, with a write in between.

## 🔧 D. The real thing, by you (and deploy it)

1. **Shard server** (FastAPI): at start-up, load the corpus, keep only `crc32(id) % N == SHARD_ID`, build BM25 + faiss from the embedding
   cache rows for those ids, replay the WAL. Endpoints: `POST /stats`, `POST /search`, `POST /docs`, `POST /upsert`, `POST /delete`, `GET /health`.
2. **Router** (FastAPI + `httpx.AsyncClient`, **one persistent client**: creating one per request costs ~600 ms on Windows!):
   `asyncio.gather` with `asyncio.wait_for(..., 0.5)` for each phase; global stats; merge-then-fuse; the `partial`/`missing_shards` fields.
3. **Prove correctness:** run your Chapter 6 evaluation through the router. **Target:** identical nDCG to single node (0.419). If it's off,
   suspect: local IDF, fuse-before-merge, too-small per-shard depth, or a different tokenizer in router vs shard.
4. **Measure** 1 shard vs 3 shards p50/p95. Do you also see 3 shards being slower?
5. **Writes:** versions + tombstones + RW lock + WAL. Write the integration tests: idempotent retry, out-of-order, delete visible in BM25 *and*
   dense, freshness (search right after the ack, no sleep), and kill -9 a shard → restart → acked writes still there.
6. **Migration:** keep `{model: index}` per shard; `/build_model` (create + dual-write on, then a background backfill thread, throttled); router
   `active_model` persisted to a file; cutover only when all shards report ready. Run a load generator during migrate → cutover → rollback and
   count failed queries (**target: 0**).
7. **Deploy:** write a Dockerfile (python:3.11-slim, CPU-only torch from `https://download.pytorch.org/whl/cpu`, bake the models into the image),
   and a `docker-compose.yml` with 3 shard services (`mem_limit: 2g`, healthchecks) + a router (`depends_on: service_healthy`) + bind-mounted
   `data/` and `artifacts/`. `docker compose up -d`, then run your eval from a container. **Target:** the same nDCG as native.
   (Corporate network? Pass `--build-arg PIP_INDEX_URL=...` for your PyPI mirror.)

### Check yourself
1. Two identical documents end up on different shards and get different BM25 scores. What's the bug and the fix?
2. Why must the router fuse *after* merging shard lists?
3. Why is "retry on timeout" safe for our writes?
4. What can go wrong if you apply a write in memory *before* appending it to the WAL? (Toy 17, TRY THIS 3)
5. Why must one search never mix query vectors from model A with document vectors from model B?
6. Why were 3 shards slower than 1 here? When would sharding make queries faster?

**Next:** [Chapter 8: Part C: LLMs and agents](08_part_c_agents.md)
