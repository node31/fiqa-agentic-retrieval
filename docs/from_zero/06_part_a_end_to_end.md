# Chapter 6: Part A end to end: indexing, querying, evaluating, explaining failures

> Goal: you can draw the whole Part A pipeline from memory, run it, read every number in `results/bench.json`,
> and explain *why* the system fails on the queries it fails on.

> ⚙️ **Running the commands:** open a terminal in the repo root (the folder with `src/`) and use the venv's Python.
> Seeing `No module named 'src'`? You're in the wrong folder. See "Before you run ANY command" in the [guide README](README.md).

Chapters 1–5 gave you the pieces. This chapter assembles them into what Part A asks for: **a single-machine search
system, benchmarked honestly, with its failures explained.**

---

## 6.1 The whole pipeline in one picture

> 🎯 **Why we're doing this:** Part A is graded as one system, not as parts. Seeing it end to end shows what is computed once (offline) vs per query (online), which is the key to both speed and the Part B design.

```mermaid
flowchart TB
    subgraph OFF["OFFLINE (once)"]
      D1["1. download FiQA<br/>scripts/download_data.py"] --> D2["2. load corpus<br/>57,638 passages"]
      D2 --> D3["3a. build BM25 index<br/>(~9 s, in memory)"]
      D2 --> D4["3b. encode all passages<br/>MiniLM, ~10 min CPU<br/>→ artifacts/emb_minilm.npy"]
    end
    subgraph ON["ONLINE (per query, ~21 ms)"]
      Q["4. query text"] --> T["5a. tokenize<br/>→ BM25 top 100"]
      Q --> E["5b. encode (8 ms)<br/>→ faiss top 100"]
      T --> F["6. linear fusion<br/>α = 0.7"]
      E --> F
      F --> K["7. top 10 doc ids"]
    end
    subgraph EV["EVALUATION"]
      K --> M["8. compare with qrels<br/>nDCG / Recall / MRR"]
      M --> R["9. results/runs/NAME.json<br/>results/bench.json"]
    end
    D3 -.loaded.-> T
    D4 -.loaded.-> E
```

## 6.2 One query, followed step by step

> 🎯 **Why we're doing this:** When a number looks wrong, you debug by following one query through each step. This diagram is that checklist.

```mermaid
sequenceDiagram
    participant U as You (eval.py)
    participant RT as Router (single node)
    participant TK as tokenizer
    participant EN as MiniLM encoder
    participant BM as BM25 index
    participant FA as faiss index
    participant HY as fusion
    U->>RT: 1. search("What is short selling?", k=10, hybrid, linear, α=0.7)
    RT->>TK: 2. tokenize → [what, short, sell]
    RT->>EN: 3. encode → q (384 numbers)  ~8 ms
    RT->>BM: 4. search(terms, 100)  ~1 ms
    BM-->>RT: 100 × (doc, BM25 score)
    RT->>FA: 5. search(q, 100)  ~5 ms
    FA-->>RT: 100 × (doc, cosine)
    RT->>HY: 6. fuse: min-max both, 0.7·dense + 0.3·bm25
    HY-->>RT: merged list (up to 200 docs)
    RT-->>U: 7. top 10 ids + scores (~21 ms total)
```

Fun detail: in this repo, even "single node" goes through the **same Router code** as the 3-shard service, just with one
in-process shard (`single_node_router()`). Part B then only changes *where* the shards live, not the logic. That's also
how we prove sharded = single-node (Chapter 7).

## 6.3 The benchmark: what we measure and how

> 🎯 **Why we're doing this:** The assignment asks for an honest comparison of configurations. Fresh processes, warm-up on train and percentiles are what make the numbers trustworthy enough to base decisions on.

Part A wants, for each configuration: **quality** (nDCG@10, Recall@10, MRR@10, overall + stratified), **latency**
(p50/p95, cold and warm), and **peak memory**. `scripts/bench.py` runs each configuration **in a fresh process**
(so "cold" is real) and collects everything into `results/bench.json`.

```mermaid
flowchart LR
    B["scripts/bench.py"] --> C1["config 1: bm25"]
    B --> C2["config 2: dense minilm"]
    B --> C3["config 3: dense bge"]
    B --> C4["config 4: hybrid rrf"]
    B --> C5["config 5: hybrid linear ✓"]
    B --> C6["... + service configs<br/>(Part B)"]
    C1 & C2 & C3 & C4 & C5 & C6 --> P["each: NEW python process<br/>python -m src.eval ...<br/>→ results/runs/NAME.json"]
    P --> J["merged into<br/>results/bench.json"]
```

### Run the benchmark yourself, on the tiny dataset (one config at a time)

> 📖 Every argument, output field and protocol decision of `src/eval.py`, in tables: [`docs/eval.md`](../eval.md).

> 🎯 **Why we're doing this:** `results/bench.json` is just the output of a program you can run. Running each config yourself
> on 24 documents (seconds, not 30 minutes) shows you exactly what produces each number, and with only 4 queries you can
> read every line of every results file.

**Three environment variables point everything at the toy data and a scratch folder**, so you can't overwrite the real
results. `toy_runs\` is gitignored. Run from the repo root, venv active:

```powershell
$env:FIQA_DATA_DIR = "$PWD\tests\fixtures\tinyfiqa"   # which dataset: the 24-doc tiny FiQA (default: data\raw\fiqa)
$env:ARTIFACTS_DIR = "$PWD\toy_runs\artifacts"        # where indexes/cluster state go (empty → embeds the 24 docs on the fly)
$env:RESULTS_DIR   = "$PWD\toy_runs\results"          # where results go (default: results\, the REAL numbers)
```

**Step 1: one config = one command.** Each line below is exactly what `bench.py` runs for that config (its `LOCAL` list):

```powershell
python -m src.eval --name bm25                    --strategy bm25
python -m src.eval --name dense-minilm            --strategy dense  --model minilm
python -m src.eval --name dense-bge-small         --strategy dense  --model bge-small
python -m src.eval --name hybrid-rrf-minilm       --strategy hybrid --model minilm    --fusion rrf    --alpha 0.5
python -m src.eval --name hybrid-linear-minilm    --strategy hybrid --model minilm    --fusion linear --alpha 0.7
python -m src.eval --name hybrid-linear-bge-small --strategy hybrid --model bge-small --fusion linear --alpha 0.7
```

Each prints one summary line and writes `toy_runs\results\runs\<name>.json`. What we got:

```
[eval] no train split in this dataset: skipping warmup (warm latency is less meaningful)
bm25                    nDCG@10=0.9343 R@10=1.0000 MRR@10=1.0000 warm p50/p95=0.25/0.33ms  cold p50/p95=0.31/0.87ms
dense-minilm            nDCG@10=0.9626 R@10=1.0000 MRR@10=1.0000 warm p50/p95=26.98/29.9ms cold p50/p95=26.99/33.09ms
dense-bge-small         nDCG@10=0.9492 ...                        warm p50/p95=40.83/42.33ms
hybrid-rrf-minilm       nDCG@10=0.9626 ...
hybrid-linear-minilm    nDCG@10=0.9579 R@10=1.0000 MRR@10=1.0000 warm p50/p95=27.0/29.22ms  cold p50/p95=27.11/28.02ms
hybrid-linear-bge-small nDCG@10=0.9492 ...
```

(BM25 0.934, dense 0.963, hybrid 0.958: the same numbers as your ch06 exercise, because it's the same pipeline at toy scale.)

**Step 2 (optional): the whole matrix.** `python scripts\bench.py` runs the 6 commands above, **plus** the 5 service
configs (it starts a real router and 1 or 3 shard processes for each, then evaluates over HTTP), and merges everything
into `toy_runs\results\bench.json`. On the tiny set it took ~7 minutes, mostly model loading and starting processes.
`python scripts\bench.py --only local` runs just step 1's six.

**What happens inside one `python -m src.eval` run:**

```mermaid
sequenceDiagram
    participant You
    participant E as src/eval.py
    participant R as single-node router (in-process)
    participant F as results file
    You->>E: 1. python -m src.eval --name X --strategy ...
    E->>R: 2. build indexes (BM25 + dense, model loaded) → startup_s
    E->>R: 3. COLD: first 20 dev queries, timed (tiny: only 4 exist)
    E->>R: 4. WARMUP: 100 train queries, NOT timed (tiny has no train split → skipped)
    E->>R: 5. WARM: every dev query, timed, ranked ids kept
    E->>E: 6. metrics (nDCG/Recall/MRR), stratified recall, p50/p95, memory
    E->>F: 7. write runs/X.json, print the one-line summary
```

**How to read a run file** (`toy_runs\results\runs\hybrid-linear-minilm.json`, shortened):

| key | example (tiny) | meaning |
|---|---|---|
| `split` | `"dev"` | which queries were evaluated |
| `overall` | `{"ndcg@10": 0.9579, "recall@10": 1.0, "mrr@10": 1.0, "n": 4}` | the averages over `n` queries (Ch. 2) |
| `stratified` | `"qlen:medium(5-15)": {"recall@10": 1.0, "n": 4}` | Recall@10 per group: by query length, and by whether the gold doc is among the 10% longest docs. **Always read `n`**: a group of 1 query proves nothing |
| `latency.cold_first20` | `{"p50_ms": 27.11, "p95_ms": 28.02, "n": 4}` | the first queries of a fresh process (model and caches still cold) |
| `latency.warm` | `{"p50_ms": 27.0, "p95_ms": 29.22, "n": 4}` | all queries after warmup: the number the 50 ms limit is about |
| `memory` | `{"process": {"rss_mb": 551.1, "peak_rss_mb": 551.5}}` | RAM of the searching process (service runs list each shard instead) |
| `run` | `{"q1": ["1","2","9","22",...], ...}` | **the actual ranked top-10 per query**: everything above is computed from this |
| `config` | `{"strategy": "hybrid", "fusion": "linear", "alpha": 0.7, ...}` | the exact settings, so the run can be reproduced |
| `startup_s` | `13.9` | seconds to build the indexes and load the model before the first query |

`bench.json` has three keys: `machine` (CPU, cores, RAM, OS: latency depends on hardware), `protocol` (how cold/warm
were measured), and `configs` (one entry per config with all of the above *except* `run`, to keep it small).

**Try this, and read the effect in the files:**
- Run α = 0.3 under a **new name** (the same name would overwrite the old file):
  `python -m src.eval --name hybrid-linear-a03 --strategy hybrid --model minilm --fusion linear --alpha 0.3`. Then compare
  `run.q3` in `hybrid-linear-minilm.json` vs `hybrid-linear-a03.json`. We got `7, 4, 14, 21, …` at α = 0.7 and `7, 14, 20, 4, …`
  at α = 0.3. With more weight on BM25, its keyword hits (doc 14 matches "should", doc 20 matches "fund") climb above dense's
  topical doc 4. nDCG doesn't change (0.9579 both times) because the only gold doc, 7, stays #1.
- Add `--limit 2`: only q1 and q2 are evaluated, and `overall.n` becomes 2.
- Compare `run` in `svc3-bm25.json` (global IDF) with `svc3-bm25-localidf.json` (from step 2). The rankings **differ in 3 of 4 queries**,
  e.g. q3 is `7, 14, 20, 5, 4` with global stats and `7, 20, 5, 14, 4` with local stats. **Yet nDCG is identical** (0.9343),
  because the gold doc stays #1. A bug can hide behind good metrics on a small test set. On the full corpus, local IDF changes
  395 of 500 top-10s (ch07).
- Rerun the same command twice: the quality numbers are identical (deterministic), the latencies wobble (that's why we use percentiles).

**When you're done:** `Remove-Item Env:FIQA_DATA_DIR, Env:ARTIFACTS_DIR, Env:RESULTS_DIR` (or close the terminal). Otherwise
your next "real" run would quietly use the tiny data.

### Our numbers (dev, single node)

| configuration | nDCG@10 | R@10 | MRR@10 | warm p50 / p95 ms | peak RSS |
|---|---|---|---|---|---|
| BM25 | 0.264 | 0.314 | 0.322 | 1.2 / 2.0 | 511 MB |
| dense MiniLM | 0.388 | 0.467 | 0.453 | 18.8 / 20.8 | 806 MB |
| dense bge-small | 0.398 | 0.468 | 0.477 | 30.6 / 34.2 | 851 MB |
| hybrid RRF (plain) | 0.373 | 0.442 | 0.449 | 22.1 / 39.9 | 812 MB |
| **hybrid linear α=0.7, MiniLM (chosen)** | **0.419** | **0.491** | **0.487** | 20.7 / 24.5 | 812 MB |
| hybrid linear α=0.7, bge-small | 0.416 | 0.486 | 0.495 | 35.3 / 40.3 | 854 MB |
| **TEST split, chosen config (run once)** | **0.397** | 0.466 | 0.479 | 21.5 / 24.9 | 813 MB |

**How to read it like a reviewer:**
- **Why MiniLM and not BGE?** BGE wins as *dense alone* (+0.010), but in *hybrid* it doesn't (0.416 vs 0.419) and costs
  +15 ms. BM25 already supplies what BGE adds. Measure the combination, not the parts!
- **Where does the 21 ms go?** Measured with `scripts/microbench.py`: query encode 8.2 ms, faiss over 57K vectors 5.3 ms,
  BM25 1.0 ms, and the rest is Python glue (building lists, fusion, id mapping).
- **Cold ≈ warm** here because the code deliberately loads the model and runs a warm-up encode at start-up. What that hides
  (measured): loading the model takes 1.5 s, and the very first encode takes 19 ms instead of 8 ms.
- **Why is memory ~800 MB?** Mostly PyTorch + the model (~650 MB). The data itself is small (88 MB of vectors, ~20 MB of BM25).

### Stratified recall: *who* does the system fail?

> 🎯 **Why we're doing this:** An average hides which users suffer. Strata point to concrete fixes (chunk long docs, help short queries) that the overall number never would.

| group | Recall@10 | n |
|---|---|---|
| short queries (< 5 words) | 0.385 | 13 |
| medium (5–15 words) | 0.491 | 403 |
| long (> 15 words) | 0.510 | 84 |
| gold answer is among the 10% longest passages | 0.464 | 185 |
| gold answer is not | 0.508 | 315 |

Short queries give both systems little to work with; long gold passages get truncated (dense) and length-penalised (BM25).
Beware small groups (n = 13 is noisy).

## 6.4 Failure analysis: learning from what goes wrong

> 🎯 **Why we're doing this:** Metrics tell you *how much* the system fails; failure analysis tells you *why* and *what to do next*. It's also an explicit assignment deliverable (3 cases, different modes, concrete fixes).

161 of the 500 dev queries have **Recall@10 = 0** (no correct doc in the top 10). `scripts/failure_cases.py` dumps
them with the top 5, plus each gold doc's rank in BM25, dense and fused. We then picked 3 with **different causes**
(full write-up: `results/failures.md`):

```mermaid
flowchart TB
    F["161 failed queries"] --> A["Case 1 (qid 4746): 'Personal finance in EFU and NAFA'<br/>BM25 ranks gold #1 (rare words!), dense can't<br/>(MiniLM splits EFU → e ##fu, meaningless).<br/>Fusion α = 0.7 caps a BM25-only doc at 0.3 → rank 19"]
    F --> B["Case 2 (qid 4029): 'Family suggests my first real estate. Advice?'<br/>gold passages are 542 and 1,871 tokens long;<br/>MiniLM reads only the first 256;<br/>the relevant part is at the END"]
    F --> C["Case 3 (qid 7490): 'Clarify on some Stocks Terminology'<br/>the real question (about 'volume') was in the post body,<br/>which the dataset dropped; query and gold share only 'stock'"]
    A --> FA["fix tried: lower α when the query has a very rare word<br/>→ rescues this query, but overall +0.0016 (noise) → not adopted"]
    B --> FB["fix tried: split long passages into chunks, score the best chunk<br/>→ cosine 0.464 → 0.548, still below the top-10 cut-off 0.583"]
    C --> FC["no retrieval fix possible: the information isn't in the query<br/>→ an agent could ask a clarifying question"]
```

And how common each failure type is (one query can be in several):

| pattern | count of 161 |
|---|---|
| **neither** retriever has any gold doc in its top 100 (first-stage miss) | 58 |
| near miss: a gold doc at fused rank 11–20 | 29 |
| all gold docs are very long (> 270 words) | 18 |
| BM25 has gold in its top 10 but dense doesn't have it in its top 100 | 10 |

**The big lesson:** a third of the failures are misses where *no later stage can help*, because the right doc never
entered the candidate pool. Fancier reranking (or an LLM) can't fix those. Better recall at stage 1 must come first.

## 6.5 The chosen operating point, checked against the constraints

> 🎯 **Why we're doing this:** A "best" system that breaks a constraint isn't a valid answer. This table is the final yes/no check before Part B builds on this configuration.

| constraint | requirement | ours | ✓ |
|---|---|---|---|
| CPU only | no GPU | yes | ✓ |
| latency | p95 ≤ 50 ms after warm-up | 24.5 ms single node; 40.5 ms through 3 shards | ✓ |
| memory | ≤ 2 GB per shard | ~723 MB per shard | ✓ |
| quality | best that fits the constraints | 0.419 dev / 0.397 test | ✓ |

---

## 🧸 A. The toy: a 20-query run of the real pipeline

The "toy" here is the real pipeline on a few queries, so you can watch it work in seconds:
```powershell
.\.venv\Scripts\python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.7 --limit 20 --name toy20
```
Then open `results/runs/toy20.json` and find: `overall` (metrics), `stratified`, `latency.cold_first20` / `latency.warm`,
`memory`, and the per-query rankings. With only 20 queries the metrics jump around a lot. Try `--limit 20` vs `--limit 100`
and notice how much. That is *why* we use all 500.

Also play with single queries interactively:
```powershell
.\.venv\Scripts\python -c "import asyncio; from src.service.router import single_node_router as s; r=s(); print(asyncio.run(r.search('should I pay off my mortgage early', k=5, include_text=True)))"
```

## 🏭 B. The real thing in this repo

| step | file / command |
|---|---|
| download | `python scripts/download_data.py` |
| embedding cache | `python -m src.index.build_index --models minilm bge-small` |
| one evaluation | `python -m src.eval --mode local --strategy hybrid --fusion linear --alpha 0.7 --name X` |
| the full matrix | `python scripts/bench.py` → `results/bench.json` (single-node + service configs) |
| ablations | `python scripts/ablations.py` (α sweep), `--rerank` (cross-encoder sweep) |
| failures | `python scripts/failure_cases.py` → `results/failure_candidates.json`; `scripts/failure_probe.py` tests the fixes |
| test (once!) | `python -m src.eval ... --split test --final-test-run` |

(The `Makefile` has short names for these: `make bench`, `make failures`, ... On Windows without `make`, run the commands above.)

## ✍️ C. The toy, by you

Fill in `exercises/ch06_pipeline/mini_pipeline.py` (check: `python -m pytest exercises\ch06_pipeline`) (~80 lines, your own Chapter 2–5 code, no repo imports):
1. Take the 24-doc `tests/fixtures/tinyfiqa` corpus and its queries/qrels.
2. Build your BM25 and dense (MiniLM) indexes over the 24 docs; implement linear fusion.
3. Evaluate BM25 vs dense vs hybrid with your metrics. With 24 docs you can **read every ranking** and say why each doc
   is where it is. Do it for at least 2 queries.

## 🔧 D. The real thing, by you

1. Glue your Chapter 3–5 pieces into one `search(query, strategy, k)` function over the full corpus.
2. Write your own harness (Chapter 2 §D): cold 20 → warm-up 100 train → 500 dev, with metrics, p50/p95 and peak RSS, and save JSON.
3. Produce **your own** benchmark table for BM25 / dense / hybrid. **Targets:** 0.264 / 0.388 / 0.419 nDCG@10.
   If you're off by more than ~0.005, diff your rankings against ours for the same query: `results/runs/*.json` has our
   per-query rankings.
4. Add stratified recall (query-length buckets; "gold in the 10% longest docs").
5. Failure analysis: list the queries with Recall@10 = 0; for each, find the gold docs' ranks in BM25 and dense separately
   (search to depth 1000). Write up 3 *different* failure modes, each with a concrete fix, and **test** at least one fix.
6. Freeze everything, then run test once.

### Check yourself
1. Why does each benchmark configuration run in a fresh process?
2. BGE beats MiniLM alone but not in hybrid. Explain.
3. 58 failures have no gold in either top-100. Why can't a reranker help them? What could?
4. Why is warm-up done with *train* queries and not dev queries?

**Next:** [Chapter 7: Part B: going distributed](07_part_b_distributed.md)
