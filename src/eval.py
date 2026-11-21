"""Evaluation harness: quality, stratified quality, cold/warm latency and peak RAM.

Easier to read, with tables and diagrams: docs/walkthroughs/eval.md

WHAT
    Runs one retrieval CONFIGURATION over a query split and reports:
      overall     nDCG@10, Recall@10, MRR@10
      stratified  Recall@10 by query length (short <5 / medium 5-15 / long >15 words)
                  Recall@10 for queries whose gold passage is among the 10% longest docs vs the rest
      latency     cold p50/p95 (first 20 queries of a FRESH process) and
                  warm p50/p95 (all dev queries, after a 100-query warmup on TRAIN queries)
      memory      peak RSS of the process doing the search (local) or max over shards (service)

    python -m src.eval --mode local   --strategy hybrid --model minilm --name hybrid-rrf
    python -m src.eval --mode service --strategy hybrid --url http://127.0.0.1:8000
    python -m src.eval --help          # prints every argument with a short explanation

ARGUMENTS (each one picks WHICH system to test, or HOW to run the test)
  Where the search runs
    --mode        local (default): the whole engine (BM25 + dense index + router) is built INSIDE this process,
                  no network. service: the engine already runs as router + shard processes (Part B); every
                  query is sent over HTTP, so latency includes the network hop.
    --url         service mode only: the router's address (default http://127.0.0.1:8000). Ignored in local mode.
  What system to test (the "configuration")
    --strategy    bm25 = keyword search only (ch03) · dense = meaning search only (ch04) · hybrid = both, fused (ch05).
    --model       dense side's embedding model: minilm (default) or bge-small. Local mode only: in service mode
                  the running router's active model decides and this is ignored.
    --fusion      hybrid only: rrf = combine by RANK (default) · linear = min-max the SCORES, then weighted sum.
    --alpha       hybrid only: the weight on DENSE; BM25 gets 1 - alpha. linear: a*dense + (1-a)*bm25.
                  rrf: list weights (1-a, a), so alpha 0.5 = plain RRF.
                  PITFALL: the defaults (rrf, 0.5) are NOT the shipped system. Pass --fusion linear --alpha 0.7.
    --rerank-k    0 = off (default). N = re-score the fused top N with the cross-encoder (ch05; slower, didn't help).
    --bm25-stats  global (default) = router sums IDF statistics from all shards first (correct).
                  local = each shard uses only its own statistics (the ch07 bug, kept for comparison).
                  Matters only with several shards: in local mode (one in-process shard) both are identical.
                  In service mode it's ignored here; set BM25_STATS=local on the ROUTER instead (bench.py does).
  Which questions
    --split       dev (default, 500 questions, used for all tuning) or test (648, the "final exam").
    --final-test-run  safety lock: --split test refuses to run without it, so you can't tune on test by accident.
    --limit N     evaluate only the first N questions (quick checks; overall.n then equals N).
  Bookkeeping
    --name        required label: printed on the summary line and used as the file name runs/<name>.json.
                  Reusing a name OVERWRITES that file.
    --threads     torch CPU threads for query encoding (default 4, fixed so latencies are comparable).
                  Local mode only; in service mode the model runs inside the shards.
    --out         write the results JSON here instead of RESULTS_DIR/runs/<name>.json.
  Environment (not arguments): FIQA_DATA_DIR = which dataset, ARTIFACTS_DIR = index/cache folder,
    RESULTS_DIR = where results go. Point all three at a scratch folder to run on the tiny toy dataset
    without touching the real results (docs/walkthroughs/eval.md).

  Examples
    python -m src.eval --name hybrid-linear-minilm --strategy hybrid --model minilm --fusion linear --alpha 0.7
    python -m src.eval --mode service --name svc3-hybrid --strategy hybrid --fusion linear --alpha 0.7
    python -m src.eval --name bm25-quick --strategy bm25 --limit 20

PROTOCOL DECISIONS (docs/topics/07_benchmarking_latency_memory.md)
    * Warmup uses TRAIN queries, so no dev query is ever "seen" before it is timed.
    * Each configuration runs in its own fresh process (scripts/bench.py does this),
      so "cold" really is cold: first inference, cold CPU caches, new connections.
    * Latency is measured by the CLIENT, end to end: for service mode that includes
      HTTP to the router, so it is the number the 50 ms constraint refers to.
    * The test split is locked behind --final-test-run. Tune on dev only.

PITFALLS
    * Percentiles from 20 samples are noisy: p95 of 20 is the 19th value.
    * Don't average p95s across runs. Recompute from raw samples.

HOW TO READ THE CODE (top to bottom)
    length_bucket / long_doc_ids / stratified   quality broken down by query/doc length
    latency_summary                             raw timings -> p50 / p95
    Searcher                                    one search() API for local OR service mode
    evaluate                                    the protocol: cold -> warmup -> warm run -> metrics
    main                                        parse flags, build Searcher, run, save JSON
    Data shapes used throughout:
        queries = {query_id: "question text"}
        qrels   = {query_id: {doc_id: grade}}   the answer key ("gold" docs = grade > 0)
        run     = {query_id: [doc_id, ...]}     our top-10, best first
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from src.beir_data import load_corpus, load_split
from src.config import PRIMARY_MODEL, RESULTS_DIR, SEED
from src.metrics import evaluate_run, recall_at_k
from src.util import percentile, rss_mb


def length_bucket(query: str) -> str:
    """Which query-length group a question falls in, counted in whitespace-separated words.

    e.g. "what is short selling?" = 4 words -> "short(<5)"; 5..15 words -> "medium";
    16+ -> "long". WHY: the assignment asks whether quality depends on query length
    (a long, chatty question and a 3-word one can behave very differently).
    """
    n = len(query.split())
    # Chained conditional expression: read it as if n < 5: short / elif n <= 15: medium / else: long.
    return "short(<5)" if n < 5 else "medium(5-15)" if n <= 15 else "long(>15)"


def long_doc_ids(corpus: dict[str, str], top_frac: float = 0.10) -> set[str]:
    """Ids of the longest ``top_frac`` of passages (by word count): the "10% longest docs".

    WHY: long passages are partly cut off by the embedding model (max_seq_length
    256 tokens, see config.py), so dense search may miss gold docs that are long.
    """
    # (length, doc_id) pairs sorted longest first; ties broken by doc_id (descending).
    lens = sorted(((len(t.split()), d) for d, t in corpus.items()), reverse=True)
    # Keep the first 10%: for 57,638 passages that is int(5763.8) = 5,763 doc ids.
    return {d for _, d in lens[: int(len(lens) * top_frac)]}


def stratified(run: dict, qrels: dict, queries: dict, long_docs: set[str]) -> dict:
    """Recall@10 per stratum, with the number of queries in each (always report n!).

    A "stratum" is a subgroup of queries (e.g. "short queries"). Averaging
    recall inside each group shows WHERE the system is weak, which one overall
    number hides. Output, e.g.:
        {"doclen:gold_in_top10pct_longest": {"recall@10": 0.41, "n": 120},
         "doclen:gold_not_longest":         {"recall@10": 0.52, "n": 380},
         "qlen:long(>15)": {...}, "qlen:medium(5-15)": {...}, "qlen:short(<5)": {...}}
    (illustrative numbers; keys come out sorted alphabetically).
    """
    # group name -> list of per-query Recall@10 values (averaged at the end).
    groups: dict[str, list[float]] = {}
    for qid, ranked in run.items():
        r = recall_at_k(ranked, qrels[qid], 10)
        # Each query lands in TWO groups (one per dimension), so the group sizes
        # of each dimension add up to the total number of queries.
        groups.setdefault("qlen:" + length_bucket(queries[qid]), []).append(r)
        # A query is in the "long gold" group if ANY of its gold docs is in the top-10% longest.
        # set(qrels[qid]) = the query's judged doc ids; `& long_docs` = those that are long.
        key = "gold_in_top10pct_longest" if set(qrels[qid]) & long_docs else "gold_not_longest"
        groups.setdefault("doclen:" + key, []).append(r)
    # Mean recall per group, rounded to 4 decimals, plus n (a mean over 12 queries is
    # far less trustworthy than one over 400).
    return {g: {"recall@10": round(sum(v) / len(v), 4), "n": len(v)} for g, v in sorted(groups.items())}


def latency_summary(ms: list[float]) -> dict:
    # p50 = median ("typical" query), p95 = at most 5% of queries are slower (the "tail").
    # Nearest-rank percentiles (src/util.py): always an actual measured value.
    return {"p50_ms": round(percentile(ms, 50), 2), "p95_ms": round(percentile(ms, 95), 2), "n": len(ms)}


class Searcher:
    """Uniform ``search(query) -> ranked doc ids`` over the in-process router or the HTTP service.

    WHY: evaluate() must not care WHERE the engine runs. Both modes go through the
    same Router code (local = one in-process shard, service = router + shard
    processes over HTTP), so a quality or latency difference between the modes
    comes from sharding + network, not from two different implementations.
    ``params`` = the search settings forwarded on every call
    (strategy, fusion, alpha, rerank_k).
    """

    def __init__(self, mode: str, url: str | None = None, **params):
        self.mode, self.params = mode, params
        if mode == "local":
            # Imported lazily: building the router loads the corpus and models, and
            # service mode never needs any of that in this process.
            from src.service.router import single_node_router

            # Router.search is `async` (it fans out to shards concurrently). This
            # script is synchronous, so we keep one private event loop and run
            # each search to completion on it (see search() below).
            self.loop = asyncio.new_event_loop()
            # "model" and "bm25_stats" configure how the router is BUILT, not each
            # search, so pop() them out of params before params is forwarded per query.
            model = params.pop("model", PRIMARY_MODEL)
            # BM25-only runs load no embedding model at all (faster start-up, less RAM).
            models = () if params.get("strategy") == "bm25" else (model,)
            self.router = single_node_router(models=models, active_model=model,
                                             bm25_stats=params.pop("bm25_stats", "global"))
        else:
            import httpx

            # The running service already decided its model and BM25 statistics
            # (see the ARGUMENTS notes above), so these flags are dropped, not sent.
            params.pop("model", None)
            params.pop("bm25_stats", None)
            # WHY one Client: it reuses the TCP connection between queries (keep-alive),
            # so we don't time a fresh connection handshake on every query.
            self.http = httpx.Client(base_url=url, timeout=30)  # persistent keep-alive client

    def search(self, query: str, k: int = 10) -> list[str]:
        # Both modes produce the same response dict: {"results": [{"doc_id", "score"}, ...], "partial", ...}.
        if self.mode == "local":
            out = self.loop.run_until_complete(self.router.search(query, k=k, **self.params))
        else:
            r = self.http.post("/search", json={"query": query, "k": k, **self.params})
            r.raise_for_status()
            out = r.json()
            # WHY fail loudly: "partial" means a shard didn't answer, so part of the
            # corpus was never searched. Scoring that would silently lower the metrics.
            if out.get("partial"):
                raise RuntimeError(f"partial result during eval: {out['missing_shards']}")
        # Metrics only need the ordered doc ids, not the scores.
        return [x["doc_id"] for x in out["results"]]

    def memory(self) -> dict:
        # Local: everything (index + models) lives in THIS process, so measure it.
        if self.mode == "local":
            return {"process": rss_mb()}
        # Service: the memory lives in the shard processes; ask the router, which
        # collects each shard's rss_mb() via /admin/status. The headline number is
        # the BIGGEST shard's peak, i.e. how much RAM one machine would need.
        st = self.http.get("/admin/status").json()
        shards = {i: {"rss_mb": s["rss_mb"], "peak_rss_mb": s["peak_rss_mb"]} for i, s in st["shards"].items()}
        return {"shards": shards, "max_shard_peak_rss_mb": max(s["peak_rss_mb"] for s in shards.values())}


def evaluate(searcher: Searcher, split: str = "dev", cold_n: int = 20, warmup_n: int = 100,
             limit: int | None = None) -> dict:
    """Run the full protocol on one split and return everything as one JSON-able dict.

    Order matters, because each phase changes the state the next one sees:
      1. cold    time the first ``cold_n`` (20) split queries in this FRESH process.
                 "Cold" = nothing is warmed up yet: first model inference, empty
                 CPU caches, new connections. This is what the first users after a
                 restart experience.
      2. warmup  run ``warmup_n`` (100) random TRAIN queries, untimed and unscored,
                 so caches/connections are "warm" (steady state).
      3. warm    time and record EVERY split query: this run is also what the
                 quality metrics are computed from.
    Returns {"split", "overall", "stratified", "latency", "memory", "run"}.
    """
    queries, qrels = load_split(split)
    # list(...)[:None] keeps everything, so --limit is optional. (queries come in a stable,
    # sorted id order, see beir_data.load_split, so "first N" is reproducible.)
    items = list(queries.items())[:limit]

    # Times ONE search from the client side, end to end. time.perf_counter() is a
    # high-resolution clock meant for measuring durations; (end - start) is in
    # seconds, * 1e3 converts to milliseconds.
    def timed(q: str):
        t0 = time.perf_counter()
        ids = searcher.search(q)
        return ids, (time.perf_counter() - t0) * 1e3

    # timed(q)[1] keeps only the milliseconds; cold results are not scored.
    # NOTE: with --limit below 20, the cold sample has fewer than 20 queries.
    cold = [timed(q)[1] for _, q in items[:cold_n]]  # 1) fresh process: first N queries
    try:
        train_q, _ = load_split("train")
    except FileNotFoundError:  # WHY: the tiny toy dataset has only a dev split; never warm up on dev queries
        train_q = {}
        print("[eval] no train split in this dataset: skipping warmup (warm latency is less meaningful)")
    # A seeded random generator: the SAME 100 training queries are picked on every run,
    # so warmups are comparable across configurations. With no train split,
    # min(100, 0) = 0 -> sample() returns [] and the loop does nothing.
    rng = random.Random(SEED)
    for q in rng.sample(list(train_q.values()), min(warmup_n, len(train_q))):  # 2) warmup on TRAIN queries
        searcher.search(q)
    # run = {qid: [doc ids best first]} (scored below); warm = one latency in ms per query.
    # PITFALL (subtle): the first 20 queries were already searched in the cold phase, so
    # a cache could make them a bit faster here; they are timed again anyway.
    run, warm = {}, []
    for qid, q in items:  # 3) the measured run
        run[qid], ms = timed(q)
        warm.append(ms)
    # Quality: nDCG@10 / Recall@10 / MRR@10 averaged over queries (src/metrics.py).
    # Floats are rounded to 4 decimals for a readable JSON; "n" (an int) is kept as-is.
    overall = {k: round(v, 4) if isinstance(v, float) else v for k, v in evaluate_run(run, qrels).items()}
    return {
        "split": split, "overall": overall,
        # The "10% longest docs" are computed over the WHOLE corpus, not just the gold docs.
        "stratified": stratified(run, qrels, queries, long_doc_ids(load_corpus())),
        "latency": {"cold_first20": latency_summary(cold), "warm": latency_summary(warm)},
        # Memory is read AFTER all queries ran, so the peak includes query-time usage.
        # The raw run is saved too, so metrics can be recomputed later without re-searching.
        "memory": searcher.memory(), "run": run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate one retrieval configuration (quality, latency, memory). "
                    "Full explanations: the ARGUMENTS section at the top of src/eval.py.",
        epilog="The shipped system: --strategy hybrid --model minilm --fusion linear --alpha 0.7 "
               "(the --fusion/--alpha defaults are rrf/0.5, NOT the shipped values).")
    ap.add_argument("--mode", choices=["local", "service"], default="local",
                    help="local: build the engine inside this process (no network). "
                         "service: send queries over HTTP to a running router + shards (Part B)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="service mode only: the router's address")
    ap.add_argument("--name", required=True,
                    help="label for this run; output file is runs/<name>.json (same name = overwrite)")
    ap.add_argument("--strategy", choices=["bm25", "dense", "hybrid"], default="hybrid",
                    help="bm25 = keywords only, dense = meaning only, hybrid = both fused")
    ap.add_argument("--model", default=PRIMARY_MODEL,
                    help="embedding model for dense: minilm or bge-small (local mode only)")
    ap.add_argument("--fusion", choices=["rrf", "linear"], default="rrf",
                    help="hybrid only: rrf = combine ranks, linear = min-max scores then weighted sum")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="hybrid only: weight on DENSE (BM25 gets 1-alpha); 0.5 with rrf = plain RRF")
    ap.add_argument("--rerank-k", type=int, default=0,
                    help="0 = off; N = re-score the fused top N with the cross-encoder")
    ap.add_argument("--bm25-stats", choices=["global", "local"], default="global",
                    help="global = IDF from all shards (correct); local = per-shard IDF (the ch07 bug). "
                         "No effect in local mode (one shard)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="dev (500 questions, for tuning) or test (648, the final exam)")
    ap.add_argument("--final-test-run", action="store_true",
                    help="safety lock: required to touch the test split")
    ap.add_argument("--limit", type=int, help="evaluate only the first N questions")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads for query encoding (local mode); fixed so latencies are comparable")
    ap.add_argument("--out", type=Path, help="results file path (default: RESULTS_DIR/runs/<name>.json)")
    a = ap.parse_args()
    # The safety lock: checked BEFORE anything expensive is loaded.
    if a.split == "test" and not a.final_test_run:
        raise SystemExit("Refusing to evaluate on TEST without --final-test-run (tune on dev only!)")
    # Imported here, not at the top, so `--help` and the test-split refusal stay instant.
    import torch

    # WHY fix the thread count: torch otherwise picks one based on the machine/load,
    # and latencies measured with different thread counts aren't comparable.
    torch.set_num_threads(a.threads)
    # Start-up time = building the Searcher (local mode: load corpus + indexes + model).
    # Reported separately as startup_s; it is NOT part of any query latency.
    t0 = time.time()
    searcher = Searcher(a.mode, a.url, strategy=a.strategy, model=a.model, fusion=a.fusion, alpha=a.alpha,
                        rerank_k=a.rerank_k, bm25_stats=a.bm25_stats)
    load_s = time.time() - t0
    res = evaluate(searcher, a.split, limit=a.limit)
    # Record the exact flags next to the results so every JSON file says how it was produced.
    # vars(a) = the parsed arguments as a dict; out/url/final_test_run are left out
    # because they say where/how to run, not what system was tested.
    res.update(name=a.name, config={k: v for k, v in vars(a).items() if k not in ("out", "url", "final_test_run")},
               startup_s=round(load_s, 1))
    out = a.out or RESULTS_DIR / "runs" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    # One summary line per run; {a.name:28s} pads the name to 28 characters so lines from
    # several runs (e.g. scripts/bench.py) line up as a table.
    o, lat = res["overall"], res["latency"]
    print(f"{a.name:28s} nDCG@10={o['ndcg@10']:.4f} R@10={o['recall@10']:.4f} MRR@10={o['mrr@10']:.4f} "
          f"warm p50/p95={lat['warm']['p50_ms']}/{lat['warm']['p95_ms']}ms "
          f"cold p50/p95={lat['cold_first20']['p50_ms']}/{lat['cold_first20']['p95_ms']}ms")


if __name__ == "__main__":
    main()
