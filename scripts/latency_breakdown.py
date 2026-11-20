"""Where does a hybrid query's time go? Single index vs 3 shards, with NO network.

Question this answers (ch07 "What sharding cost in latency"):
    Is "split search + merge" actually cheaper than "search the whole index"?
    And how much of the 3-shard service latency is network/coordination, not search?

What it does (in ONE process, so there are no HTTP hops at all):
    * builds one BM25 + one faiss index over all 57,638 docs  ("single")
    * builds 3 BM25 + 3 faiss indexes, one per crc32 shard     ("sharded")
    * for every dev query (500), times each step with the real repo functions:
        encode (once, shared)  |  single: bm25 + faiss + fuse
        sharded: DFS stats (local_stats + merge) | per-shard bm25 + faiss | merge_shard_lists x2 + fuse
    * "sharded, parallel" = the SLOWEST shard's search time (what 3 processes running at the same time would cost),
      "sharded, sequential" = the sum of the 3 shards (one process doing them one after another)

Settings match scripts/microbench.py and the benchmark: 4 torch threads, 4 faiss threads, depth 100,
linear fusion alpha 0.7, MiniLM vectors from the embedding cache (so no corpus re-encoding).

    python -m scripts.latency_breakdown          # ~1-2 min; writes results/latency_breakdown.json
"""
import json
import statistics
import time

import faiss
import torch

from src.beir_data import load_corpus, load_split
from src.config import CANDIDATE_DEPTH, DEFAULT_ALPHA, RESULTS_DIR
from src.index.build_index import load_cache
from src.index.sharding import shard_of
from src.retrieval.bm25 import BM25Index, CorpusStats, tokenize
from src.retrieval.dense import DenseIndex, get_embedder
from src.retrieval.hybrid import fuse, merge_shard_lists

THREADS, N_SHARDS, MODEL = 4, 3, "minilm"
torch.set_num_threads(THREADS)
faiss.omp_set_num_threads(THREADS)


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3


def p(values: list[float], q: int) -> float:
    # Nearest-rank percentile on the sorted list (same idea as src/util.percentile).
    s = sorted(values)
    return round(s[max(0, min(len(s) - 1, round(q / 100 * len(s) + 0.5) - 1))], 3)


def main() -> None:
    corpus = load_corpus()
    queries, _ = load_split("dev")
    ids, _, mat = load_cache(MODEL)
    row = {d: i for i, d in enumerate(ids)}

    # ---- build: one whole-corpus index pair, and one pair per shard -----------------------
    all_ids = list(corpus)
    single_bm = BM25Index().build(all_ids, [corpus[d] for d in all_ids])
    single_dn = DenseIndex(mat.shape[1])
    single_dn.add(all_ids, mat[[row[d] for d in all_ids]])
    shards = []
    for s in range(N_SHARDS):
        sid = [d for d in all_ids if shard_of(d, N_SHARDS) == s]
        dn = DenseIndex(mat.shape[1])
        dn.add(sid, mat[[row[d] for d in sid]])
        shards.append((BM25Index().build(sid, [corpus[d] for d in sid]), dn))
    emb = get_embedder(MODEL)

    steps: dict[str, list[float]] = {}

    def rec(name: str, value: float) -> None:
        steps.setdefault(name, []).append(value)

    qtexts = list(queries.values())
    for i, text in enumerate(qtexts[:20] + qtexts):  # first 20 = warm-up, not recorded
        record = i >= 20
        t0 = time.perf_counter()
        terms = list(dict.fromkeys(tokenize(text)))
        q = emb.encode_queries([text])[0]
        t_encode = ms(t0)

        # Single index: search everything, then fuse. No stats round, no merge.
        t0 = time.perf_counter()
        bm = single_bm.search(terms, CANDIDATE_DEPTH)
        t_s_bm = ms(t0)
        t0 = time.perf_counter()
        dn = single_dn.search(q, CANDIDATE_DEPTH)
        t_s_dn = ms(t0)
        t0 = time.perf_counter()
        fuse(bm, dn, "linear", DEFAULT_ALPHA, top=10)
        t_s_fuse = ms(t0)

        # Sharded: DFS stats, per-shard search with the global stats, then merge + fuse.
        t0 = time.perf_counter()
        g = CorpusStats(0, 0.0, {})
        for sbm, _ in shards:
            g = g.merge(sbm.local_stats(terms))
        t_dfs = ms(t0)
        per_shard, bms, dns = [], [], []
        for sbm, sdn in shards:
            t0 = time.perf_counter()
            bms.append(sbm.search(terms, CANDIDATE_DEPTH, g))
            t_bm = ms(t0)
            t0 = time.perf_counter()
            dns.append(sdn.search(q, CANDIDATE_DEPTH))
            per_shard.append((t_bm, ms(t0)))
        t0 = time.perf_counter()
        fuse(merge_shard_lists(bms, CANDIDATE_DEPTH), merge_shard_lists(dns, CANDIDATE_DEPTH), "linear",
             DEFAULT_ALPHA, top=10)
        t_merge_fuse = ms(t0)

        if record:
            rec("encode", t_encode)
            rec("single.bm25", t_s_bm)
            rec("single.faiss", t_s_dn)
            rec("single.fuse", t_s_fuse)
            rec("single.search_total", t_s_bm + t_s_dn + t_s_fuse)
            rec("sharded.dfs_stats", t_dfs)
            rec("sharded.bm25_per_shard_max", max(b for b, _ in per_shard))
            rec("sharded.faiss_per_shard_max", max(d for _, d in per_shard))
            rec("sharded.search_parallel", max(b + d for b, d in per_shard))
            rec("sharded.search_sequential", sum(b + d for b, d in per_shard))
            rec("sharded.merge_and_fuse", t_merge_fuse)
            rec("sharded.total_parallel", t_dfs + max(b + d for b, d in per_shard) + t_merge_fuse)

    out = {"threads": THREADS, "n_queries": len(qtexts), "depth": CANDIDATE_DEPTH,
           "shard_sizes": [len(b) for b, _ in shards],
           "steps_ms": {k: {"p50": p(v, 50), "p95": p(v, 95), "mean": round(statistics.mean(v), 3)}
                        for k, v in steps.items()}}
    (RESULTS_DIR / "latency_breakdown.json").write_text(json.dumps(out, indent=1))
    for k, v in out["steps_ms"].items():
        print(f"{k:32s} p50 {v['p50']:7.3f}  p95 {v['p95']:7.3f}  mean {v['mean']:7.3f}")


if __name__ == "__main__":
    main()
