"""Toy 12: the sharded-IDF bug. Same document, different score, depending on which shard holds it.

Concept: BM25 needs corpus-wide statistics, but shards only know their own slice.
Fix: the DFS phase (router sums N, total length and df across shards, then sends them back).
Real code: src/service/router.py::Router.search (phase 1)   Doc: docs/08_sharding_scatter_gather.md
"""
import json

from _common import ROOT, banner, note, step, try_this

from src.retrieval.bm25 import BM25Index, tokenize

banner("Toy 12: local IDF vs global IDF across 2 shards")
shard_a = {  # a shard that happens to hold MANY docs about 'dividends'
    "a1": "dividend stocks pay a dividend every quarter", "a2": "dividend reinvestment plans",
    "a3": "qualified dividend tax rate", "a4": "dividend yield explained", "a5": "my answer about dividend ETFs"}
shard_b = {  # a shard where 'dividend' is rare
    "b1": "my answer about dividend ETFs", "b2": "mortgage refinancing basics", "b3": "how to open a roth ira",
    "b4": "credit card interest", "b5": "emergency fund size"}
A = BM25Index().build(list(shard_a), list(shard_a.values()))
B = BM25Index().build(list(shard_b), list(shard_b.values()))
terms = tokenize("dividend ETFs")
step(1, "a5 and b1 have IDENTICAL text. Score them with each shard's LOCAL stats:")
la, lb = dict(A.search(terms, 5)), dict(B.search(terms, 5))
print(f"     a5 (on shard A) = {la['a5']:.3f}     b1 (on shard B) = {lb['b1']:.3f}")
note("On shard A 'dividend' is common (low IDF), on shard B it is rare (high IDF). The merged ranking now "
     "depends on WHERE a doc was hashed, not on its relevance.")
g = A.local_stats(terms).merge(B.local_stats(terms))
step(2, f"DFS phase: global stats = sum over shards -> N={g.n_docs}, df={g.df}")
ga, gb = dict(A.search(terms, 5, g)), dict(B.search(terms, 5, g))
print(f"     a5 = {ga['a5']:.3f}     b1 = {gb['b1']:.3f}   (equal, as they should be)")
bench = ROOT / "results" / "bench.json"
cfg = json.loads(bench.read_text())["configs"] if bench.exists() else {}
if "svc3-bm25" in cfg and "svc3-bm25-localidf" in cfg:
    g_, l_ = cfg["svc3-bm25"], cfg["svc3-bm25-localidf"]
    cost = (f"in results/bench.json warm p50 is {g_['latency']['warm']['p50_ms']} ms with DFS vs "
            f"{l_['latency']['warm']['p50_ms']} ms without, and nDCG@10 is {g_['overall']['ndcg@10']} vs "
            f"{l_['overall']['ndcg@10']}")
else:
    cost = "compare svc3-bm25 with svc3-bm25-localidf in results/bench.json"
note(f"Cost of the fix: one extra router->shards round trip per query ({cost}). With random hash sharding over "
     "57K docs, local and global stats differ only slightly, so the QUALITY gap is tiny either way, but the scores "
     "become exactly reproducible and equal to single-node.")
try_this("Shard by TOPIC (all dividend docs on one shard) instead of by hash. How big does the error get?",
         "Cache global df at the router and refresh every 10 s. What breaks with live writes?",
         "Why does dense retrieval NOT need a DFS phase?")
