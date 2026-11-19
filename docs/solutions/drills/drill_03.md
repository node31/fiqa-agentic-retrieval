# Drill 3 solution: Rerank in the default path

The router already supports `rerank_k`. Change the default in `SearchReq` (or send
`rerank_k=20` from the eval) and run:

```bash
python -m src.eval --mode service --url http://127.0.0.1:8000 --strategy hybrid --rerank-k 20 --name svc3-rerank20
```
Compare with `results/ablation_rerank_minilm.json`. Reranking needs the passage **text**, so the router
adds a fetch phase (one more fan-out) plus 20 cross-encoder passes. On our CPU, check whether
p95 ≤ 50 ms still holds. If not, it can't be the chosen operating point (the assignment says such configs are
still valid in the comparison table, just not as the chosen point).
