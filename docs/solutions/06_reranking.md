# Solutions: reranking

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Rerank depth vs. quality is a recall-precision trade: deeper k lets the reranker rescue more relevant docs from ranks 11–k, at linear cost. Past the point where first-stage Recall@k saturates, extra depth buys nothing.

**2.** A cross-encoder cannot index a corpus: its score is a function of (query, passage) jointly; there's no per-passage representation to store.

**3.** See your run of examples/09: batching amortises Python/tokenizer overhead and uses matrix-matrix products (better CPU utilisation).

**4.** Reranking only the top-10 permutes the same 10 documents. Recall@10 asks "how many gold docs are in the set?", and the set is unchanged. MRR and nDCG depend on positions inside it, so they move (here: down).

**5.** λ = 0 reproduces the hybrid ranking (nDCG 0.419); λ = 1 reproduces pure CE reranking at that k (0.406 at k = 10). In between, the curve tends to peak somewhat below λ = 0.5 when the CE is weaker in-domain. Normalize per query over the candidate list only, or the scales will not be comparable.
