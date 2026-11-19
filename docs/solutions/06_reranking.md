# Solutions: reranking

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Rerank depth vs. quality is a recall-precision trade: deeper k lets the reranker rescue more relevant docs from ranks 11–k, at linear cost. Past the point where first-stage Recall@k saturates, extra depth buys nothing.

**2.** A cross-encoder cannot index a corpus: its score is a function of (query, passage) jointly; there's no per-passage representation to store.

**3.** See your run of examples/09: batching amortises Python/tokenizer overhead and uses matrix-matrix products (better CPU utilisation).
