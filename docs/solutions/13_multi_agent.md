# Solutions: multi agent

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** (1) Cost and latency: every routing decision becomes an LLM call; (2) non-determinism and loops: an LLM orchestrator can re-plan forever or forget constraints; budgets and timeouts are trivially enforced in code and unit-testable.

**2.** ~3 slots each (10 = 4 + 3 + 3). For a 2-part question over-split into 3, one half is spread over two sub-queries: no coverage loss, but each sub-query keeps fewer of its top ranks, so recall per half drops.

**3.** Skip verify() when every worker's top-1 score (dense cosine) is above τ; saves 1 call on easy compound queries.

**4.** Coverage is preserved only if the reranker can't push one half out entirely. Reranking all 10 keeps the same set (same coverage) but reorders it, which may raise MRR. If you rerank a larger pool and take the top 10, coverage can drop.
