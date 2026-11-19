# Drill 8 solution: Router-side df cache

Keep `self._df_cache: dict[str, (count, ts)]` plus cached `n_docs/total_len`. On a query, use cached
df for terms younger than TTL; fetch only *missing* terms via the stats fan-out (often none → phase 1
skipped). Invalidate on writes the router forwards (it sees every write: increment/decrement is
imprecise because an upsert changes terms, so simply drop affected terms). Expect p50 ≈ local-IDF
latency (6.8 ms for BM25) with global-IDF quality. The staleness is bounded by the TTL; document it.
