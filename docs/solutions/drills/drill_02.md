# Drill 2 solution: z-score fusion

```python
# src/retrieval/hybrid.py
def zscore(lst):
    if len(lst) < 2:
        return {d: 0.0 for d, _ in lst}
    xs = [s for _, s in lst]
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    return {d: (s - mu) / sd for d, s in lst}

# in fuse():
if method == "zscore":
    nb, nd = zscore(bm25), zscore(dense)
    lo = min([*nb.values(), *nd.values(), 0.0])          # missing docs get the minimum
    docs = set(nb) | set(nd)
    return _sorted({d: alpha * nd.get(d, lo) + (1 - alpha) * nb.get(d, lo) for d in docs}, top)
```
Test: two lists where one has an outlier. z-score is less squashed than min-max. Add `("zscore",
ALPHAS_LIN)` to `scripts/ablations.py`. Also add "zscore" to the `--fusion` choices in `src/eval.py`
and to the router's `SearchReq` (any string passes today, but `fuse` raises on unknown values).
