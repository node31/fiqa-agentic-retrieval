# Drill 5 solution: BM25 compaction

```python
def compact(self) -> None:
    """Rebuild the base segment from live docs (base rows + delta), dropping tombstones."""
    live = sorted(self.row_of.items(), key=lambda x: x[1])
    counts = []
    for doc_id, row in live:
        if row < self._n_base:
            csr = self._base_csr
            lo, hi = csr.indptr[row], csr.indptr[row + 1]
            counts.append((doc_id, {self.terms[j]: int(tf) for j, tf in zip(csr.indices[lo:hi], csr.data[lo:hi])}))
        else:
            counts.append((doc_id, dict(self._delta_terms[row])))
    # rebuild via a tiny internal builder that takes term-count dicts instead of text
    ...
```
Simplest correct approach: keep a `self._text` dict (costs RAM), then `self.__init__(self.k1, self.b);
self.build(ids, texts)`. Invariant test: scores for a set of queries are identical before and after
`compact()`, while `len(self.doc_ids)` shrinks to `len(self)` and `_delta` is empty. Discuss *when* to
trigger it (dead fraction > 20%, delta size > 10K) and *how* to do it without blocking readers (build
a new index off-lock, then swap the reference under the write lock).
