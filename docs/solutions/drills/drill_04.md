# Drill 4 solution: Critic role (near-duplicate removal)

```python
# src/agent/orchestrator.py (after slot allocation)
def dedupe_near(ids, texts, threshold=0.9):
    """Drop a passage if its word-set Jaccard with an earlier kept passage is >= threshold."""
    kept, sets = [], []
    for d in ids:
        s = set(texts.get(d, "").lower().split())
        if all(len(s & t) / max(1, len(s | t)) < threshold for t in sets):
            kept.append(d); sets.append(s)
    return kept
```
Workers already return snippets (`h["text"]`): build `texts = {h["doc_id"]: h["text"] for lst in lists.values() for h in lst}`,
apply `dedupe_near` to the interleaved list *before* the fill, then fill to 10. Test: give two workers
identical texts under different ids and assert only one survives. Discuss: is it an *agent* if it uses no LLM?
(It's a role. An LLM critic would judge semantic duplicates but costs a call.)
