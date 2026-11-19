# Drill 9 solution: Confidence gating for the single agent

```python
first = tool(question, "hybrid")
if first and first[0]["score"] >= tau:            # linear-fusion scores are in [0, 1]
    return AgentResult([h["doc_id"] for h in first], path="single:gated", searches=1)
```
Tune τ on the plain dev sample so recall does not drop. Measure the LLM-call reduction. Pitfall:
fused scores are *relative* (min-max within the query), so the top-1 is often near α. Use the
*dense cosine* of the top-1 or the gap to rank 2 as the confidence signal instead.
