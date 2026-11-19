# Drill 10 solution: GET /doc/{id}

```python
# store.py
def get_doc(self, doc_id):
    with self.lock.read():
        v = self.versions.get(doc_id)
        return {"doc_id": doc_id, "text": self.docs.get(doc_id), "version": v[0] if v else None,
                "deleted": bool(v and v[1])}
# router.py: dispatch op "doc" -> store.get_doc(p["doc_id"]); route with shard_of(doc_id)
@app.get("/doc/{doc_id}")
async def doc(doc_id: str):
    i = shard_of(doc_id, len(router.shards))
    ok, missing = await router._fan("doc", {i: {"doc_id": doc_id}})
    ...
```
