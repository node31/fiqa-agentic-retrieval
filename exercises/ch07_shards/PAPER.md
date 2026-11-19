# ch07 worksheet (pen and paper first)

1. Why must the router and the shards never use Python's built-in `hash(doc_id)` to pick a shard? ____________
2. Growing from 3 to 4 shards with `crc32 % n`, roughly what fraction of docs move? ____ Why? ____________
3. Shard A has 5 docs, 4 containing "dividend". Shard B has 5 docs, 1 containing "dividend". N = 10 total.
   - local idf("dividend") on A = ln(1 + (5 − 4 + 0.5)/(4 + 0.5)) = ____ ; on B = ____
   - global idf = ln(1 + (10 − 5 + 0.5)/(5 + 0.5)) = ____
   - So an identical doc scores higher on shard ____ with local stats. Why is that wrong?
4. Why must each shard return its top-k, and why is that enough for the global top-k? (prove it in 2 lines)
5. Write the outcome (applied / no-op) of each write for doc x:
   upsert v1 → ____ ; upsert v1 (retry) → ____ ; upsert v3 → ____ ; upsert v2 (late) → ____ ;
   delete v4 → ____ ; upsert v3 (very late) → ____ ; is x visible at the end? ____
6. Draw the crash timeline: when is it safe to ack the client: before or after the WAL fsync? ____________
7. A crash leaves half a line at the end of the WAL. If you only *skip* it during replay, what goes wrong on the
   NEXT write + restart? ____________

---
✅ Answers: [`solutions/ch07_shards/ANSWERS.md`](../solutions/ch07_shards/ANSWERS.md). Open it only *after* you've written yours.
