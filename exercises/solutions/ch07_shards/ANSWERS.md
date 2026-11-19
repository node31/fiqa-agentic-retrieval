# ch07 worksheet: answers

> Fill in `exercises/ch07_shards/PAPER.md` yourself **before** reading this.

**1.** Python salts `hash()` of strings with a random seed **per process** (`PYTHONHASHSEED`, a defence against
hash-flooding attacks). The router and every shard are separate processes, so they would disagree about which shard
owns a doc: writes go to one shard, reads ask another. `zlib.crc32(bytes)` is a fixed function, the same answer on every machine, forever.

**2.** About **75% (3/4)** of docs move. A doc stays only if `h % 3 == h % 4`. Over h = 0…11 (the pattern repeats
every 12) that holds for h = 0, 1, 2 only → 3/12 = 25% stay. (Measured on 100,000 ids: 74.98% move.)
Modulo hashing reshuffles almost everything when N changes. **Consistent hashing** would move only ~1/4, i.e. only the docs
the new shard takes over (toy 15).

**3.**
- local idf on A = ln(1 + (5 − 4 + 0.5)/(4 + 0.5)) = ln(1.333) = **0.288**
- local idf on B = ln(1 + (5 − 1 + 0.5)/(1 + 0.5)) = ln(4) = **1.386**
- global idf = ln(1 + (10 − 5 + 0.5)/(5 + 0.5)) = ln(2) = **0.693**

The same doc scores ~4.8× higher on **shard B**. That's wrong because a score must depend on the doc and the query, not on
which shard a random hash put the doc on. The router merges lists from all shards by score, so those scores must be on one scale.
With local stats "dividend" looks rare on B and common on A, purely by accident.
The fix is two phases: (1) the router collects N, total length and df from every shard and sums them, (2) every shard
scores with the summed stats. With that fix, 3 shards give exactly the single-node scores (checked on all 500 dev queries in `--full`).

**4.** Each shard returns its own top-k (computed with global stats), and the router merges them and takes the top k.
Proof that this is enough: take any doc d in the true global top-k. Fewer than k docs in the whole corpus beat d, so
fewer than k docs *on d's shard* beat it. So d is in its shard's top-k and reaches the router. ∎
(This relies on scores being comparable across shards, i.e. global stats. For fusion, each shard returns its top-100
BM25 and top-100 dense, and the router merges each globally *then* fuses: toy 13.)

**5.**

| write | outcome |
|---|---|
| upsert v1 | **applied** |
| upsert v1 (retry) | **no-op** (same version: idempotent) |
| upsert v3 | **applied** |
| upsert v2 (late) | **no-op** (older than v3) |
| delete v4 | **applied** (stores the tombstone (4, None)) |
| upsert v3 (very late) | **no-op** (the tombstone remembers v4) |

**Not visible** at the end. Without the tombstone, i.e. if delete just removed the entry, the very late v3 would
*resurrect* the deleted doc.

**6.** Ack **after** the WAL fsync. The timeline is: receive → append to the WAL → flush + fsync → apply to the indexes → ack.
- Crash before the fsync: the client got no ack, so it retries. Idempotent versions make the retry safe.
- Crash after the fsync: replay on restart re-applies the write.

If you ack *before* the fsync, a crash can lose a write the client was told succeeded. That's the one unforgivable outcome.

**7.** The next append is glued to the torn fragment on the same line: `{"op":"ups{"op":"upsert","doc_id":"c",…}`.
That line isn't valid JSON, so the **acked** write c is skipped at the following restart, and data is lost.
The fix is to *truncate* the torn tail when opening the WAL. We hit exactly this bug in the real project
(`src/service/wal.py` `_truncate_torn_tail`, plus a regression test). The ch07 checker's crash-twice test catches it.
