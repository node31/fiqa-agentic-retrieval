# Solutions: migration

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** If the snapshot is taken, the lock released, then dual-write enabled, a write landing in between updates the old index only and is not in the snapshot: it never reaches the new index. Doing both under one write-lock acquisition makes them atomic w.r.t. writes.

**2.** Router.finalize raises 'cannot drop the active model' (HTTP 409). Dropping the serving index would make every dense query fail.

**3.** Run it: re-encoding competes for CPU with query encoding on each shard, so query p95 rises during the backfill; throttle_s trades migration duration against serving latency.

**4.** In Router.cutover, before flipping: evaluate 50 fixed dev queries with model=target vs current (both indexes exist), compute nDCG@10, raise if target < current − ε. Note the cost: 100 extra searches at cutover time.
