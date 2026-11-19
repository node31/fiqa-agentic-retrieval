# Drill 6 solution: WAL snapshots

Every N applied writes (under the write lock):
1. write `snapshot.tmp` = {docs, versions, models, dense vectors (npy), wal_offset}; `fsync`; atomic `os.replace` to `snapshot`.
2. truncate the WAL (or start `wal.<seq>.jsonl` and delete older segments).

Recovery: load the snapshot instead of the base corpus, then replay WAL records *after* the snapshot.
Replay stays idempotent, so a crash between steps 1 and 2 is harmless (records get applied twice
→ no-ops). Pitfalls: snapshotting under the lock blocks writes (copy-on-write or a background thread
from a consistent copy instead); the BM25 index can be rebuilt from docs instead of serialised.
