"""Toy 17: write-ahead log. Crash mid-write, restart, replay, nothing acknowledged is lost.

Concept: log first, apply second; replay is idempotent; torn final records are skipped.
Real code: src/service/wal.py, ShardStore.load (replay)    Doc: docs/09_live_updates_consistency.md
"""
import tempfile
from pathlib import Path

from _common import TOY_DOCS, banner, note, step, try_this

from src.retrieval.dense import HashEmbedder
from src.service.store import ShardStore

banner("Toy 17: WAL crash recovery")
emb = HashEmbedder()
embed = lambda m, t: emb.encode_docs(list(t))  # noqa: E731
state = Path(tempfile.mkdtemp())
s = ShardStore(state_dir=state, embed=embed, use_cache=False).load(TOY_DOCS, ["minilm"])
step(1, "Acknowledged writes: upsert w1, upsert w2, delete w2")
s.upsert("w1", 1, "walrus pension plan")
s.upsert("w2", 1, "walrus annuity")
s.delete("w2", 2)
wal = state / "wal.jsonl"
for line in wal.read_text().splitlines():
    print(f"     WAL: {line[:95]}...")
step(2, "CRASH while appending a 4th record (power cut mid-line)")
with open(wal, "a") as f:
    f.write('{"op":"upsert","doc_id":"w3","vers')
s.wal.close()
del s
note("All in-memory state (BM25, vectors, versions) is gone. Only the base corpus + WAL survive on disk.")
step(3, "Restart: load base corpus, then replay the WAL")
s2 = ShardStore(state_dir=state, embed=embed, use_cache=False).load(TOY_DOCS, ["minilm"])
print(f"     replayed={s2.counters['replayed']}  w1={s2.docs.get('w1')!r}  w2 present={'w2' in s2.docs}  "
      f"w3 present={'w3' in s2.docs}")
note("w3's torn record was never acknowledged (the crash happened BEFORE the ack), so dropping it is correct.")
note("The WAL record also stores the embedding vectors, so replay does not need to re-run the model: "
     "recovery is fast and deterministic.")
step(4, "Crash during/after recovery, restart AGAIN: replaying the same WAL a second time is idempotent")
s2.wal.close()
s3 = ShardStore(state_dir=state, embed=embed, use_cache=False).load(TOY_DOCS, ["minilm"])
print(f"     replayed={s3.counters['replayed']}  same docs as after the 1st restart: {s3.docs == s2.docs}  "
      f"same versions: {s3.versions == s2.versions}")
print(f"     and a re-delivered upsert(w1, v1) on top -> applied={s3.upsert('w1', 1, 'walrus pension plan')['applied']}")
try_this("Remove os.fsync from WAL.append and explain what can be lost on POWER loss (not process crash).",
         "The WAL grows forever. Design snapshotting: dump state + truncate the log (log compaction).",
         "Apply first, log second. Construct a crash timing that loses an ACKNOWLEDGED write.")
