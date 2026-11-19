"""ch07: sharding, a router with global statistics, versioned writes and a WAL, all in one process.

Run the checker:   python -m pytest exercises/ch07_shards
Run the demo:      python exercises/ch07_shards/mini_shards.py
"""
from __future__ import annotations

import json  # noqa: F401
import os  # noqa: F401
import sys
import zlib  # noqa: F401
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import use_chapter  # noqa: E402

use_chapter("ch03_bm25")
from my_bm25 import B, K1, idf, tokenize  # noqa: E402,F401  (your ch03 code)

Ranked = list[tuple[str, float]]


# ---- 1. partitioning ----------------------------------------------------------------------
def shard_of(doc_id: str, n_shards: int) -> int:
    """crc32(doc_id as UTF-8 bytes) % n_shards. Stable across processes (unlike hash())."""
    raise NotImplementedError("TODO (ch07): implement shard_of with zlib.crc32")


# ---- 2. a shard and the router ------------------------------------------------------------------
class Shard:
    """Holds SOME of the docs. Knows only its own docs, so its local statistics are partial."""

    def __init__(self, docs: dict[str, str]):
        raise NotImplementedError("TODO (ch07): tokenise docs; keep tf per doc and doc lengths")

    def stats(self, terms: list[str]) -> dict:
        """This shard's numbers: {"n_docs": int, "total_len": int, "df": {term: df on THIS shard}}."""
        raise NotImplementedError("TODO (ch07): implement Shard.stats")

    def search(self, terms: list[str], k: int, stats: dict) -> Ranked:
        """BM25 over this shard's docs, using the GIVEN stats (n_docs, total_len, df) for idf and avgdl.
        Return top-k [(doc_id, score)] with score > 0, sorted by (-score, doc_id)."""
        raise NotImplementedError("TODO (ch07): implement Shard.search")


def router_search(shards: list[Shard], query: str, k: int = 10, use_global_stats: bool = True) -> Ranked:
    """Scatter-gather.
    use_global_stats=True : phase 1 sum every shard's stats → phase 2 every shard searches with the SUM.
    use_global_stats=False: every shard searches with its OWN stats (the classic bug).
    Merge all shards' lists, sort by (-score, doc_id), return the top-k. Query terms: dedupe tokenize(query)."""
    raise NotImplementedError("TODO (ch07): implement router_search")


# ---- 3. versioned, idempotent writes --------------------------------------------------------------
class VersionedStore:
    """doc_id -> (version, text or None for a tombstone). A write applies ONLY if its version is newer."""

    def __init__(self):
        self.versions: dict[str, tuple[int, str | None]] = {}

    def upsert(self, doc_id: str, version: int, text: str) -> bool:
        """Apply if version > stored version (or unseen). Return True if applied, False if it was a no-op."""
        raise NotImplementedError("TODO (ch07): implement upsert")

    def delete(self, doc_id: str, version: int) -> bool:
        """Like upsert, but store a tombstone (version, None) so an older upsert can't resurrect the doc."""
        raise NotImplementedError("TODO (ch07): implement delete")

    def get(self, doc_id: str) -> str | None:
        """Current text, or None if unknown or deleted."""
        raise NotImplementedError("TODO (ch07): implement get")


# ---- 4. write-ahead log ---------------------------------------------------------------------------
class WAL:
    """Append-only JSON-lines file. One record per line, fsync on every append."""

    def __init__(self, path: str | Path):
        """Open for appending. FIRST truncate a torn (partial) last line left by a crash."""
        raise NotImplementedError("TODO (ch07): implement WAL.__init__ (truncate torn tail, then open 'a')")

    def append(self, record: dict) -> None:
        """Write json.dumps(record) + '\\n', flush, os.fsync."""
        raise NotImplementedError("TODO (ch07): implement WAL.append")

    def replay(self) -> list[dict]:
        """Every complete record in file order (skip a line that isn't valid JSON)."""
        raise NotImplementedError("TODO (ch07): implement WAL.replay")

    def close(self) -> None:
        raise NotImplementedError("TODO (ch07): implement WAL.close")


if __name__ == "__main__":
    import tempfile
    import time

    from common import dataset_from_argv

    use_chapter("ch01_data")
    from loader import load_jsonl

    name, data_dir = dataset_from_argv()                 # add --full for the real 57,638 docs (3 shards)
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    n = 2 if name == "tiny" else 3
    t0 = time.perf_counter()
    parts = [{d: t for d, t in corpus.items() if shard_of(d, n) == i} for i in range(n)]
    shards = [Shard(p) for p in parts]
    print(f"[{name}] shard sizes: {[len(p) for p in parts]} (built in {time.perf_counter() - t0:.1f}s)")
    for flag in (True, False):
        print(f"global_stats={flag}:", [(d, round(s, 3)) for d, s in router_search(shards, "short selling", 5, flag)])
    store = VersionedStore()
    print("writes:", [store.upsert("x", 1, "v1"), store.upsert("x", 1, "v1"), store.upsert("x", 3, "v3"),
                      store.upsert("x", 2, "OLD"), store.delete("x", 4), store.upsert("x", 3, "late")], store.get("x"))
    with tempfile.TemporaryDirectory() as tmp:
        w = WAL(Path(tmp) / "wal.jsonl"); w.append({"op": "upsert", "doc_id": "a", "version": 1}); w.close()
        with open(Path(tmp) / "wal.jsonl", "ab") as f:
            f.write(b'{"op":"ups')  # crash mid-append
        w = WAL(Path(tmp) / "wal.jsonl"); w.append({"op": "upsert", "doc_id": "c", "version": 1}); w.close()
        print("after crash + write + restart:", WAL(Path(tmp) / "wal.jsonl").replay())
