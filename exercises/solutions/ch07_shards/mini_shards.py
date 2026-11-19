"""SOLUTION for ch07: sharding, a router with global statistics, versioned writes and a WAL, in one process.

Try the exercise yourself first (exercises/ch07_shards/mini_shards.py). Compare afterwards.
Check this file: $env:EXERCISES_SOLUTIONS="1"; python -m pytest exercises\\ch07_shards [--full]
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("EXERCISES_SOLUTIONS", "1")   # WHY: make use_chapter() load the ch03 SOLUTION
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import dataset_from_argv, use_chapter  # noqa: E402

use_chapter("ch03_bm25")
from my_bm25 import B, K1, idf, tokenize  # noqa: E402

Ranked = list[tuple[str, float]]


# ---- 1. partitioning ----------------------------------------------------------------------
def shard_of(doc_id: str, n_shards: int) -> int:
    # PITFALL: Python's hash("abc") is randomised per process (PYTHONHASHSEED), so the router and the shards
    # would disagree about where a doc lives. crc32 is a fixed function of the bytes: same answer everywhere.
    return zlib.crc32(doc_id.encode("utf-8")) % n_shards


# ---- 2. a shard and the router ------------------------------------------------------------------
class Shard:
    """A slice of the corpus with its own small inverted index."""

    def __init__(self, docs: dict[str, str]):
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_len: dict[str, int] = {}
        for doc_id, text in docs.items():
            tf = Counter(tokenize(text))
            self.doc_len[doc_id] = sum(tf.values())
            for term, count in tf.items():
                self.postings[term].append((doc_id, count))
        self.total_len = sum(self.doc_len.values())

    def stats(self, terms: list[str]) -> dict:
        """Phase 1 of a query: 'how many docs do I have, how long are they, how many contain each term?'"""
        return {"n_docs": len(self.doc_len), "total_len": self.total_len,
                "df": {t: len(self.postings.get(t, ())) for t in terms}}

    def search(self, terms: list[str], k: int, stats: dict) -> Ranked:
        """Phase 2: BM25 over MY docs, but with the stats I'm GIVEN (global ones, if the router is correct)."""
        avgdl = stats["total_len"] / max(1, stats["n_docs"])
        scores: dict[str, float] = defaultdict(float)
        for t in terms:
            df = stats["df"].get(t, 0)
            if df <= 0 or t not in self.postings:
                continue
            w = idf(stats["n_docs"], df)                    # WHY stats, not self: idf must be the same on every shard
            for doc_id, tf in self.postings[t]:
                norm = 1 - B + B * self.doc_len[doc_id] / avgdl
                scores[doc_id] += w * tf * (K1 + 1) / (tf + K1 * norm)
        return sorted(((d, s) for d, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))[:k]


def _sum_stats(parts: list[dict], terms: list[str]) -> dict:
    """Global stats = the SUM of every shard's counts (N, total length and df all add up exactly)."""
    return {"n_docs": sum(p["n_docs"] for p in parts),
            "total_len": sum(p["total_len"] for p in parts),
            "df": {t: sum(p["df"][t] for p in parts) for t in terms}}


def router_search(shards: list[Shard], query: str, k: int = 10, use_global_stats: bool = True) -> Ranked:
    terms = list(dict.fromkeys(tokenize(query)))          # dedupe, keep order
    if use_global_stats:
        g = _sum_stats([s.stats(terms) for s in shards], terms)     # phase 1: gather stats from all shards
        per_shard = [s.search(terms, k, g) for s in shards]         # phase 2: everyone scores on the same scale
    else:
        per_shard = [s.search(terms, k, s.stats(terms)) for s in shards]   # the bug: each shard its own scale
    # WHY top-k per shard is enough: a doc in the global top-k has < k docs above it overall, so it also has
    # < k docs above it on its own shard → it is in that shard's top-k. (Only true if scores are comparable!)
    merged = [hit for lst in per_shard for hit in lst]
    return sorted(merged, key=lambda x: (-x[1], x[0]))[:k]


# ---- 3. versioned, idempotent writes --------------------------------------------------------------
class VersionedStore:
    def __init__(self):
        self.versions: dict[str, tuple[int, str | None]] = {}

    def _newer(self, doc_id: str, version: int) -> bool:
        # WHY ">" and not ">=": a retry of the SAME version must be a no-op (idempotent), not a second apply.
        return doc_id not in self.versions or version > self.versions[doc_id][0]

    def upsert(self, doc_id: str, version: int, text: str) -> bool:
        if not self._newer(doc_id, version):
            return False
        self.versions[doc_id] = (version, text)
        return True

    def delete(self, doc_id: str, version: int) -> bool:
        if not self._newer(doc_id, version):
            return False
        # PITFALL: don't `del` the entry. The tombstone remembers "deleted at v4", so a late upsert v3 is rejected.
        self.versions[doc_id] = (version, None)
        return True

    def get(self, doc_id: str) -> str | None:
        return self.versions.get(doc_id, (0, None))[1]


# ---- 4. write-ahead log ---------------------------------------------------------------------------
class WAL:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._truncate_torn_tail()
        self._f = open(self.path, "a", encoding="utf-8")

    def _truncate_torn_tail(self) -> None:
        """A crash mid-append leaves a last line without "\\n". It was never acked, so cut it off.
        PITFALL: if we only SKIPPED it at replay, the next append would be glued onto it
        ('{"op":"ups{"op":"upsert",...}') and that ACKED record would be unreadable after the next restart."""
        if not self.path.exists():
            return
        with open(self.path, "r+b") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)       # keep everything up to the last complete line (or nothing)

    def append(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()                  # Python buffer → OS
        os.fsync(self._f.fileno())       # OS buffer → disk. Only after this may the client get its "ok".

    def replay(self) -> list[dict]:
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue             # defensive: a torn line can only be the last one, and we truncate it
        return out

    def close(self) -> None:
        self._f.close()


if __name__ == "__main__":
    import tempfile
    import time

    use_chapter("ch01_data")
    from loader import load_jsonl

    name, data_dir = dataset_from_argv()
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    n = 2 if name == "tiny" else 3
    t0 = time.perf_counter()
    shards = [Shard({d: t for d, t in corpus.items() if shard_of(d, n) == i}) for i in range(n)]
    print(f"[{name}] {n} shards, sizes {[len(s.doc_len) for s in shards]}, built in {time.perf_counter() - t0:.1f}s")
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
