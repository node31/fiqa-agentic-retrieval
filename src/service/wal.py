"""Write-ahead log (WAL) + a readers-writer lock: the two primitives behind safe live updates.

WRITE-AHEAD LOG
    Rule: *append the intent to durable storage BEFORE changing in-memory state*.
    If the process crashes at any point:
      * before the append  -> the write never happened. The client got no ack and retries.
      * after the append   -> on restart we replay the log and re-apply the write.
    Replaying a write twice is harmless because writes are versioned and
    idempotent (see store.py). So the log only needs "at-least-once" semantics.
    A crash *during* the append leaves a torn last line, which replay() skips. That
    is safe, because that write was never acknowledged.
    See docs/09_live_updates_consistency.md and examples/17_wal_crash_recovery.py.

READERS-WRITER LOCK
    Many searches may run in parallel (they only read), but a write must update
    BM25, every dense index, the doc store and the version table *together*. If a
    reader ran in the middle, it could find a doc in BM25 whose vector does not
    exist yet. The RW lock gives writers exclusivity and readers concurrency.
    It prefers writers, so a stream of reads cannot starve a write.

PITFALLS
    * ``flush()`` only moves data to the OS. ``os.fsync`` is what survives power loss.
    * Never hold the write lock while doing slow work (embedding). Compute first,
      lock briefly to apply (store.py follows this rule).
"""
from __future__ import annotations

import base64
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


class RWLock:
    """Writer-preferring readers-writer lock built on one Condition variable."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            # WHY also wait on _waiting_writers: writer preference, so no starvation.
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def encode_vec(v: np.ndarray) -> str:
    """float32 vector -> compact base64 string for JSON."""
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode("ascii")


def decode_vec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


class WAL:
    """Append-only JSON-lines log with fsync on every append."""

    def __init__(self, path: Path | str, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._f = open(self.path, "a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())  # WHY: durability across power loss, not just process crash

    def replay(self) -> Iterator[dict]:
        """Yield every complete record. A torn (partial) final line is skipped."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # PITFALL: a crash mid-append leaves half a line. It was never acked,
                    # so dropping it is correct. Anything else would invent data.
                    continue

    def close(self) -> None:
        self._f.close()
