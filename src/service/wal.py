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
    See docs/topics/09_live_updates_consistency.md and examples/17_wal_crash_recovery.py.

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

GLOSSARY
    flush vs fsync  Python's file object keeps a buffer in the process. ``flush()`` hands it
                    to the operating system (now a process crash cannot lose it, but the OS
                    may still hold it in RAM). ``os.fsync(fd)`` asks the OS to push it to the
                    physical disk and waits, so it also survives power loss / an OS crash.
    torn write      a write that was cut off part-way (process killed mid-``write``), leaving
                    a partial last line such as ``{"op":"ups`` with no closing ``}`` or newline.
    at-least-once   every acked record is applied one or more times; idempotency makes the
                    "more" harmless.

WAL FILE FORMAT  (``<state_dir>/wal.jsonl``, one JSON object per line, oldest first)
    {"op":"upsert","doc_id":"42","version":7,"text":"...","vecs":{"minilm":"<base64>",...}}
    {"op":"delete","doc_id":"99","version":3,"text":null,"vecs":{}}
    The line ending ``\\n`` is the "commit marker": a line without one is a torn write.

HOW TO READ THIS FILE
    ``RWLock`` (the lock), then ``encode_vec``/``decode_vec`` (vectors <-> JSON-safe text),
    then ``WAL``: open (repair a torn tail) -> ``append`` (write + flush + fsync) ->
    ``replay`` (read back on restart).
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
    """Writer-preferring readers-writer lock built on one Condition variable.

    Usage: ``with lock.read(): ...`` (shared) or ``with lock.write(): ...`` (exclusive).
    A ``threading.Condition`` is a mutex plus a waiting room: ``wait()`` releases the mutex
    and sleeps until some other thread calls ``notify_all()``, then re-acquires it. Each
    waiter re-checks its condition in a ``while`` loop, because waking up does not
    guarantee the condition is now true (another thread may have got there first).

    PITFALL: not re-entrant. Taking ``write()`` while already holding ``read()`` (or
    ``write()``) on the same thread deadlocks: the thread waits for itself. Even a nested
    ``read()`` inside ``read()`` deadlocks if a writer queued up in between (writer preference).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        # State, only read/changed while holding self._cond:
        self._readers = 0            # readers currently inside a read() block
        self._writer = False         # True while one writer is inside a write() block
        self._waiting_writers = 0    # writers queued; new readers step aside for them

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            # WHY also wait on _waiting_writers: writer preference, so no starvation.
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        # The mutex is NOT held while the caller's read block runs; the counter alone
        # keeps writers out. That is what lets many readers run at the same time.
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                # The last reader out wakes any writer waiting for _readers to reach 0.
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            # Announce ourselves first, so readers arriving from now on queue behind us.
            self._waiting_writers += 1
            # Wait until no writer is active AND every current reader has left.
            while self._writer or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                # Wake everyone: waiting writers and readers re-check their conditions.
                self._cond.notify_all()


def encode_vec(v: np.ndarray) -> str:
    """float32 vector -> compact base64 string for JSON."""
    # WHY base64 of raw bytes (not a JSON list of floats): exact round-trip (no decimal
    # rounding) and ~4/3 x the 4 bytes/float, i.e. ~2 KB for 384 dims vs ~7 KB as text.
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode("ascii")


def decode_vec(s: str) -> np.ndarray:
    # Inverse of encode_vec. The result is a read-only view over the decoded bytes;
    # copy it before modifying in place.
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


class WAL:
    """Append-only JSON-lines log with fsync on every append."""

    def __init__(self, path: Path | str, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # fsync=False trades durability for speed (used by tests); a power loss could then
        # lose acked writes, a plain process crash still cannot (flush() reaches the OS).
        self.fsync = fsync
        # Repair BEFORE opening for append, otherwise the first new record would be glued
        # onto the torn fragment (see _truncate_torn_tail).
        self.truncated_bytes = self._truncate_torn_tail()
        # Mode "a" (append): every write goes to the end of the file; existing records are
        # never rewritten. The handle stays open for the WAL's lifetime (one open per append
        # would be much slower).
        self._f = open(self.path, "a", encoding="utf-8")

    def _truncate_torn_tail(self) -> int:
        """Cut a partial final line left by a crash mid-append. Returns the bytes removed.

        PITFALL (found by the toy audit): without this, the next append is glued onto the
        fragment, e.g. ``{"op":"ups{"op":"upsert",...}\\n``. That whole line then fails to parse
        on the NEXT replay, so a write that was acked AFTER recovery silently disappears.
        The fragment itself was never acked, so dropping it loses nothing.
        """
        if not self.path.exists():
            return 0
        # "r+b" = read and write, binary: we work on raw bytes (no newline translation,
        # no UTF-8 decoding of a fragment that may end mid-character).
        with open(self.path, "r+b") as f:
            data = f.read()
            # Empty file, or ends with "\n" = last record is complete: nothing to repair.
            if not data or data.endswith(b"\n"):
                return 0
            keep = data.rfind(b"\n") + 1  # 0 if the whole file is one torn line
            # Cut the file right after the last complete line, then fsync so the repair
            # itself is durable before any new record is appended.
            f.truncate(keep)
            f.flush()
            os.fsync(f.fileno())
            return len(data) - keep

    def append(self, record: dict) -> None:
        """Durably add one record. When this returns, the record is on disk (if fsync=True)."""
        # Compact separators keep lines short; json.dumps escapes any "\n" inside the text,
        # so one record is always exactly one physical line.
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        # flush: Python buffer -> OS. fsync: OS cache -> disk. Both are needed (see GLOSSARY).
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())  # WHY: durability across power loss, not just process crash

    def replay(self) -> Iterator[dict]:
        """Yield every complete record. A torn (partial) final line is skipped."""
        if not self.path.exists():
            return
        # A separate read-only handle, so replay never disturbs the append handle.
        # Iterating a text file yields one line (one record) at a time; the whole log is
        # never loaded into memory at once.
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
