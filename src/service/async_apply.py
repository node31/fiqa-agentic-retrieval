"""Async apply mode: WAL -> ack -> background apply (the alternative to ADR 0007's sync apply).

WHAT
    An optional second write path for ``ShardStore`` (``apply_mode="async"``). The default sync
    path stays untouched; this module only runs when a shard is started with APPLY_MODE=async.

        sync  (store._write)  : check -> embed -> lock -> WAL+fsync -> apply -> ack     (ack = durable + searchable)
        async (this module)   : check -> WAL+fsync -> enqueue -> ack ... applier: embed batch (parallel) -> lock -> apply
                                                                                           (ack = durable; searchable soon)

WHY HAVE BOTH
    To compare them under heavy write load (scripts/write_load.py, ADR 0013). Async makes the ack
    cheap (no embedding on the client's path: WAL + fsync only) and absorbs bursts. The price: a
    write is not searchable the instant it is acked, so the freshness guarantee becomes "within 1 s".

PITFALL: "BATCHING IS ASYNC'S BIG WIN" IS FALSE HERE (measured, MiniLM, 4 torch threads, CPU,
the same 256 random FiQA passages for every chunk size)
    docs per model call   1     2     4     8    16    32   256
    ms per doc          25.6  26.1  27.6  31.5  30.9  32.3  24.7
    A small batch is padded to its LONGEST passage (up to 256 tokens) and attention grows faster
    than linearly with length, so small batches of random-length docs cost MORE per doc. A very
    large call is sorted by length inside sentence-transformers, so padding mostly disappears, but
    it still only saves ~4%. (On a GPU batching usually wins; not measured here.) Our first version
    embedded each batch in ONE call on ONE thread, so under load it was slower than sync (whose
    concurrent requests embed in parallel) and broke the 1 s promise (p95 1.07 s, max 1.31 s).
    Hence: ``embed_workers`` threads embed ``embed_chunk`` (=1) docs per call in parallel, the same
    parallelism sync gets from concurrent requests.

HOW THE 1-SECOND FRESHNESS TARGET IS KEPT: BACK-PRESSURE
    If writers outpace the applier, the queue would grow forever and freshness would degrade without
    bound. So before accepting a write, ``submit`` WAITS while (a) the oldest not-yet-applied write is
    older than ``max_lag_s`` (default 0.5 s), or (b) the PREDICTED time until this write is applied
    exceeds ``max_lag_s``, or (c) the queue holds ``max_pending`` items. The prediction counts DOCS,
    not batches (a batch of 18 docs costs ~6x a batch of 3): remaining time of the batch in flight
    + parallel "rounds" needed for the docs queued ahead (ceil(docs / embed_workers)) x the measured
    seconds per round. The estimate rises fast and falls slowly, so a slow spell is not forgotten
    after one fast batch. Budget split: <= 0.5 s predicted, the rest of the 1 s is headroom for
    prediction error. Writers slow down instead of freshness falling behind; that is the trade
    every async ingestion system has to make.

THE FOUR THINGS ASYNC MUST GET RIGHT (that sync gets for free)
    1. Versions: "is this write newer?" must compare with the latest ACCEPTED version (``accepted``),
       not the latest APPLIED one. Otherwise v2 (queued, not applied yet) followed by a retry of v2
       would both look "newer" than the applied v1.
    2. Order: a single applier thread applies in queue order = WAL order, so replay after a crash
       reproduces exactly the same state.
    3. Nothing can fail after the ack, or the ack was a lie. ``submit`` validates the payload BEFORE
       the ack. (A real system also needs a dead-letter queue for writes that still fail to apply;
       here a failing apply marks the shard unhealthy, and a restart replays the WAL.)
    4. Crash: acked-but-unapplied writes are in the WAL (text only, no vectors, because embedding
       happens later). Restart replays them; ``ShardStore._apply`` embeds any missing vector.

HOW TO READ THIS FILE
    ``submit`` (client thread: validate, back-pressure, WAL, enqueue, ack) ->
    ``_run`` (applier thread loop) -> ``_apply_batch`` (embed outside the lock, apply under it) ->
    ``flush`` / ``stats`` (tests, health, load test).
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.util import percentile

log = logging.getLogger(__name__)


class AsyncApplier:
    """Queue + background thread that applies acked writes to a ``ShardStore`` in batches."""

    def __init__(self, store, max_lag_s: float = 0.5, batch_max: int = 32, max_pending: int = 2000,
                 embed_workers: int = 8, embed_chunk: int = 1):
        self.store = store
        self.max_lag_s, self.batch_max, self.max_pending = max_lag_s, batch_max, max_pending
        # Parallel embedding (see the PITFALL in the module docstring): up to ``embed_workers``
        # model calls at once, ``embed_chunk`` docs each. torch releases the GIL inside a forward
        # pass, so threads really run in parallel (sync mode relies on the same thing).
        self.embed_workers, self.embed_chunk = max(1, embed_workers), max(1, embed_chunk)
        self._pool = ThreadPoolExecutor(self.embed_workers, thread_name_prefix=f"embed-{store.shard_id}") \
            if self.embed_workers > 1 else None
        # Latest ACCEPTED version per doc written since start (may be ahead of store.versions,
        # which only moves when the applier applies). See point 1 in the module docstring.
        self.accepted: dict[str, int] = {}
        # Serialises "check version -> WAL append -> enqueue", so queue order == WAL order.
        self._accept_lock = threading.Lock()
        # Guards the queue; writers and the applier wait/notify on it.
        self._cond = threading.Condition()
        # Items: (t_enqueued, op, doc_id, version, text). t_enqueued ~ the ack time.
        self._queue: collections.deque = collections.deque()
        # Enqueue time of the oldest write in the batch being applied right now (None = idle).
        # The back-pressure check must count it: those writes are acked but not visible yet.
        self._inflight_oldest: float | None = None
        # When the in-flight batch started and how many parallel rounds it needs (for "how long
        # until it finishes?" in the back-pressure prediction).
        self._inflight_start = 0.0
        self._inflight_rounds = 0
        # Measured cost model (seconds), None until the first batch finishes:
        #   _round_s = time of one parallel embedding round (embed_workers calls of embed_chunk docs)
        #   _fixed_s = per-batch cost that doesn't scale with docs (lock + apply + bookkeeping)
        self._round_s: float | None = None
        self._fixed_s: float | None = None
        # Recent visible-lag samples in ms (enqueue -> applied), for health() and the load test.
        self.lags_ms: collections.deque = collections.deque(maxlen=20000)
        # Per-batch timing breakdown (size, embed s, lock-wait s, apply s): where the applier's
        # time goes. Exposed by stats() so a load test can explain a slow applier.
        self.batch_log: collections.deque = collections.deque(maxlen=5000)
        self.counters = {"accepted": 0, "applied": 0, "superseded": 0, "batches": 0, "backpressure_waits": 0}
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=f"applier-{store.shard_id}", daemon=True)

    def start(self) -> "AsyncApplier":
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Drain what is queued, then stop the thread (used by tests and ShardStore.close)."""
        self.flush(timeout)
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout)
        if self._pool:
            self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------ client side
    def latest_version(self, doc_id: str) -> int | None:
        """The highest version known for ``doc_id``: accepted (maybe queued) or applied."""
        a = self.accepted.get(doc_id)
        cur = self.store.versions.get(doc_id)
        known = [v for v in (a, cur[0] if cur else None) if v is not None]
        return max(known) if known else None

    def _is_newer(self, doc_id: str, version: int) -> bool:
        cur = self.latest_version(doc_id)
        return cur is None or version > cur

    def submit(self, op: str, doc_id: str, version: int, text: str | None) -> dict:
        """Accept one write: validate, wait for capacity, WAL + fsync, enqueue, ack."""
        self.store._check()
        # Point 3: reject bad input BEFORE the ack. After the ack it would be too late.
        if op == "upsert" and (not isinstance(text, str) or not text.strip()):
            raise ValueError("upsert needs a non-empty text")
        # Cheap early exit for retries (no waiting, no lock), like step 1 of the sync path.
        if not self._is_newer(doc_id, version):
            return self._noop(doc_id)
        self._wait_for_capacity()
        with self._accept_lock:
            # Re-check under the lock: another writer may have accepted a newer version meanwhile.
            if not self._is_newer(doc_id, version):
                return self._noop(doc_id)
            # Durable BEFORE the ack. Text only: the vectors don't exist yet (the applier makes
            # them), so a replay after a crash re-embeds (ShardStore._apply handles that).
            if self.store.wal:
                self.store.wal.append({"op": op, "doc_id": doc_id, "version": version, "text": text, "vecs": {}})
            self.accepted[doc_id] = version
            with self._cond:
                self._queue.append((time.perf_counter(), op, doc_id, version, text))
                self.counters["accepted"] += 1
                self._cond.notify_all()
        # "applied": True keeps the reply shape of sync mode for the router and clients; here it
        # means "accepted and durable". "visible": False says it is not searchable yet.
        return {"applied": True, "visible": False, "mode": "async", "doc_id": doc_id, "version": version,
                "shard": self.store.shard_id}

    def _noop(self, doc_id: str) -> dict:
        self.store.counters["noop"] += 1
        return {"applied": False, "reason": "stale_or_duplicate", "current_version": self.latest_version(doc_id)}

    def _oldest_age(self, now: float) -> float:
        """Seconds since the oldest acked-but-not-applied write was enqueued (0 if none)."""
        oldest = [t for t in (self._queue[0][0] if self._queue else None, self._inflight_oldest) if t is not None]
        return now - min(oldest) if oldest else 0.0

    def _rounds(self, n_docs: int) -> int:
        """Parallel embedding rounds needed for ``n_docs`` docs (every maintained model embeds each doc).

        Example: 18 docs, chunk 1, 1 model, 8 workers -> 18 calls -> ceil(18 / 8) = 3 rounds.
        During a migration 2 models are maintained, so the same 18 docs need 36 calls = 5 rounds.
        """
        calls = math.ceil(n_docs / self.embed_chunk) * max(1, len(self.store.dense))
        return math.ceil(calls / self.embed_workers)

    def predicted_wait(self, now: float) -> float:
        """Seconds until a write accepted NOW would be applied (searchable), from the cost model.

        = what is left of the batch in flight
        + the rounds needed for the docs queued ahead plus this one (they form the next batches)
        + the fixed per-batch cost of each of those batches.
        """
        round_s, fixed_s = self._round_s or 0.05, self._fixed_s or 0.005
        left = 0.0
        if self._inflight_oldest is not None:
            left = max(0.0, self._inflight_rounds * round_s + fixed_s - (now - self._inflight_start))
        n = len(self._queue) + 1
        n_batches = math.ceil(n / self.batch_max)
        return left + self._rounds(n) * round_s + n_batches * fixed_s

    def _too_far_behind(self, now: float) -> bool:
        """True if accepting one more write would likely make it visible later than ``max_lag_s``.

        Two signals:
          * the oldest pending write is already older than max_lag_s (the applier is behind), or
          * the PREDICTED wait (``predicted_wait``: counts docs and parallel rounds) is too long.
        PITFALL (found by the back-pressure test): the age check alone misses a fast BURST. 100 writes
        arriving in 5 ms are all "young" when accepted, yet the last one waits for every batch before
        it. Predicting the drain time catches that before the ack.
        PITFALL (found by scripts/write_load.py on the real corpus): predicting "batches ahead x
        average batch time" fails because batch cost scales with the NUMBER OF DOCS in it (3 to 18
        under load); a big batch after small ones overshot the budget (max lag 1.4 s).
        """
        idle = not self._queue and self._inflight_oldest is None
        if idle:
            return False  # always admit when nothing is pending, or a slow model could block forever
        if len(self._queue) >= self.max_pending or self._oldest_age(now) > self.max_lag_s:
            return True
        return self.predicted_wait(now) > self.max_lag_s

    def _wait_for_capacity(self) -> None:
        """Back-pressure: block while the applier is too far behind (keeps freshness < 1 s)."""
        with self._cond:
            waited = False
            while not self._stop and self._too_far_behind(time.perf_counter()):
                waited = True
                self._cond.wait(timeout=0.01)
            if waited:
                self.counters["backpressure_waits"] += 1

    # ------------------------------------------------------------------ applier side
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait(timeout=0.1)
                if self._stop and not self._queue:
                    return
                # No artificial delay to "collect" a batch: under load, writes pile up while the
                # previous batch is applied, so batches grow by themselves; when idle, a lone
                # write is applied immediately (lowest lag).
                batch = [self._queue.popleft() for _ in range(min(len(self._queue), self.batch_max))]
                self._inflight_oldest = batch[0][0]
                self._inflight_start = time.perf_counter()
                self._inflight_rounds = self._rounds(sum(1 for it in batch if it[1] == "upsert"))
            try:
                timing = self._apply_batch(batch)
            except Exception:  # crash-only, like sync: the WAL has every acked write
                log.exception("async applier failed; shard marked unhealthy (restart replays the WAL)")
                self.store.healthy = False
                with self._cond:
                    self._inflight_oldest = None
                    self._stop = True
                    self._cond.notify_all()
                return
            with self._cond:
                # Update the cost model used by predicted_wait. "Rise fast, fall slowly": a slower
                # sample moves the estimate halfway at once, a faster one only 20%, so one lucky
                # fast batch doesn't make back-pressure optimistic in the middle of a slow spell.
                rounds, embed_s, fixed_s = timing
                if rounds:
                    self._round_s = self._ewma(self._round_s, embed_s / rounds)
                self._fixed_s = self._ewma(self._fixed_s, fixed_s)
                self._inflight_oldest = None
                self._cond.notify_all()  # wakes writers waiting on back-pressure and flush()

    @staticmethod
    def _ewma(old: float | None, x: float) -> float:
        if old is None:
            return x
        return old + (0.5 if x > old else 0.2) * (x - old)

    def _embed_parallel(self, model: str, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` as ``embed_chunk``-sized model calls, up to ``embed_workers`` at a time.

        Order is preserved (results are stacked in submission order), so row i is texts[i].
        """
        chunks = [texts[i:i + self.embed_chunk] for i in range(0, len(texts), self.embed_chunk)]
        if self._pool is None or len(chunks) == 1:
            return self.store._embed(model, texts)
        return np.vstack(list(self._pool.map(lambda c: self.store._embed(model, c), chunks)))

    def _apply_batch(self, batch: list) -> tuple[int, float, float]:
        """Embed (parallel, outside the lock) then apply (in order, under one write lock).

        Returns (embedding rounds, embed seconds, fixed seconds = lock wait + apply) for the cost model.
        """
        store = self.store
        # 1) Embed every upsert in the batch with every maintained model, OUTSIDE the lock
        #    (searches keep running): one doc per model call, calls in parallel (module PITFALL).
        ups = [i for i, it in enumerate(batch) if it[1] == "upsert"]
        row_of = {i: r for r, i in enumerate(ups)}
        t0 = time.perf_counter()
        vecs = {m: self._embed_parallel(m, [batch[i][4] for i in ups]) for m in list(store.dense)} if ups else {}
        t1 = time.perf_counter()
        # 2) Apply in queue order under ONE short write lock for the whole batch, so a reader sees
        #    BM25 and dense change together (same consistency rule as sync mode).
        applied = superseded = 0
        with store.lock.write():
            t2 = time.perf_counter()
            for i, (_, op, doc_id, version, text) in enumerate(batch):
                # Same rule as sync: only a strictly newer version changes state. A queued v1 followed
                # by v2 in the same batch applies both, in order; the end state is v2.
                if not store._is_newer(doc_id, version):
                    superseded += 1
                    continue
                v = {m: vecs[m][row_of[i]] for m in vecs} if op == "upsert" else {}
                store._apply(op, doc_id, version, text, v)  # missing model (migration) -> embeds there
                applied += 1
        now = time.perf_counter()
        self.batch_log.append((len(batch), t1 - t0, t2 - t1, now - t2))
        self.lags_ms.extend((now - it[0]) * 1e3 for it in batch)
        self.counters["applied"] += applied
        self.counters["superseded"] += superseded
        self.counters["batches"] += 1
        store.counters["applied"] += applied
        rounds = self._rounds(len(ups)) if ups else 0
        return rounds, t1 - t0, now - t1

    # ------------------------------------------------------------------ observability
    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every accepted write is applied. True if drained within ``timeout``."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._queue or self._inflight_oldest is not None:
                left = deadline - time.perf_counter()
                if left <= 0 or self._stop:
                    return not (self._queue or self._inflight_oldest is not None)
                self._cond.wait(timeout=min(left, 0.05))
        return True

    def stats(self) -> dict:
        with self._cond:
            pending = len(self._queue) + (1 if self._inflight_oldest is not None else 0)
            age = self._oldest_age(time.perf_counter())
        lags = list(self.lags_ms)
        bl = list(self.batch_log)

        def ms(col: int) -> dict | None:
            xs = [b[col] * 1e3 for b in bl]
            return {"p50": round(percentile(xs, 50), 1), "p95": round(percentile(xs, 95), 1),
                    "max": round(max(xs), 1)} if xs else None

        batches = {"size_avg": round(sum(b[0] for b in bl) / len(bl), 1), "size_max": max(b[0] for b in bl),
                   "embed_ms": ms(1), "lock_wait_ms": ms(2), "apply_ms": ms(3)} if bl else None
        return {"pending": pending, "oldest_pending_age_ms": round(age * 1e3, 1),
                "lag_ms": {"p50": round(percentile(lags, 50), 1), "p95": round(percentile(lags, 95), 1),
                           "max": round(max(lags), 1)} if lags else None,
                "batch_timing": batches,
                "cost_model_ms": {"round": round((self._round_s or 0) * 1e3, 1), "fixed": round((self._fixed_s or 0) * 1e3, 1)},
                "embed_workers": self.embed_workers, "embed_chunk": self.embed_chunk,
                "max_lag_s": self.max_lag_s, "batch_max": self.batch_max, **self.counters}
