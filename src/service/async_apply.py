"""Async apply mode: WAL -> ack -> background apply (the alternative to ADR 0007's sync apply).

WHAT
    An optional second write path for ``ShardStore`` (``apply_mode="async"``). The default sync
    path stays untouched; this module only runs when a shard is started with APPLY_MODE=async.

        sync  (store._write)  : check -> embed -> lock -> WAL+fsync -> apply -> ack     (ack = durable + searchable)
        async (this module)   : check -> WAL+fsync -> enqueue -> ack ... applier thread: embed BATCH -> lock -> apply
                                                                                           (ack = durable; searchable soon)

WHY HAVE BOTH
    To compare them under heavy write load (scripts/write_load.py, ADR 0013). Async makes the ack
    cheap (no embedding on the client's path) and lets the applier embed many docs in ONE batched
    model call, which is far more efficient than one call per doc. The price: a write is not
    searchable the instant it is acked, so the freshness guarantee becomes "within N seconds".

HOW THE 1-SECOND FRESHNESS TARGET IS KEPT: BACK-PRESSURE
    If writers outpace the applier, the queue would grow forever and freshness would degrade without
    bound. So before accepting a write, ``submit`` WAITS while (a) the oldest not-yet-applied write is
    older than ``max_lag_s`` (default 0.5 s), or (b) the PREDICTED wait for this write (batches
    queued ahead of it x the measured seconds per batch) exceeds ``max_lag_s``, or (c) the queue
    holds ``max_pending`` items. The applier takes at most ``batch_max`` writes per batch, so one
    batch is short. Together: visible-lag <= ~max_lag_s + one batch < 1 s. Writers slow down instead
    of freshness falling behind; that is the trade every async ingestion system has to make.

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
import threading
import time

from src.util import percentile

log = logging.getLogger(__name__)


class AsyncApplier:
    """Queue + background thread that applies acked writes to a ``ShardStore`` in batches."""

    def __init__(self, store, max_lag_s: float = 0.5, batch_max: int = 32, max_pending: int = 2000):
        self.store = store
        self.max_lag_s, self.batch_max, self.max_pending = max_lag_s, batch_max, max_pending
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
        # Smoothed time per applied batch (seconds); None until the first batch finishes.
        self._batch_s: float | None = None
        # Recent visible-lag samples in ms (enqueue -> applied), for health() and the load test.
        self.lags_ms: collections.deque = collections.deque(maxlen=20000)
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

    def _too_far_behind(self, now: float) -> bool:
        """True if accepting one more write would likely make it visible later than ``max_lag_s``.

        Two signals:
          * the oldest pending write is already older than max_lag_s (the applier is behind), or
          * the PREDICTED wait is too long: batches ahead of this write x the measured time per batch.
        PITFALL (found by the back-pressure test): the age check alone misses a fast BURST. 100 writes
        arriving in 5 ms are all "young" when accepted, yet the last one waits for every batch before
        it. Predicting the drain time catches that before the ack.
        """
        idle = not self._queue and self._inflight_oldest is None
        if idle:
            return False  # always admit when nothing is pending, or a slow model could block forever
        if len(self._queue) >= self.max_pending or self._oldest_age(now) > self.max_lag_s:
            return True
        batches_ahead = -(-(len(self._queue) + 1) // self.batch_max) + (self._inflight_oldest is not None)
        return batches_ahead * (self._batch_s or 0.05) > self.max_lag_s

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
            t_batch = time.perf_counter()
            try:
                self._apply_batch(batch)
            except Exception:  # crash-only, like sync: the WAL has every acked write
                log.exception("async applier failed; shard marked unhealthy (restart replays the WAL)")
                self.store.healthy = False
                with self._cond:
                    self._inflight_oldest = None
                    self._stop = True
                    self._cond.notify_all()
                return
            with self._cond:
                # Smoothed seconds per batch (exponential moving average), used by _too_far_behind
                # to predict how long a newly accepted write would wait.
                dt = time.perf_counter() - t_batch
                self._batch_s = dt if self._batch_s is None else 0.7 * self._batch_s + 0.3 * dt
                self._inflight_oldest = None
                self._cond.notify_all()  # wakes writers waiting on back-pressure and flush()

    def _apply_batch(self, batch: list) -> None:
        store = self.store
        # 1) Embed every upsert in the batch with every maintained model: ONE batched call per
        #    model, OUTSIDE the lock (searches keep running). This batching is async's big win.
        ups = [i for i, it in enumerate(batch) if it[1] == "upsert"]
        row_of = {i: r for r, i in enumerate(ups)}
        vecs = {m: store._embed(m, [batch[i][4] for i in ups]) for m in list(store.dense)} if ups else {}
        # 2) Apply in queue order under ONE short write lock for the whole batch, so a reader sees
        #    BM25 and dense change together (same consistency rule as sync mode).
        applied = superseded = 0
        with store.lock.write():
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
        self.lags_ms.extend((now - it[0]) * 1e3 for it in batch)
        self.counters["applied"] += applied
        self.counters["superseded"] += superseded
        self.counters["batches"] += 1
        store.counters["applied"] += applied

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
        return {"pending": pending, "oldest_pending_age_ms": round(age * 1e3, 1),
                "lag_ms": {"p50": round(percentile(lags, 50), 1), "p95": round(percentile(lags, 95), 1),
                           "max": round(max(lags), 1)} if lags else None,
                "max_lag_s": self.max_lag_s, "batch_max": self.batch_max, **self.counters}
