"""ShardStore: one shard's documents, versions, BM25 index and dense indexes.

WHAT
    The transport-agnostic core of a shard. shard_server.py wraps it in HTTP.
    Tests and the single-node baseline call it directly, in-process. Same code,
    so "single node vs sharded" compares partitioning only, not two different
    implementations.

STATE
    docs      {doc_id: text}                   the passages this shard owns
    versions  {doc_id: (version, deleted)}     includes TOMBSTONES for deletes
    bm25      BM25Index                        lexical index
    dense     {model_key: DenseIndex}          one exact index per embedding model
    ready     {model_key: bool}                False while a migration backfill runs

WRITE PATH (upsert/delete), docs/09_live_updates_consistency.md
    1. Fast idempotency check: if stored version >= incoming, it is a no-op.
    2. Compute embeddings for every model this shard maintains, OUTSIDE the lock.
       This is slow work (~10 ms), so readers are not blocked during it.
    3. Take the exclusive lock, re-check the version (another writer may have won),
       append to the WAL (durable), then apply to docs + versions + BM25 + all
       dense indexes, and release the lock.
    4. Ack. The write is now searchable, so freshness lag = 0 after the ack.

    That is the DEFAULT, "sync" apply mode (ADR 0007). ``apply_mode="async"`` (ADR 0013,
    src/service/async_apply.py) is the alternative kept for comparison: version check -> WAL +
    fsync -> ack, and a background thread embeds in batches and applies; searchable within
    ~1 s (enforced by back-pressure). scripts/write_load.py compares both under heavy writes.

    Readers hold the shared lock, so they see the state before or after a
    write, never in between. That is what keeps lexical and dense consistent.

CRASH-ONLY RECOVERY
    If applying throws halfway (it should not, since everything fallible happens
    in step 2), the shard marks itself unhealthy. The router then treats it as
    down (partial results), and a restart rebuilds state from base corpus +
    WAL replay. Restart is the ONE recovery path, so it is the tested path.

MIGRATION (Part B3), docs/10_model_migration.md
    ``start_build(model)`` adds an empty index for the new model. From that
    instant every write is DUAL-WRITTEN to both models. A background thread
    then backfills existing docs, skipping any doc whose version changed since
    the snapshot (the newer vector is already there from the dual-write).

HOW TO READ THIS FILE (top to bottom)
    1. ``__init__``  : empty state + the WAL file handle (if a state_dir is given).
    2. ``load``      : build indexes from the base corpus, then replay the WAL.
                       This is ALSO the restart path, so there is no separate "recover" code.
    3. reads         : ``stats`` / ``search`` / ``get_docs`` take the SHARED (read) lock.
    4. writes        : ``upsert`` / ``delete`` -> ``_write`` (checks, WAL, lock) -> ``_apply`` (mutates).
    5. migration     : ``start_build`` -> background ``_backfill``; ``drop_model``.
    6. module-level caches for the offline embedding matrix (bottom of the file).

GLOSSARY (terms used throughout)
    WAL          write-ahead log: an append-only file where every write is recorded
                 BEFORE memory changes, so it can be re-applied after a crash (wal.py).
    idempotent   applying the same write twice gives the same result as applying it once.
                 Here: a retry with the same version is a no-op, so clients can retry safely.
    tombstone    a marker that says "this doc was deleted at version v". We keep it in
                 ``versions`` as (v, True) even though the text is gone, so a LATE upsert
                 with an older version cannot bring the deleted doc back to life.
    RWLock       readers-writer lock: many readers at once OR exactly one writer (wal.py).
    dual-write   during a migration, every new write goes to the old AND the new model's index.
    backfill     filling the new model's index with the docs that existed before it was created.

LIFE OF ONE UPSERT  (doc "42", version 7, text "...")
    router -> shard_server -> ShardStore.upsert -> _write:
        versions["42"] is (5, False)  -> 7 > 5, so it is newer; continue
        embed text with every model   -> {"minilm": vec, "bge-small": vec}   (no lock held)
        take write lock, re-check version, WAL.append(record) + fsync      (durable now)
        _apply: docs / versions / BM25 / every dense index updated         (visible now)
        release lock, return {"applied": True, ...}                         (the "ack")
    The same upsert retried later: 7 > 7 is False -> {"applied": False, "reason": "stale_or_duplicate"}.

WHAT HAPPENS ON RESTART
    A new process calls ``load(corpus, models)`` again: base corpus at version 0, then
    every WAL record is re-applied in order, each one passing through the same
    "is it newer?" check. Because of that check, replaying records that were already
    applied (or are superseded by later records) is harmless.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.config import MODELS
from src.index.build_index import load_cache, text_hash
from src.retrieval.bm25 import BM25Index, CorpusStats
from src.retrieval.dense import DenseIndex, get_embedder
from src.service.wal import WAL, RWLock, decode_vec, encode_vec

# Signature of an embedding function: (model_key, texts) -> float32 matrix of shape (len(texts), dim).
# WHY injectable: tests pass a fast fake embedder so they never load a real neural model.
EmbedFn = Callable[[str, Sequence[str]], np.ndarray]


def _default_embed(model: str, texts: Sequence[str]) -> np.ndarray:
    return get_embedder(model).encode_docs(texts)


class ShardUnhealthy(RuntimeError):
    """Raised when the shard is in a failed state and must be restarted."""


class ShardStore:
    def __init__(self, shard_id: int = 0, n_shards: int = 1, state_dir: Path | None = None,
                 embed: EmbedFn | None = None, use_cache: bool = True, apply_mode: str = "sync",
                 max_lag_s: float = 0.5, batch_max: int = 32, max_pending: int = 2000):
        if apply_mode not in ("sync", "async"):
            raise ValueError(f"apply_mode must be 'sync' or 'async', not {apply_mode!r}")
        self.shard_id, self.n_shards = shard_id, n_shards
        self.docs: dict[str, str] = {}
        # versions layout: {"42": (7, False), "99": (3, True)}
        #   (version, deleted). deleted=True is a TOMBSTONE: "99" is gone from docs and every
        #   index, but we remember version 3 so an older upsert (version <= 3) is rejected.
        self.versions: dict[str, tuple[int, bool]] = {}
        self.bm25 = BM25Index()
        # One DenseIndex per embedding model key, e.g. {"minilm": ..., "bge-small": ...}.
        self.dense: dict[str, DenseIndex] = {}
        # ready[m] is False while model m's index is still being backfilled (partial results);
        # progress[m] is the backfill fraction 0.0..1.0. Both are reported by health().
        self.ready: dict[str, bool] = {}
        self.progress: dict[str, float] = {}
        # One lock guards ALL of docs/versions/bm25/dense, so they change together atomically.
        self.lock = RWLock()
        self.healthy = True
        # applied = writes that changed state, noop = stale/duplicate writes ignored,
        # replayed = WAL records re-applied at startup. Exposed for tests and health checks.
        self.counters = {"applied": 0, "noop": 0, "replayed": 0}
        # No state_dir -> purely in-memory shard (tests, single-node baseline): no WAL, no
        # models.json, and nothing survives a restart.
        self.state_dir = Path(state_dir) if state_dir else None
        self.wal = WAL(self.state_dir / "wal.jsonl") if self.state_dir else None
        self._embed = embed or _default_embed
        # use_cache=False: never touch artifacts/emb_*.npy, always call the embedder
        # (toy corpora in tests/examples, where loading the 57K-row FiQA cache is wasted work).
        self._use_cache = use_cache
        # apply_mode: "sync" (default, ADR 0007) applies before the ack; "async" (ADR 0013) acks
        # after the WAL and applies in a background thread (src/service/async_apply.py). The
        # applier is created at the end of load(), after WAL replay, so replay stays single-threaded.
        self.apply_mode = apply_mode
        self._async_cfg = {"max_lag_s": max_lag_s, "batch_max": batch_max, "max_pending": max_pending}
        self._applier = None

    # ------------------------------------------------------------------ loading
    def load(self, corpus: dict[str, str], models: Sequence[str]) -> "ShardStore":
        """Index this shard's slice of the base corpus, then replay the WAL.

        This runs both on first start and on every restart: state is always rebuilt as
        "base corpus (version 0) + every logged write since", never loaded from a snapshot.
        No lock is taken because nothing else can touch the store until load() returns.
        """
        # WHY prefer the persisted list: if a migration added a model before the crash,
        # the restarted shard must keep serving it, even if the caller passes the old list.
        models = self._persisted_models() or list(models)
        self.docs = dict(corpus)
        self.versions = {d: (0, False) for d in corpus}  # base corpus = version 0
        self.bm25.build(list(corpus), list(corpus.values()))
        for m in models:
            idx = DenseIndex(MODELS[m].dim)
            items = list(corpus.items())
            # Add in chunks of 4096 docs so the temporary (chunk, dim) vector matrix stays small
            # (if the cache misses, each chunk is also one batched call to the embedder).
            for i in range(0, len(items), 4096):
                chunk = items[i:i + 4096]
                idx.add([d for d, _ in chunk], self._vectors_for(m, chunk))
            # Built synchronously from the whole corpus, so the index is complete: ready.
            self.dense[m], self.ready[m], self.progress[m] = idx, True, 1.0
        self._persist_models()
        if self.wal:
            # Replay every complete WAL record in the order it was written (oldest first).
            # Record shape (see _write): {"op": "upsert"|"delete", "doc_id", "version",
            #                             "text" (None for delete), "vecs": {model: base64}}
            for rec in self.wal.replay():
                # Same "is it newer?" rule as a live write: duplicates and superseded
                # records are skipped, which is what makes replay idempotent.
                if self._is_newer(rec["doc_id"], rec["version"]):
                    # WHY vectors are stored in the WAL: replay does not have to re-run the
                    # embedding model (slow), and it reproduces the exact acked vectors.
                    vecs = {m: decode_vec(s) for m, s in rec.get("vecs", {}).items()}
                    self._apply(rec["op"], rec["doc_id"], rec["version"], rec.get("text"), vecs)
                    self.counters["replayed"] += 1
        if self.apply_mode == "async" and self._applier is None:
            from src.service.async_apply import AsyncApplier

            self._applier = AsyncApplier(self, **self._async_cfg).start()
        return self

    def _vectors_for(self, model: str, items: Sequence[tuple[str, str]]) -> np.ndarray:
        """Vectors for (doc_id, text) pairs: reuse the offline cache when the text is unchanged."""
        # out[j] will hold the vector for items[j]; rows are filled from the cache or the embedder.
        out = np.zeros((len(items), MODELS[model].dim), dtype=np.float32)
        cache = _cache_for(model) if self._use_cache else None
        # todo = positions in `items` that still need encoding (all of them if there is no cache).
        todo = list(range(len(items)))
        if cache is not None:
            # ids[r] / hashes[r] describe row r of the cached matrix `mat` (see build_index.py).
            ids, hashes, mat = cache
            row = _row_index(model, ids)
            todo = []
            for j, (d, t) in enumerate(items):
                r = row.get(d)
                # WHY compare text hashes: a doc that was edited since the cache was built
                # has the same id but different text, so its cached vector would be wrong.
                if r is not None and hashes[r] == text_hash(t):
                    out[j] = mat[r]  # reading one row of the mmap'd matrix pulls just that row from disk
                else:
                    todo.append(j)
        if todo:
            # One batched embedder call for all misses (much faster than one call per doc).
            out[todo] = self._embed(model, [items[j][1] for j in todo])
        return out

    # models.json (in state_dir) is a sorted JSON list such as ["bge-small", "minilm"]: the set of
    # dense indexes this shard maintains. It is rewritten whenever a model is added or dropped,
    # so a restart rebuilds the same set of indexes the shard had before it went down.
    def _persisted_models(self) -> list[str]:
        p = self.state_dir / "models.json" if self.state_dir else None
        return json.loads(p.read_text()) if p and p.exists() else []

    def _persist_models(self) -> None:
        if self.state_dir:
            (self.state_dir / "models.json").write_text(json.dumps(sorted(self.dense)))

    # ------------------------------------------------------------------ reads
    # Every read holds the SHARED lock: many reads run in parallel, but none can overlap a
    # write, so a reader never sees a doc in BM25 whose dense vector is not there yet.
    def stats(self, terms: Sequence[str]) -> dict:
        """This shard's BM25 statistics (doc count, total length, df per term) for ``terms``.

        The router sums these over all shards to get GLOBAL statistics, then passes them
        back into ``search`` so every shard scores with the same IDF (comparable scores).
        """
        self._check()
        with self.lock.read():
            return self.bm25.local_stats(terms).to_dict()

    def search(self, terms: Sequence[str], k: int, stats: dict | None = None,
               vector: Sequence[float] | None = None, model: str | None = None) -> dict:
        """Return this shard's top-k for BM25 (if terms) and dense (if vector)."""
        self._check()
        with self.lock.read():
            # `stats` = global BM25 statistics from the router; None = use this shard's own
            # (only correct when there is a single shard).
            bm = self.bm25.search(terms, k, CorpusStats.from_dict(stats) if stats else None) if terms else []
            dn = []
            if vector is not None:
                # The caller embedded the query with `model`; searching another model's index
                # would compare vectors from different spaces, so fail loudly instead.
                # NOTE: an index that exists but is not ready yet (backfill running) is still
                # searched here; it just returns results from the docs added so far.
                if model not in self.dense:
                    raise KeyError(f"shard {self.shard_id} has no index for model {model!r}")
                dn = self.dense[model].search(np.asarray(vector, dtype=np.float32), k)
        return {"bm25": bm, "dense": dn, "shard": self.shard_id}

    def get_docs(self, ids: Sequence[str]) -> dict[str, str]:
        # Ids this shard does not own (or that were deleted) are silently left out.
        with self.lock.read():
            return {d: self.docs[d] for d in ids if d in self.docs}

    # ------------------------------------------------------------------ writes
    def _is_newer(self, doc_id: str, version: int) -> bool:
        """True if a write at ``version`` should be applied to ``doc_id``.

        Unknown doc -> yes. Otherwise only a STRICTLY greater version wins, so:
        same version = duplicate/retry (no-op), lower version = late/stale write (rejected).
        Tombstones count too: cur[0] is the delete's version, whatever cur[1] says.
        """
        cur = self.versions.get(doc_id)
        return cur is None or version > cur[0]

    # Public write API. The version comes from the CLIENT (the router just forwards it, like
    # Elasticsearch "external" versions), which is what lets the shard tell a retry (same
    # version) from a genuinely new write.
    def upsert(self, doc_id: str, version: int, text: str) -> dict:
        if self._applier:  # async mode: WAL -> ack, applied later (async_apply.py)
            return self._applier.submit("upsert", doc_id, version, text)
        return self._write("upsert", doc_id, version, text)

    def delete(self, doc_id: str, version: int) -> dict:
        if self._applier:
            return self._applier.submit("delete", doc_id, version, None)
        return self._write("delete", doc_id, version, None)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every acked write is searchable. Always True at once in sync mode."""
        return self._applier.flush(timeout) if self._applier else True

    def close(self) -> None:
        """Stop the async applier (after draining it) and close the WAL file."""
        if self._applier:
            self._applier.stop()
        if self.wal:
            self.wal.close()

    def _write(self, op: str, doc_id: str, version: int, text: str | None) -> dict:
        """The write path from the module docstring (steps 1-4). Returns the ack dict."""
        self._check()
        # Step 1 runs WITHOUT the lock: a cheap early exit for retries, so a duplicate never
        # pays for embedding. It may race with another writer; step 3 re-checks under the lock.
        if not self._is_newer(doc_id, version):  # step 1: cheap idempotency check
            self.counters["noop"] += 1
            return {"applied": False, "reason": "stale_or_duplicate", "current_version": self.versions[doc_id][0]}
        # Step 2: embed once per maintained model ({model_key: 1-D float32 vector}).
        # Deletes need no vectors. No lock is held, so searches keep running meanwhile.
        models = list(self.dense)
        vecs = {m: self._embed(m, [text])[0] for m in models} if op == "upsert" else {}  # step 2
        with self.lock.write():  # step 3
            if not self._is_newer(doc_id, version):  # WHY re-check: a concurrent writer may have won
                self.counters["noop"] += 1
                return {"applied": False, "reason": "stale_or_duplicate", "current_version": self.versions[doc_id][0]}
            # PITFALL: this rare catch-up embeds while HOLDING the write lock (slow); it is
            # accepted because it only happens if start_build() ran in the gap.
            for m in self.dense:  # a migration may have started between steps 2 and 3
                if op == "upsert" and m not in vecs:
                    vecs[m] = self._embed(m, [text])[0]
            # WAL FIRST, then memory. wal.append() returns only after fsync, so from here on
            # the write survives a crash even if _apply below never runs (replay redoes it).
            # Vectors are base64 strings because JSON has no binary type (see wal.encode_vec).
            if self.wal:
                self.wal.append({"op": op, "doc_id": doc_id, "version": version, "text": text,
                                 "vecs": {m: encode_vec(v) for m, v in vecs.items()}})
            self._apply(op, doc_id, version, text, vecs)
            self.counters["applied"] += 1
        # Step 4: the lock is released, so the write is already visible to the next search.
        return {"applied": True, "doc_id": doc_id, "version": version, "shard": self.shard_id}

    def _apply(self, op: str, doc_id: str, version: int, text: str | None, vecs: dict) -> None:
        """Mutate every index. Caller holds the write lock (or is single-threaded at load)."""
        try:
            if op == "upsert":
                # Insert-or-replace in every structure. BM25Index.add and DenseIndex.add both
                # remove an existing entry for doc_id first, so an upsert never leaves duplicates.
                self.docs[doc_id] = text
                self.versions[doc_id] = (version, False)
                self.bm25.add(doc_id, text)
                for m, idx in self.dense.items():
                    v = vecs.get(m)
                    if v is None:  # e.g. WAL record written before this model existed
                        v = self._embed(m, [text])[0]
                    # [None, :] turns the 1-D vector (dim,) into a (1, dim) batch, as add() expects.
                    idx.add([doc_id], np.asarray(v, dtype=np.float32)[None, :])
            elif op == "delete":
                # Remove the content everywhere, but KEEP the version entry (as a tombstone).
                # BM25 only marks the row dead (its own tombstone); faiss removes the vector.
                self.docs.pop(doc_id, None)
                self.versions[doc_id] = (version, True)  # tombstone blocks older upserts
                self.bm25.delete(doc_id)
                for idx in self.dense.values():
                    idx.remove(doc_id)
            else:
                raise ValueError(op)
        except Exception:
            # WHY give up instead of trying to undo: a half-applied write left the indexes
            # inconsistent with each other, and undo code is rarely tested. The write is
            # already in the WAL (if this shard has one), so a restart rebuilds a correct state.
            self.healthy = False  # crash-only: refuse traffic until restarted
            raise

    def _check(self) -> None:
        # Called first by stats/search/writes (not get_docs/health): an unhealthy shard fails fast,
        # and the router treats the error like a down shard (partial results) until a restart.
        if not self.healthy:
            raise ShardUnhealthy(f"shard {self.shard_id} is unhealthy; restart to recover from WAL")

    # ------------------------------------------------------------------ migration
    def start_build(self, model: str, batch: int = 256, reencode: bool = False, throttle_s: float = 0.0) -> dict:
        """Create an index for ``model``, start dual-writing to it, and backfill in the background.

        Args:
            reencode: ignore the offline embedding cache, i.e. what a REAL model change costs.
            throttle_s: sleep between batches. Rate-limiting the backfill protects serving
                latency, because the backfill competes with queries for CPU.
        """
        # Under the write lock, atomically: (a) register the empty index, so every write from
        # now on is DUAL-WRITTEN to it (_write embeds for every model in self.dense), and
        # (b) take a snapshot of what exists right now. Every doc is then covered by exactly one
        # of the two: older docs by the backfill, newer writes by the dual-write.
        with self.lock.write():
            if model in self.dense:
                return {"status": "exists", "ready": self.ready[model]}
            self.dense[model] = DenseIndex(MODELS[model].dim)
            self.ready[model], self.progress[model] = False, 0.0
            # snapshot = [(doc_id, version_at_snapshot, text), ...]; tombstoned docs are not in
            # self.docs, so they are naturally excluded.
            snapshot = [(d, self.versions[d][0], t) for d, t in self.docs.items()]
        # WHY persist now: if the shard restarts mid-backfill, load() sees the model in
        # models.json and simply builds its index in full at startup.
        self._persist_models()
        # daemon=True: the thread does not keep the process alive at shutdown; an unfinished
        # backfill is not lost work, because the restart path above rebuilds the index anyway.
        threading.Thread(target=self._backfill, args=(model, snapshot, batch, reencode, throttle_s),
                         daemon=True).start()
        return {"status": "started", "n_docs": len(snapshot)}

    def _backfill(self, model: str, snapshot: list, batch: int, reencode: bool = False,
                  throttle_s: float = 0.0) -> None:
        """Background thread: add the snapshot's docs to ``model``'s index, one batch at a time."""
        import time

        for i in range(0, len(snapshot), batch):
            chunk = snapshot[i:i + batch]
            items = [(d, t) for d, _, t in chunk]
            # Slow part (cache lookup or encoding) happens OUTSIDE the lock, like in _write.
            vecs = self._embed(model, [t for _, t in items]) if reencode else self._vectors_for(model, items)
            # Short write lock per batch: searches interleave between batches instead of
            # waiting for the whole backfill.
            with self.lock.write():
                idx = self.dense.get(model)
                if idx is None:
                    return  # model dropped mid-build
                # PITFALL (race): only insert if the doc is unchanged since the snapshot
                # and the dual-write has not already inserted a newer vector.
                # `self.versions.get(d) == (v, False)` fails if the doc was updated (newer v) or
                # deleted (tombstone) since the snapshot; `d not in idx` fails if a dual-write
                # already put a vector there. Either way the stale snapshot vector is dropped.
                keep = [j for j, (d, v, _) in enumerate(chunk)
                        if d not in idx and self.versions.get(d) == (v, False)]
                if keep:
                    idx.add([chunk[j][0] for j in keep], vecs[keep])
            # Progress = fraction of the snapshot processed (skipped docs count as done).
            self.progress[model] = (i + len(chunk)) / max(len(snapshot), 1)
            if throttle_s:
                time.sleep(throttle_s)
        # Every snapshot doc is now covered (backfilled or newer via dual-write): the router
        # may switch queries to this model once every shard reports ready.
        self.progress[model], self.ready[model] = 1.0, True

    def drop_model(self, model: str) -> dict:
        """Stop maintaining ``model`` (end of a migration or a rollback); frees its index."""
        with self.lock.write():
            existed = self.dense.pop(model, None) is not None
            self.ready.pop(model, None)
            self.progress.pop(model, None)
        self._persist_models()
        return {"dropped": existed}

    def health(self) -> dict:
        """Status snapshot for the router / dashboards: health flag, doc count, per-model
        readiness and backfill progress, write counters and process memory (RSS, in MB)."""
        from src.util import rss_mb

        return {
            "shard_id": self.shard_id, "healthy": self.healthy, "n_docs": len(self.docs),
            "models": {m: {"ready": self.ready.get(m, False), "progress": round(self.progress.get(m, 0), 3),
                           "n": len(idx)} for m, idx in self.dense.items()},
            "counters": dict(self.counters), "apply_mode": self.apply_mode,
            **({"async": self._applier.stats()} if self._applier else {}), **rss_mb(),
        }


# Process-wide caches of the offline embedding artifacts (build_index.py), keyed by model key.
#   _EMB_CACHE[model] = (ids, hashes, mmap'd matrix) from load_cache(), or None if no usable cache.
#   _ROW_CACHE[model] = {doc_id: row number in that matrix}, so a lookup is O(1) not O(n).
# WHY module-level (not per ShardStore): several stores in one process (tests, the single-node
# baseline, in-process shards) share one copy instead of each parsing the 57K-id file.
_ROW_CACHE: dict[str, dict[str, int]] = {}
_EMB_CACHE: dict[str, object] = {}


def _cache_for(model: str):
    """load_cache() once per process (parsing the 57K-id metadata file is not free)."""
    if model not in _EMB_CACHE:
        _EMB_CACHE[model] = load_cache(model)
    return _EMB_CACHE[model]


def _row_index(model: str, ids: list[str]) -> dict[str, int]:
    # Built once per model: {doc_id: row}. Assumes `ids` is always the same list for a model,
    # which holds because _cache_for() also loads the cache only once per process.
    if model not in _ROW_CACHE:
        _ROW_CACHE[model] = {d: i for i, d in enumerate(ids)}
    return _ROW_CACHE[model]
