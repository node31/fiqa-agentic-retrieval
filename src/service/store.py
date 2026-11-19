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

EmbedFn = Callable[[str, Sequence[str]], np.ndarray]


def _default_embed(model: str, texts: Sequence[str]) -> np.ndarray:
    return get_embedder(model).encode_docs(texts)


class ShardUnhealthy(RuntimeError):
    """Raised when the shard is in a failed state and must be restarted."""


class ShardStore:
    def __init__(self, shard_id: int = 0, n_shards: int = 1, state_dir: Path | None = None,
                 embed: EmbedFn | None = None, use_cache: bool = True):
        self.shard_id, self.n_shards = shard_id, n_shards
        self.docs: dict[str, str] = {}
        self.versions: dict[str, tuple[int, bool]] = {}
        self.bm25 = BM25Index()
        self.dense: dict[str, DenseIndex] = {}
        self.ready: dict[str, bool] = {}
        self.progress: dict[str, float] = {}
        self.lock = RWLock()
        self.healthy = True
        self.counters = {"applied": 0, "noop": 0, "replayed": 0}
        self.state_dir = Path(state_dir) if state_dir else None
        self.wal = WAL(self.state_dir / "wal.jsonl") if self.state_dir else None
        self._embed = embed or _default_embed
        self._use_cache = use_cache

    # ------------------------------------------------------------------ loading
    def load(self, corpus: dict[str, str], models: Sequence[str]) -> "ShardStore":
        """Index this shard's slice of the base corpus, then replay the WAL."""
        models = self._persisted_models() or list(models)
        self.docs = dict(corpus)
        self.versions = {d: (0, False) for d in corpus}  # base corpus = version 0
        self.bm25.build(list(corpus), list(corpus.values()))
        for m in models:
            idx = DenseIndex(MODELS[m].dim)
            items = list(corpus.items())
            for i in range(0, len(items), 4096):
                chunk = items[i:i + 4096]
                idx.add([d for d, _ in chunk], self._vectors_for(m, chunk))
            self.dense[m], self.ready[m], self.progress[m] = idx, True, 1.0
        self._persist_models()
        if self.wal:
            for rec in self.wal.replay():
                if self._is_newer(rec["doc_id"], rec["version"]):
                    vecs = {m: decode_vec(s) for m, s in rec.get("vecs", {}).items()}
                    self._apply(rec["op"], rec["doc_id"], rec["version"], rec.get("text"), vecs)
                    self.counters["replayed"] += 1
        return self

    def _vectors_for(self, model: str, items: Sequence[tuple[str, str]]) -> np.ndarray:
        """Vectors for (doc_id, text) pairs: reuse the offline cache when the text is unchanged."""
        out = np.zeros((len(items), MODELS[model].dim), dtype=np.float32)
        cache = _cache_for(model) if self._use_cache else None
        todo = list(range(len(items)))
        if cache is not None:
            ids, hashes, mat = cache
            row = _row_index(model, ids)
            todo = []
            for j, (d, t) in enumerate(items):
                r = row.get(d)
                if r is not None and hashes[r] == text_hash(t):
                    out[j] = mat[r]
                else:
                    todo.append(j)
        if todo:
            out[todo] = self._embed(model, [items[j][1] for j in todo])
        return out

    def _persisted_models(self) -> list[str]:
        p = self.state_dir / "models.json" if self.state_dir else None
        return json.loads(p.read_text()) if p and p.exists() else []

    def _persist_models(self) -> None:
        if self.state_dir:
            (self.state_dir / "models.json").write_text(json.dumps(sorted(self.dense)))

    # ------------------------------------------------------------------ reads
    def stats(self, terms: Sequence[str]) -> dict:
        self._check()
        with self.lock.read():
            return self.bm25.local_stats(terms).to_dict()

    def search(self, terms: Sequence[str], k: int, stats: dict | None = None,
               vector: Sequence[float] | None = None, model: str | None = None) -> dict:
        """Return this shard's top-k for BM25 (if terms) and dense (if vector)."""
        self._check()
        with self.lock.read():
            bm = self.bm25.search(terms, k, CorpusStats.from_dict(stats) if stats else None) if terms else []
            dn = []
            if vector is not None:
                if model not in self.dense:
                    raise KeyError(f"shard {self.shard_id} has no index for model {model!r}")
                dn = self.dense[model].search(np.asarray(vector, dtype=np.float32), k)
        return {"bm25": bm, "dense": dn, "shard": self.shard_id}

    def get_docs(self, ids: Sequence[str]) -> dict[str, str]:
        with self.lock.read():
            return {d: self.docs[d] for d in ids if d in self.docs}

    # ------------------------------------------------------------------ writes
    def _is_newer(self, doc_id: str, version: int) -> bool:
        cur = self.versions.get(doc_id)
        return cur is None or version > cur[0]

    def upsert(self, doc_id: str, version: int, text: str) -> dict:
        return self._write("upsert", doc_id, version, text)

    def delete(self, doc_id: str, version: int) -> dict:
        return self._write("delete", doc_id, version, None)

    def _write(self, op: str, doc_id: str, version: int, text: str | None) -> dict:
        self._check()
        if not self._is_newer(doc_id, version):  # step 1: cheap idempotency check
            self.counters["noop"] += 1
            return {"applied": False, "reason": "stale_or_duplicate", "current_version": self.versions[doc_id][0]}
        models = list(self.dense)
        vecs = {m: self._embed(m, [text])[0] for m in models} if op == "upsert" else {}  # step 2
        with self.lock.write():  # step 3
            if not self._is_newer(doc_id, version):  # WHY re-check: a concurrent writer may have won
                self.counters["noop"] += 1
                return {"applied": False, "reason": "stale_or_duplicate", "current_version": self.versions[doc_id][0]}
            for m in self.dense:  # a migration may have started between steps 2 and 3
                if op == "upsert" and m not in vecs:
                    vecs[m] = self._embed(m, [text])[0]
            if self.wal:
                self.wal.append({"op": op, "doc_id": doc_id, "version": version, "text": text,
                                 "vecs": {m: encode_vec(v) for m, v in vecs.items()}})
            self._apply(op, doc_id, version, text, vecs)
            self.counters["applied"] += 1
        return {"applied": True, "doc_id": doc_id, "version": version, "shard": self.shard_id}

    def _apply(self, op: str, doc_id: str, version: int, text: str | None, vecs: dict) -> None:
        """Mutate every index. Caller holds the write lock (or is single-threaded at load)."""
        try:
            if op == "upsert":
                self.docs[doc_id] = text
                self.versions[doc_id] = (version, False)
                self.bm25.add(doc_id, text)
                for m, idx in self.dense.items():
                    v = vecs.get(m)
                    if v is None:  # e.g. WAL record written before this model existed
                        v = self._embed(m, [text])[0]
                    idx.add([doc_id], np.asarray(v, dtype=np.float32)[None, :])
            elif op == "delete":
                self.docs.pop(doc_id, None)
                self.versions[doc_id] = (version, True)  # tombstone blocks older upserts
                self.bm25.delete(doc_id)
                for idx in self.dense.values():
                    idx.remove(doc_id)
            else:
                raise ValueError(op)
        except Exception:
            self.healthy = False  # crash-only: refuse traffic until restarted
            raise

    def _check(self) -> None:
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
        with self.lock.write():
            if model in self.dense:
                return {"status": "exists", "ready": self.ready[model]}
            self.dense[model] = DenseIndex(MODELS[model].dim)
            self.ready[model], self.progress[model] = False, 0.0
            snapshot = [(d, self.versions[d][0], t) for d, t in self.docs.items()]
        self._persist_models()
        threading.Thread(target=self._backfill, args=(model, snapshot, batch, reencode, throttle_s),
                         daemon=True).start()
        return {"status": "started", "n_docs": len(snapshot)}

    def _backfill(self, model: str, snapshot: list, batch: int, reencode: bool = False,
                  throttle_s: float = 0.0) -> None:
        import time

        for i in range(0, len(snapshot), batch):
            chunk = snapshot[i:i + batch]
            items = [(d, t) for d, _, t in chunk]
            vecs = self._embed(model, [t for _, t in items]) if reencode else self._vectors_for(model, items)
            with self.lock.write():
                idx = self.dense.get(model)
                if idx is None:
                    return  # model dropped mid-build
                # PITFALL (race): only insert if the doc is unchanged since the snapshot
                # and the dual-write has not already inserted a newer vector.
                keep = [j for j, (d, v, _) in enumerate(chunk)
                        if d not in idx and self.versions.get(d) == (v, False)]
                if keep:
                    idx.add([chunk[j][0] for j in keep], vecs[keep])
            self.progress[model] = (i + len(chunk)) / max(len(snapshot), 1)
            if throttle_s:
                time.sleep(throttle_s)
        self.progress[model], self.ready[model] = 1.0, True

    def drop_model(self, model: str) -> dict:
        with self.lock.write():
            existed = self.dense.pop(model, None) is not None
            self.ready.pop(model, None)
            self.progress.pop(model, None)
        self._persist_models()
        return {"dropped": existed}

    def health(self) -> dict:
        from src.util import rss_mb

        return {
            "shard_id": self.shard_id, "healthy": self.healthy, "n_docs": len(self.docs),
            "models": {m: {"ready": self.ready.get(m, False), "progress": round(self.progress.get(m, 0), 3),
                           "n": len(idx)} for m, idx in self.dense.items()},
            "counters": dict(self.counters), **rss_mb(),
        }


_ROW_CACHE: dict[str, dict[str, int]] = {}
_EMB_CACHE: dict[str, object] = {}


def _cache_for(model: str):
    """load_cache() once per process (parsing the 57K-id metadata file is not free)."""
    if model not in _EMB_CACHE:
        _EMB_CACHE[model] = load_cache(model)
    return _EMB_CACHE[model]


def _row_index(model: str, ids: list[str]) -> dict[str, int]:
    if model not in _ROW_CACHE:
        _ROW_CACHE[model] = {d: i for i, d in enumerate(ids)}
    return _ROW_CACHE[model]
