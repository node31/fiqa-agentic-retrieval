"""Router: scatter a query to every shard, gather, merge globally, fuse, return top-k.

REQUEST PATH FOR A QUERY (docs/08_sharding_scatter_gather.md has the diagram)
    0. Tokenise the query (BM25) and, IN PARALLEL, embed it once with the active
       model. Embedding at the router means N shards do not each repeat ~5 ms of work.
    1. DFS phase ("distributed frequency statistics", the name Elasticsearch uses
       for ``dfs_query_then_fetch``): ask every shard for N, total length and df of
       the query terms, then sum them into GLOBAL stats.
    2. Query phase: send terms + global stats + query vector to every shard. Each
       returns its top-``depth`` BM25 list and top-``depth`` dense list.
    3. Merge: global BM25 top-depth and global dense top-depth (scores are
       comparable because of step 1 and because of the shared model).
    4. Fuse (RRF/linear) on the GLOBAL lists. Merge, THEN fuse (ADR 0003).
    5. Optional fetch phase: get passage text for the final ids (and rerank).

FAILURE POLICY (docs/adr/0006-partial-results.md)
    Every shard call has a timeout. A slow or dead shard is reported, not hidden:
    the response carries ``partial: true`` and ``missing_shards: [...]``.
    ``failure_policy="partial"`` (default) returns what we have, because for a
    search box a slightly worse answer beats an error. ``"fail"`` raises
    instead, which suits callers that need completeness (e.g. our eval harness).

WRITE PATH
    ``shard_of(doc_id)`` picks exactly one owner shard. The router forwards the
    write and retries on timeout, which is SAFE only because writes are idempotent
    (same id + version = no-op).

MODEL MIGRATION (Part B3)
    ``active_model`` is a flag read once per request. ``cutover`` flips it only
    after every shard reports the target index ready, and ``rollback`` flips it
    back. In-flight requests keep the model they started with. Shards keep
    both indexes (dual-write), so either model can serve at any instant, and
    no query fails during a flip.

    Admin sequence: POST /admin/migrate (shards build the new index in the
    background) -> poll /admin/status until every shard says ready -> POST
    /admin/cutover -> (if quality drops) POST /admin/rollback -> once happy,
    POST /admin/finalize to drop the old index. ``active_model`` and
    ``previous_model`` are saved to ``router_state.json`` so a router restart
    does not silently revert a cutover.

LIFECYCLE OF ONE WRITE
    client POST /upsert {doc_id, version, text} -> ``Router.write`` ->
    ``shard_of`` picks owner shard i -> POST shard_i/upsert -> the shard embeds
    the text, appends the record to its WAL (write-ahead log: an append-only
    file written BEFORE the in-memory change, replayed on restart), applies it
    to BM25 + dense indexes, and only then replies. So "acked" means "durable
    AND already searchable" (docs/adr/0007-sync-apply-wal.md).

HOW TO READ THIS FILE (top to bottom)
    1. ``ShardClient``, ``dispatch``, ``LocalShardClient``, ``HttpShardClient``:
       HOW the router talks to a shard. One interface, two transports
       (in-process for tests and the single-node baseline, HTTP for the real
       cluster), so the ``Router`` logic is identical in both cases.
    2. ``Router._fan``: the scatter-gather primitive every other method uses.
    3. ``Router.search``: the query path above, phase by phase.
    4. ``Router.write``: the write path.
    5. ``Router.status/migrate/cutover/rollback/finalize``: model migration.
    6. ``create_app``: the FastAPI layer that maps HTTP routes onto ``Router``.
    A from-scratch walkthrough is in docs/from_zero/07_part_b_distributed.md.

GLOSSARY (terms used in comments below)
    shard           one process holding a slice of the corpus; doc d lives on
                    shard ``crc32(d) % N`` (``src.index.sharding.shard_of``).
    scatter-gather  send a request to many shards at once (scatter, also called
    / fan-out       fan-out), then wait for and combine their replies (gather).
                    Latency = the SLOWEST shard, so the tail grows with N: 3
                    shards measured p50/p95 ~28.6/40.5 ms vs 20.7/24.5 ms on one
                    node ("tail-latency amplification"), with identical nDCG.
    partial result  an answer built from fewer than all shards; always flagged.
    idempotent      applying the same request twice == applying it once.
    dual-write      during a migration every write is indexed under BOTH models.
    cutover         the instant the router starts querying with the new model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from src.config import (ARTIFACTS_DIR, CANDIDATE_DEPTH, DEFAULT_ALPHA, DEFAULT_FUSION, PRIMARY_MODEL,
                        SHARD_TIMEOUT_S)
from src.index.sharding import shard_of
from src.retrieval.bm25 import CorpusStats, tokenize
from src.retrieval.hybrid import fuse, merge_shard_lists
from pydantic import BaseModel

log = logging.getLogger("router")


# ---------------------------------------------------------------------- transport: how we reach a shard
class ShardClient(Protocol):
    """Anything with ``await call(op, payload, timeout) -> dict`` can act as a shard.

    ``Protocol`` = structural typing ("duck typing" a type checker understands):
    classes do NOT inherit from it, they just need a matching ``call`` method.
    ``op`` is an RPC name such as "search"; ``payload`` is its JSON-able argument dict.
    """
    async def call(self, op: str, payload: dict, timeout: float) -> dict: ...


def dispatch(store, op: str, p: dict) -> dict:
    """Map an RPC name to a ShardStore method. Shared by the HTTP server and in-process client.

    Response shapes (all plain JSON-able dicts):
        stats        {"n_docs", "total_len", "df": {term: count}}   this shard's LOCAL stats
        search       {"bm25": [[doc_id, score], ...], "dense": [[doc_id, score], ...], "shard": i}
                     each list is this shard's top-k, best first ((id, score) tuples in-process)
        docs         {doc_id: text} for the requested ids this shard owns
        upsert/delete {"applied": True, ...} or {"applied": False, "reason": "stale_or_duplicate", ...}
        build_model  {"status": ..., "ready": bool}   starts a background backfill
        health       {"shard_id", "healthy", "n_docs", "models": {m: {"ready", "progress", "n"}}, ...}
    """
    # WHY one shared function: the HTTP shard server and LocalShardClient both call it,
    # so the in-process tests exercise exactly the same op -> method mapping as production.
    if op == "stats":
        return store.stats(p["terms"])
    if op == "search":
        # stats/vector/model are optional: no stats -> shard uses its LOCAL BM25 stats,
        # no vector -> dense list comes back empty (BM25-only query).
        return store.search(p["terms"], p["k"], p.get("stats"), p.get("vector"), p.get("model"))
    if op == "docs":
        return store.get_docs(p["ids"])
    if op == "upsert":
        return store.upsert(p["doc_id"], int(p["version"]), p["text"])
    if op == "delete":
        return store.delete(p["doc_id"], int(p["version"]))
    if op == "flush":
        # Wait until every acked write is searchable (a no-op in sync mode). Load tests use it.
        return {"drained": store.flush(float(p.get("timeout", 10)))}
    if op == "build_model":
        return store.start_build(p["model"], reencode=bool(p.get("reencode")), throttle_s=float(p.get("throttle_s", 0)))
    if op == "drop_model":
        return store.drop_model(p["model"])
    if op == "health":
        return store.health()
    raise ValueError(f"unknown op {op!r}")


class LocalShardClient:
    """Calls a ShardStore in-process. Used for the single-node baseline and fast tests."""

    def __init__(self, store):
        self.store = store

    async def call(self, op: str, payload: dict, timeout: float) -> dict:
        # PITFALL: this is synchronous work inside an ``async def``. It runs to completion
        # on the event loop (no await inside), so ``timeout`` cannot interrupt it. Fine for
        # tests and the single-node baseline; it is NOT a model of a slow remote shard.
        return dispatch(self.store, op, payload)


class HttpShardClient:
    """Calls a shard server over HTTP with keep-alive connections (httpx).

    Keep-alive: the TCP connection is reused across requests instead of being opened
    and closed each time, which saves a handshake (~0.1-1 ms locally, much more across
    a network) on every shard call. That only works because ``client`` is ONE shared
    ``httpx.AsyncClient`` (a connection pool) created in ``create_app``, not one per call.
    """

    def __init__(self, url: str, client):
        self.url, self.client = url.rstrip("/"), client

    async def call(self, op: str, payload: dict, timeout: float) -> dict:
        # Every RPC is ``POST {shard_url}/{op}`` with a JSON body (see shard_server.py).
        r = await self.client.post(f"{self.url}/{op}", json=payload, timeout=timeout)
        # WHY raise on 4xx/5xx: an unhealthy shard answers 503; raising makes _fan count
        # it as missing instead of treating an error body as a valid result.
        r.raise_for_status()
        return r.json()


class ShardFailure(RuntimeError):
    """Raised when shards needed for an answer are unavailable. ``missing`` = their indexes.

    ``create_app`` turns it into HTTP 503 with ``missing_shards`` in the body.
    """

    def __init__(self, missing: set[int]):
        super().__init__(f"shards unavailable: {sorted(missing)}")
        self.missing = missing


def _default_query_embed(model: str, text: str):
    """Embed one query with ``model``; returns a 1-D numpy vector.

    Blocking, CPU-bound call: the router always runs it via ``asyncio.to_thread``.
    Tests inject a cheap fake through ``Router(query_embed=...)``.
    """
    # Lazy import: torch/sentence-transformers are heavy; BM25-only tests never pay for them.
    from src.retrieval.dense import get_embedder

    return get_embedder(model).encode_queries([text])[0]


class Router:
    """Scatter-gather coordinator: owns no documents, only the list of shards + routing state.

    Args:
        shards: one ``ShardClient`` per shard, in shard-id order (index i == shard i,
            which must match ``shard_of(doc_id, len(shards))`` used for writes).
        active_model: embedding model used for queries (flipped by cutover/rollback).
        timeout: per-shard deadline in seconds for search-path calls (stats/search/docs).
        failure_policy: "partial" (answer from the shards that replied, flagged) or
            "fail" (raise ``ShardFailure`` if any shard is missing).
        bm25_stats: "global" (DFS phase, correct) or anything else = each shard uses its
            LOCAL stats (the buggy baseline kept for the experiment that changed
            395/500 FiQA dev top-10s).
        state_path: JSON file persisting active/previous model across router restarts.
        query_embed: ``(model, text) -> vector``; injectable for tests.
        write_timeout: per-attempt deadline for writes (embedding on the shard is slow-ish).
    """

    def __init__(self, shards: Sequence[ShardClient], active_model: str = PRIMARY_MODEL,
                 timeout: float = SHARD_TIMEOUT_S, failure_policy: str = "partial",
                 bm25_stats: str = "global", state_path: Path | None = None,
                 query_embed: Callable | None = None, write_timeout: float = 30.0):
        self.shards = list(shards)
        self.timeout, self.write_timeout = timeout, write_timeout
        self.failure_policy, self.bm25_stats = failure_policy, bm25_stats
        self.query_embed = query_embed or _default_query_embed
        self.state_path = state_path
        self.active_model, self.previous_model = active_model, None
        # WHY saved state wins over the constructor argument: after a cutover the
        # ACTIVE_MODEL env var is stale; a restart must keep serving the cut-over model.
        if state_path and Path(state_path).exists():
            st = json.loads(Path(state_path).read_text())
            self.active_model, self.previous_model = st["active_model"], st.get("previous_model")

    # ------------------------------------------------------------------ plumbing
    async def _fan(self, op: str, payloads: dict[int, dict], timeout: float | None = None):
        """Call shards concurrently. Returns ({shard: response}, {missing shards}).

        ``payloads`` maps shard index -> request body, so callers can scatter to ALL shards
        (``self._all(...)``), to one shard (writes), or send different bodies per shard
        (``fetch``). A shard that errors or times out never raises here; it just lands in
        the "missing" set and the CALLER decides what that means (partial vs fail vs retry).
        """
        # PITFALL: ``or`` means timeout=0 falls back to the default too.
        timeout = timeout or self.timeout

        async def one(i: int, p: dict):
            # Returns (i, response) or (i, None). Catching here keeps one bad shard from
            # cancelling the whole gather below.
            try:
                # WHY wait_for on top of the HTTP timeout: it also bounds in-process clients.
                # (Only clients that actually ``await`` can be cut off; see LocalShardClient.)
                return i, await asyncio.wait_for(self.shards[i].call(op, p, timeout), timeout)
            except Exception as e:  # timeout, connection refused, 5xx...
                log.warning("shard %d %s failed: %r", i, op, e)
                return i, None

        # asyncio.gather: start all coroutines at once and wait until EVERY one finishes;
        # results come back in argument order. Total wait ~= the slowest shard (<= timeout).
        res = await asyncio.gather(*(one(i, p) for i, p in payloads.items()))
        return {i: r for i, r in res if r is not None}, {i for i, r in res if r is None}

    def _all(self, payload: dict) -> dict[int, dict]:
        """Same payload for every shard (a full fan-out). The dict object is shared, not copied."""
        return {i: payload for i in range(len(self.shards))}

    def _persist(self) -> None:
        """Save active/previous model so a router restart keeps the current cutover state."""
        if self.state_path:
            Path(self.state_path).write_text(json.dumps(
                {"active_model": self.active_model, "previous_model": self.previous_model}))

    # ------------------------------------------------------------------ search
    async def search(self, query: str, k: int = 10, strategy: str = "hybrid", fusion: str = DEFAULT_FUSION,
                     alpha: float = DEFAULT_ALPHA, include_text: bool = False, rerank_k: int = 0,
                     depth: int = CANDIDATE_DEPTH) -> dict:
        """Run one query across all shards (phases 0-5 in the module docstring).

        Args:
            strategy: "bm25", "dense" or "hybrid" (both lists, then fused).
            fusion / alpha: "rrf" or "linear"; ``alpha`` is the weight on DENSE.
            include_text: also return each hit's passage text (extra fetch round-trip).
            rerank_k: if > 0, rerank the top ``rerank_k`` with a cross-encoder.
            depth: how many candidates EACH shard returns per list (default 100).

        Returns:
            {"results": [{"doc_id", "score", ("text")}...], "partial": bool,
             "missing_shards": [shard ids], "model", "strategy", "latency_ms"}
        """
        if strategy not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown strategy {strategy!r}")
        t0 = time.perf_counter()
        model = self.active_model  # WHY capture once: a cutover mid-request must not mix models
        # Phase 0a: tokenise. dict.fromkeys dedupes while keeping order, so a repeated
        # query word is sent (and counted in df) once. Dense-only needs no terms at all.
        terms = list(dict.fromkeys(tokenize(query))) if strategy != "dense" else []
        # Phase 0b: start embedding NOW but don't wait for it. asyncio.to_thread runs the
        # blocking, CPU-heavy encoder in a worker thread so the event loop stays free;
        # create_task schedules it immediately, so it overlaps with the DFS round-trip below.
        emb = (asyncio.create_task(asyncio.to_thread(self.query_embed, model, query))
               if strategy != "bm25" else None)
        missing: set[int] = set()
        stats = None
        # Phase 1 (DFS) is skipped when it cannot matter: no terms, local-stats mode, or a
        # single shard (whose local stats ARE the global stats).
        if terms and self.bm25_stats == "global" and len(self.shards) > 1:  # phase 1: DFS
            ok, miss = await self._fan("stats", self._all({"terms": terms}))
            # A shard that misses this phase is flagged even if it answers phase 2: its docs
            # were then scored with stats that leave out its own counts (slightly off).
            missing |= miss
            # Sum N, total length and per-term df over shards = what one big index would see.
            # WHY: BM25's IDF and length normalisation depend on these. With per-shard
            # stats the SAME document scores differently depending on which shard holds it,
            # so per-shard scores would not be comparable when merged.
            g = CorpusStats(0, 0.0, {})
            for r in ok.values():
                g = g.merge(CorpusStats.from_dict(r))
            stats = g.to_dict()
        # Join the embedding task started in phase 0b. .tolist(): numpy arrays are not JSON.
        vec = (await emb).tolist() if emb else None
        # Phase 2: every shard scores its own docs with the GLOBAL stats and the router's
        # query vector, and returns {"bm25": top-depth, "dense": top-depth}.
        ok, miss = await self._fan("search", self._all(  # phase 2: query
            {"terms": terms, "k": depth, "stats": stats, "vector": vec, "model": model}))
        missing |= miss
        # Partial-result decision: no shard answered -> nothing to return; some missing and
        # policy "fail" -> error. Otherwise continue with what we have and flag it below.
        if not ok or (missing and self.failure_policy == "fail"):
            raise ShardFailure(missing)
        # Phase 3: union the per-shard lists and keep the best ``depth`` by score: one global
        # list for BM25, one for dense. WHY each shard returns ``depth`` (not k): any doc in
        # the global top-depth is necessarily in its own shard's top-depth, so none is lost.
        bm = merge_shard_lists([r["bm25"] for r in ok.values()], depth)  # phase 3: merge
        dn = merge_shard_lists([r["dense"] for r in ok.values()], depth)
        # Phase 4: fuse the two GLOBAL lists. WHY not fuse on each shard: RRF uses ranks
        # and linear fusion normalises scores per list, so both depend on the whole list;
        # per-shard fusion would see different lists and give a different answer (ADR 0003).
        ranked = bm if strategy == "bm25" else dn if strategy == "dense" else fuse(bm, dn, fusion, alpha)
        texts: dict[str, str] = {}
        # Phase 5: only now fetch text, and only for the few ids we will show or rerank.
        # WHY a separate round-trip: shipping text for 100 candidates x N shards in phase 2
        # would be mostly wasted bytes.
        if include_text or rerank_k:  # phase 5: fetch (+ rerank)
            texts = await self.fetch([d for d, _ in ranked[:max(k, rerank_k)]])
        if rerank_k:
            from src.retrieval.rerank import rerank

            # Cross-encoder rerank is CPU-heavy and blocking -> worker thread. Only the head
            # is reordered; the tail keeps its fused order.
            head = await asyncio.to_thread(rerank, query, ranked[:rerank_k], texts)
            ranked = head + ranked[rerank_k:]
        # Response: "partial" + "missing_shards" make a degraded answer visible to the caller.
        results = [{"doc_id": d, "score": round(float(s), 6), **({"text": texts.get(d, "")} if include_text else {})}
                   for d, s in ranked[:k]]
        return {"results": results, "partial": bool(missing), "missing_shards": sorted(missing),
                "model": model, "strategy": strategy, "latency_ms": round((time.perf_counter() - t0) * 1e3, 2)}

    async def fetch(self, ids: Sequence[str]) -> dict[str, str]:
        """Fetch phase: get passage text for ``ids``. Returns {doc_id: text}.

        Unlike search this is NOT a full fan-out: each id is sent only to its owner shard
        (same ``shard_of`` rule as writes), and shards that own none of the ids are skipped.
        """
        n = len(self.shards)
        # Group ids by owner shard: {shard: [ids]}, one request per shard involved.
        by_shard: dict[int, list[str]] = {}
        for d in ids:
            by_shard.setdefault(shard_of(d, n), []).append(d)
        # PITFALL: missing shards are ignored here; their docs just have no text
        # (search() then shows "" for them) and the response's "partial" flag does not see it.
        ok, _ = await self._fan("docs", {i: {"ids": v} for i, v in by_shard.items()})
        return {d: t for r in ok.values() for d, t in r.items()}

    # ------------------------------------------------------------------ writes
    async def write(self, op: str, doc_id: str, version: int, text: str | None = None, retries: int = 2) -> dict:
        """Route an upsert/delete to its owner shard, retrying (safe: writes are idempotent).

        ``version`` is chosen by the CALLER and must increase per doc_id. The shard keeps the
        highest version it has seen (deletes leave a "tombstone" version), so a retry or a
        late, out-of-order older write is a harmless no-op ({"applied": False, ...}).
        WHY that matters for retries: if attempt 1 actually succeeded but its reply was lost
        (timeout), attempt 2 must not apply the write twice or overwrite something newer.
        Returns the shard's reply plus ``attempts``; raises ShardFailure after the last retry.
        """
        # Exactly one owner shard: no fan-out for writes (no replicas in this project).
        i = shard_of(doc_id, len(self.shards))
        payload = {"doc_id": doc_id, "version": version, **({"text": text} if op == "upsert" else {})}
        for attempt in range(retries + 1):
            # _fan with a single-entry payload map = "call shard i with a timeout, never raise".
            ok, _ = await self._fan(op, {i: payload}, timeout=self.write_timeout)
            if i in ok:
                return {**ok[i], "attempts": attempt + 1}
            # 50 ms, 100 ms, 200 ms...: give a briefly overloaded/restarting shard time to recover
            # instead of hammering it.
            await asyncio.sleep(0.05 * 2 ** attempt)  # exponential backoff
        raise ShardFailure({i})

    # ------------------------------------------------------------------ migration
    # Blue/green model migration: "blue" = the model serving today, "green" = the new one.
    # Both indexes live side by side on every shard; queries switch between them by flipping
    # one router variable (active_model), so switching AND undoing it are instant.
    async def status(self) -> dict:
        """Health of every shard (doc counts, per-model ready/progress) plus router model state."""
        # POST /health via the RPC route returns 200 even for an unhealthy shard (the body says
        # healthy: false), so the caller can inspect it; a dead shard shows up in missing_shards.
        ok, missing = await self._fan("health", self._all({}), timeout=5.0)
        return {"active_model": self.active_model, "previous_model": self.previous_model,
                "shards": {str(i): ok[i] for i in sorted(ok)}, "missing_shards": sorted(missing)}

    async def flush(self, timeout: float = 10.0) -> dict:
        """Wait until every shard has applied every acked write (instant in sync mode).

        Only meaningful for APPLY_MODE=async (ADR 0013): returns once each shard's background
        applier has drained its queue, so tests and load scripts can measure "everything visible".
        """
        ok, missing = await self._fan("flush", self._all({"timeout": timeout}), timeout=timeout + 2.0)
        return {"drained": all(r.get("drained") for r in ok.values()) and not missing,
                "missing_shards": sorted(missing)}

    async def migrate(self, target: str, reencode: bool = False, throttle_s: float = 0.0) -> dict:
        """Start building ``target`` on every shard (dual-write begins immediately).

        Returns as soon as every shard has STARTED; the backfill (embedding all existing
        docs with ``target``) runs in a background thread on each shard. Poll ``status()``
        for progress. Queries keep using the old model meanwhile.
        """
        ok, missing = await self._fan("build_model", self._all(
            {"model": target, "reencode": reencode, "throttle_s": throttle_s}), timeout=30.0)
        # WHY all-or-nothing: a shard without the new index could never be cut over.
        # build_model is idempotent ("exists" if already started), so re-running is safe.
        if missing:
            raise ShardFailure(missing)
        # WHY warm now: loading the query encoder takes seconds, so do it before cutover.
        await asyncio.to_thread(self.query_embed, target, "warmup")
        return {"target": target, "shards": ok}

    async def _ready_everywhere(self, model: str) -> bool:
        """True only if EVERY shard answered, is healthy, and has finished backfilling ``model``."""
        st = await self.status()
        return not st["missing_shards"] and all(
            s["healthy"] and s["models"].get(model, {}).get("ready") for s in st["shards"].values())

    async def cutover(self, target: str) -> dict:
        """Make ``target`` the model used for new queries (also used by rollback).

        Safety gate: refuse unless the target index is complete on every shard, otherwise
        dense results would silently miss the docs not yet backfilled.
        """
        if not await self._ready_everywhere(target):
            raise RuntimeError(f"refusing cutover: {target} not ready on every shard")
        # Make sure the query encoder is loaded BEFORE flipping, so the first query after the
        # flip does not pay a multi-second model load (a latency spike users would see).
        await asyncio.to_thread(self.query_embed, target, "warmup")
        # The flip itself: one assignment. Requests already in flight captured the old
        # model in search() and finish with it; new requests see the new one.
        if target != self.active_model:
            self.previous_model, self.active_model = self.active_model, target
            self._persist()
        return {"active_model": self.active_model, "previous_model": self.previous_model}

    async def rollback(self) -> dict:
        """Cut back over to ``previous_model``. Instant because shards still hold its index
        (dual-write kept it up to date) until ``finalize`` drops it."""
        if not self.previous_model:
            raise RuntimeError("nothing to roll back to")
        return await self.cutover(self.previous_model)

    async def finalize(self, model: str) -> dict:
        """Drop an old model's index everywhere (ends dual-write). Never the active one.

        After this, rollback to ``model`` is no longer possible (its index is gone).
        """
        if model == self.active_model:
            raise RuntimeError("cannot drop the active model")
        # Best-effort: unlike migrate, missing shards are reported, not raised; re-run
        # finalize later to clean them up (drop_model on an absent model is a no-op).
        ok, missing = await self._fan("drop_model", self._all({"model": model}), timeout=30.0)
        if self.previous_model == model:
            self.previous_model = None
            self._persist()
        return {"dropped": model, "shards": ok, "missing_shards": sorted(missing)}


def single_node_router(models: Sequence[str] = (PRIMARY_MODEL,), active_model: str | None = None, **kw) -> Router:
    """Whole corpus in ONE in-process ShardStore: the single-node baseline.

    ``models=()`` builds a BM25-only node (no embeddings needed).
    Same ``Router`` code with one shard and no HTTP, so "single node vs 3 shards"
    comparisons measure partitioning + network cost, not two different implementations.
    """
    from src.beir_data import load_corpus
    from src.service.store import ShardStore

    store = ShardStore(0, 1).load(load_corpus(), models)
    return Router([LocalShardClient(store)], active_model=active_model or (models[0] if models else PRIMARY_MODEL), **kw)


# ---------------------------------------------------------------------- HTTP API (request bodies)
# Pydantic models = the JSON body schema of each endpoint. FastAPI validates incoming JSON
# against them (wrong type -> automatic HTTP 422) and fills in the defaults shown.
class SearchReq(BaseModel):
    """Body of POST /search; field names match ``Router.search`` keyword arguments."""
    query: str
    k: int = 10
    strategy: str = "hybrid"
    fusion: str = DEFAULT_FUSION
    alpha: float = DEFAULT_ALPHA
    include_text: bool = False
    rerank_k: int = 0


class WriteReq(BaseModel):
    """Body of POST /upsert and POST /delete (``text`` is required for upsert only)."""
    doc_id: str
    version: int
    text: str | None = None


class ModelReq(BaseModel):
    """Body of the /admin/* model endpoints (``reencode``/``throttle_s`` used by migrate only)."""
    model: str
    reencode: bool = False
    throttle_s: float = 0.0


def create_app():
    """FastAPI app for the router process. Shards come from env SHARD_URLS.

    Env vars: SHARD_URLS (comma list, in shard-id order), ACTIVE_MODEL (ignored if
    ROUTER_STATE already exists), FAILURE_POLICY, BM25_STATS, ROUTER_STATE (JSON path),
    SHARD_TIMEOUT_S (read by src.config). cluster.py / docker-compose set them.

    PITFALL: the request models above live at MODULE level. With
    ``from __future__ import annotations``, FastAPI cannot resolve models defined
    inside this function, and it silently treats the body as query params (HTTP 422).
    """
    import os

    import httpx
    from fastapi import FastAPI, HTTPException
    urls = [u for u in os.environ.get("SHARD_URLS", "").split(",") if u]
    # ONE shared async HTTP client (connection pool) for all shards and all requests, so
    # connections stay open between queries (keep-alive). Limits cap open sockets per router.
    client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=64, max_connections=128))
    router = Router([HttpShardClient(u, client) for u in urls],
                    active_model=os.environ.get("ACTIVE_MODEL", PRIMARY_MODEL),
                    failure_policy=os.environ.get("FAILURE_POLICY", "partial"),
                    bm25_stats=os.environ.get("BM25_STATS", "global"),
                    state_path=Path(os.environ.get("ROUTER_STATE", ARTIFACTS_DIR / "router_state.json")))
    app = FastAPI(title="FiQA router")
    # Exposed so tests/tools holding the app can reach the Router object directly.
    app.state.router = router

    # Translate Router exceptions into HTTP status codes in one place:
    #   ShardFailure            -> 503 Service Unavailable (+ which shards are missing)
    #   RuntimeError/ValueError -> 409 Conflict (e.g. "refusing cutover", unknown strategy)
    # PITFALL: ShardFailure subclasses RuntimeError, so it MUST be caught first.
    async def guard(coro):
        try:
            return await coro
        except ShardFailure as e:
            raise HTTPException(503, {"error": str(e), "missing_shards": sorted(e.missing)})
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409, str(e))

    # WHY warm at startup: the first query would otherwise pay the multi-second encoder load.
    # cluster.py waits for /health, which only answers after startup hooks finish.
    @app.on_event("startup")
    async def _warm():
        await asyncio.to_thread(router.query_embed, router.active_model, "warmup")

    @app.post("/search")
    async def search(req: SearchReq):
        # model_dump() -> dict of the body fields -> keyword args of Router.search.
        return await guard(router.search(**req.model_dump()))

    @app.post("/upsert")
    async def upsert(req: WriteReq):
        if req.text is None:
            raise HTTPException(422, "text required")
        return await guard(router.write("upsert", req.doc_id, req.version, req.text))

    @app.post("/delete")
    async def delete(req: WriteReq):
        return await guard(router.write("delete", req.doc_id, req.version))

    @app.get("/health")
    async def health():
        # Liveness of the ROUTER process only (cheap, no shard calls). Use /admin/status
        # to see per-shard health.
        return {"ok": True, "active_model": router.active_model, "n_shards": len(router.shards)}

    # Admin endpoints drive the blue/green model migration (see module docstring).
    @app.get("/admin/status")
    async def status():
        return await router.status()

    # Wait until every shard has applied every acked write (async apply mode, ADR 0013).
    @app.post("/admin/flush")
    async def flush(timeout: float = 10.0):
        return await router.flush(timeout)

    @app.post("/admin/migrate")
    async def migrate(req: ModelReq):
        return await guard(router.migrate(req.model, req.reencode, req.throttle_s))

    @app.post("/admin/cutover")
    async def cutover(req: ModelReq):
        return await guard(router.cutover(req.model))

    @app.post("/admin/rollback")
    async def rollback():
        return await guard(router.rollback())

    @app.post("/admin/finalize")
    async def finalize(req: ModelReq):
        return await guard(router.finalize(req.model))

    return app


if __name__ == "__main__":
    # Started as ``python -m src.service.router`` by cluster.py / docker-compose.
    # uvicorn = the ASGI web server that runs the FastAPI app's event loop and sockets.
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="warning")
