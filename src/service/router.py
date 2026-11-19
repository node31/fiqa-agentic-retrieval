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

log = logging.getLogger("router")


class ShardClient(Protocol):
    async def call(self, op: str, payload: dict, timeout: float) -> dict: ...


def dispatch(store, op: str, p: dict) -> dict:
    """Map an RPC name to a ShardStore method. Shared by the HTTP server and in-process client."""
    if op == "stats":
        return store.stats(p["terms"])
    if op == "search":
        return store.search(p["terms"], p["k"], p.get("stats"), p.get("vector"), p.get("model"))
    if op == "docs":
        return store.get_docs(p["ids"])
    if op == "upsert":
        return store.upsert(p["doc_id"], int(p["version"]), p["text"])
    if op == "delete":
        return store.delete(p["doc_id"], int(p["version"]))
    if op == "build_model":
        return store.start_build(p["model"])
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
        return dispatch(self.store, op, payload)


class HttpShardClient:
    """Calls a shard server over HTTP with keep-alive connections (httpx)."""

    def __init__(self, url: str, client):
        self.url, self.client = url.rstrip("/"), client

    async def call(self, op: str, payload: dict, timeout: float) -> dict:
        r = await self.client.post(f"{self.url}/{op}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()


class ShardFailure(RuntimeError):
    def __init__(self, missing: set[int]):
        super().__init__(f"shards unavailable: {sorted(missing)}")
        self.missing = missing


def _default_query_embed(model: str, text: str):
    from src.retrieval.dense import get_embedder

    return get_embedder(model).encode_queries([text])[0]


class Router:
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
        if state_path and Path(state_path).exists():
            st = json.loads(Path(state_path).read_text())
            self.active_model, self.previous_model = st["active_model"], st.get("previous_model")

    # ------------------------------------------------------------------ plumbing
    async def _fan(self, op: str, payloads: dict[int, dict], timeout: float | None = None):
        """Call shards concurrently. Returns ({shard: response}, {missing shards})."""
        timeout = timeout or self.timeout

        async def one(i: int, p: dict):
            try:
                # WHY wait_for on top of the HTTP timeout: it also bounds in-process clients.
                return i, await asyncio.wait_for(self.shards[i].call(op, p, timeout), timeout)
            except Exception as e:  # timeout, connection refused, 5xx...
                log.warning("shard %d %s failed: %r", i, op, e)
                return i, None

        res = await asyncio.gather(*(one(i, p) for i, p in payloads.items()))
        return {i: r for i, r in res if r is not None}, {i for i, r in res if r is None}

    def _all(self, payload: dict) -> dict[int, dict]:
        return {i: payload for i in range(len(self.shards))}

    def _persist(self) -> None:
        if self.state_path:
            Path(self.state_path).write_text(json.dumps(
                {"active_model": self.active_model, "previous_model": self.previous_model}))

    # ------------------------------------------------------------------ search
    async def search(self, query: str, k: int = 10, strategy: str = "hybrid", fusion: str = DEFAULT_FUSION,
                     alpha: float = DEFAULT_ALPHA, include_text: bool = False, rerank_k: int = 0,
                     depth: int = CANDIDATE_DEPTH) -> dict:
        if strategy not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown strategy {strategy!r}")
        t0 = time.perf_counter()
        model = self.active_model  # WHY capture once: a cutover mid-request must not mix models
        terms = list(dict.fromkeys(tokenize(query))) if strategy != "dense" else []
        emb = (asyncio.create_task(asyncio.to_thread(self.query_embed, model, query))
               if strategy != "bm25" else None)
        missing: set[int] = set()
        stats = None
        if terms and self.bm25_stats == "global" and len(self.shards) > 1:  # phase 1: DFS
            ok, miss = await self._fan("stats", self._all({"terms": terms}))
            missing |= miss
            g = CorpusStats(0, 0.0, {})
            for r in ok.values():
                g = g.merge(CorpusStats.from_dict(r))
            stats = g.to_dict()
        vec = (await emb).tolist() if emb else None
        ok, miss = await self._fan("search", self._all(  # phase 2: query
            {"terms": terms, "k": depth, "stats": stats, "vector": vec, "model": model}))
        missing |= miss
        if not ok or (missing and self.failure_policy == "fail"):
            raise ShardFailure(missing)
        bm = merge_shard_lists([r["bm25"] for r in ok.values()], depth)  # phase 3: merge
        dn = merge_shard_lists([r["dense"] for r in ok.values()], depth)
        ranked = bm if strategy == "bm25" else dn if strategy == "dense" else fuse(bm, dn, fusion, alpha)
        texts: dict[str, str] = {}
        if include_text or rerank_k:  # phase 5: fetch (+ rerank)
            texts = await self.fetch([d for d, _ in ranked[:max(k, rerank_k)]])
        if rerank_k:
            from src.retrieval.rerank import rerank

            head = await asyncio.to_thread(rerank, query, ranked[:rerank_k], texts)
            ranked = head + ranked[rerank_k:]
        results = [{"doc_id": d, "score": round(float(s), 6), **({"text": texts.get(d, "")} if include_text else {})}
                   for d, s in ranked[:k]]
        return {"results": results, "partial": bool(missing), "missing_shards": sorted(missing),
                "model": model, "strategy": strategy, "latency_ms": round((time.perf_counter() - t0) * 1e3, 2)}

    async def fetch(self, ids: Sequence[str]) -> dict[str, str]:
        n = len(self.shards)
        by_shard: dict[int, list[str]] = {}
        for d in ids:
            by_shard.setdefault(shard_of(d, n), []).append(d)
        ok, _ = await self._fan("docs", {i: {"ids": v} for i, v in by_shard.items()})
        return {d: t for r in ok.values() for d, t in r.items()}

    # ------------------------------------------------------------------ writes
    async def write(self, op: str, doc_id: str, version: int, text: str | None = None, retries: int = 2) -> dict:
        """Route an upsert/delete to its owner shard, retrying (safe: writes are idempotent)."""
        i = shard_of(doc_id, len(self.shards))
        payload = {"doc_id": doc_id, "version": version, **({"text": text} if op == "upsert" else {})}
        for attempt in range(retries + 1):
            ok, _ = await self._fan(op, {i: payload}, timeout=self.write_timeout)
            if i in ok:
                return {**ok[i], "attempts": attempt + 1}
            await asyncio.sleep(0.05 * 2 ** attempt)  # exponential backoff
        raise ShardFailure({i})

    # ------------------------------------------------------------------ migration
    async def status(self) -> dict:
        ok, missing = await self._fan("health", self._all({}), timeout=5.0)
        return {"active_model": self.active_model, "previous_model": self.previous_model,
                "shards": {str(i): ok[i] for i in sorted(ok)}, "missing_shards": sorted(missing)}

    async def migrate(self, target: str) -> dict:
        """Start building ``target`` on every shard (dual-write begins immediately)."""
        ok, missing = await self._fan("build_model", self._all({"model": target}), timeout=30.0)
        if missing:
            raise ShardFailure(missing)
        # WHY warm now: loading the query encoder takes seconds, so do it before cutover.
        await asyncio.to_thread(self.query_embed, target, "warmup")
        return {"target": target, "shards": ok}

    async def _ready_everywhere(self, model: str) -> bool:
        st = await self.status()
        return not st["missing_shards"] and all(
            s["healthy"] and s["models"].get(model, {}).get("ready") for s in st["shards"].values())

    async def cutover(self, target: str) -> dict:
        if not await self._ready_everywhere(target):
            raise RuntimeError(f"refusing cutover: {target} not ready on every shard")
        await asyncio.to_thread(self.query_embed, target, "warmup")
        if target != self.active_model:
            self.previous_model, self.active_model = self.active_model, target
            self._persist()
        return {"active_model": self.active_model, "previous_model": self.previous_model}

    async def rollback(self) -> dict:
        if not self.previous_model:
            raise RuntimeError("nothing to roll back to")
        return await self.cutover(self.previous_model)

    async def finalize(self, model: str) -> dict:
        """Drop an old model's index everywhere (ends dual-write). Never the active one."""
        if model == self.active_model:
            raise RuntimeError("cannot drop the active model")
        ok, missing = await self._fan("drop_model", self._all({"model": model}), timeout=30.0)
        if self.previous_model == model:
            self.previous_model = None
            self._persist()
        return {"dropped": model, "shards": ok, "missing_shards": sorted(missing)}


def single_node_router(models: Sequence[str] = (PRIMARY_MODEL,), **kw) -> Router:
    """Whole corpus in ONE in-process ShardStore: the single-node baseline."""
    from src.beir_data import load_corpus
    from src.service.store import ShardStore

    store = ShardStore(0, 1).load(load_corpus(), models)
    return Router([LocalShardClient(store)], active_model=models[0], **kw)


def create_app():
    """FastAPI app for the router process. Shards come from env SHARD_URLS."""
    import os

    import httpx
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    urls = [u for u in os.environ.get("SHARD_URLS", "").split(",") if u]
    client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=64, max_connections=128))
    router = Router([HttpShardClient(u, client) for u in urls],
                    active_model=os.environ.get("ACTIVE_MODEL", PRIMARY_MODEL),
                    failure_policy=os.environ.get("FAILURE_POLICY", "partial"),
                    bm25_stats=os.environ.get("BM25_STATS", "global"),
                    state_path=Path(os.environ.get("ROUTER_STATE", ARTIFACTS_DIR / "router_state.json")))
    app = FastAPI(title="FiQA router")
    app.state.router = router

    class SearchReq(BaseModel):
        query: str
        k: int = 10
        strategy: str = "hybrid"
        fusion: str = DEFAULT_FUSION
        alpha: float = DEFAULT_ALPHA
        include_text: bool = False
        rerank_k: int = 0

    class WriteReq(BaseModel):
        doc_id: str
        version: int
        text: str | None = None

    class ModelReq(BaseModel):
        model: str

    async def guard(coro):
        try:
            return await coro
        except ShardFailure as e:
            raise HTTPException(503, {"error": str(e), "missing_shards": sorted(e.missing)})
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409, str(e))

    @app.on_event("startup")
    async def _warm():
        await asyncio.to_thread(router.query_embed, router.active_model, "warmup")

    @app.post("/search")
    async def search(req: SearchReq):
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
        return {"ok": True, "active_model": router.active_model, "n_shards": len(router.shards)}

    @app.get("/admin/status")
    async def status():
        return await router.status()

    @app.post("/admin/migrate")
    async def migrate(req: ModelReq):
        return await guard(router.migrate(req.model))

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
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="warning")
