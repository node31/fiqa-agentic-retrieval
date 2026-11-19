"""Shard server: one process = one shard = its own BM25 + dense indexes, over HTTP.

WHAT
    A thin FastAPI wrapper around ``ShardStore``. Every RPC is ``POST /<op>``
    with a JSON body (stats, search, docs, upsert, delete, build_model,
    drop_model, health). ``router.dispatch`` holds the op -> method mapping.

CONFIG (env vars, set by docker-compose or scripts/run_cluster.py)
    SHARD_ID, N_SHARDS   which slice of the corpus this process owns
    PORT                 HTTP port
    STATE_DIR            where the WAL + models.json live (persist across restarts)
    SHARD_MODELS         comma list of embedding models to index at start
    TORCH_THREADS        intra-op threads for this process (avoid oversubscription:
                         4 processes x 32 threads on 32 cores = thrashing)

ENDPOINT CONCURRENCY
    Endpoints are plain ``def`` (not ``async def``), so FastAPI runs them in a
    thread pool. Many searches run concurrently under the store's shared read
    lock, and writes take the exclusive lock.

ERRORS
    KeyError (unknown model) -> 409, ShardUnhealthy -> 503 (router marks shard missing).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from src.config import ARTIFACTS_DIR, PRIMARY_MODEL


def build_store():
    from src.beir_data import load_corpus
    from src.index.sharding import shard_of
    from src.service.store import ShardStore

    sid, n = int(os.environ.get("SHARD_ID", "0")), int(os.environ.get("N_SHARDS", "1"))
    models = [m for m in os.environ.get("SHARD_MODELS", PRIMARY_MODEL).split(",") if m]
    state = Path(os.environ.get("STATE_DIR", ARTIFACTS_DIR / f"shard{sid}"))
    t0 = time.time()
    corpus = {d: t for d, t in load_corpus().items() if shard_of(d, n) == sid}
    store = ShardStore(sid, n, state_dir=state).load(corpus, models)
    if os.environ.get("EMBED_BACKEND", "st") != "hash":
        from src.retrieval.dense import get_embedder

        for m in store.dense:  # WHY preload: the first write must not pay a multi-second model load
            get_embedder(m)
    logging.warning("shard %d/%d ready: %d docs, models=%s, replayed=%d, %.1fs",
                    sid, n, len(store.docs), list(store.dense), store.counters["replayed"], time.time() - t0)
    return store


def create_app(store=None):
    from fastapi import FastAPI, HTTPException

    from src.service.router import dispatch
    from src.service.store import ShardUnhealthy

    if "TORCH_THREADS" in os.environ:
        import torch

        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    store = store or build_store()
    app = FastAPI(title=f"FiQA shard {store.shard_id}")
    app.state.store = store

    @app.get("/health")
    def health_get():
        h = store.health()
        if not h["healthy"]:
            raise HTTPException(503, h)
        return h

    @app.post("/{op}")
    def rpc(op: str, payload: dict):
        try:
            return dispatch(store, op, payload)
        except ShardUnhealthy as e:
            raise HTTPException(503, str(e))
        except KeyError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.WARNING)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8101")), log_level="warning")
