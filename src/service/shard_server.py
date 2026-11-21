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
    APPLY_MODE           "sync" (default: apply before ack) or "async" (ack after WAL, apply in
                         the background; ASYNC_MAX_LAG_S, ASYNC_BATCH_MAX, ASYNC_EMBED_WORKERS
                         tune it).
    TORCH_THREADS        intra-op threads for this process (avoid oversubscription:
                         4 processes x 32 threads on 32 cores = thrashing)

ENDPOINT CONCURRENCY
    Endpoints are plain ``def`` (not ``async def``), so FastAPI runs them in a
    thread pool. Many searches run concurrently under the store's shared read
    lock, and writes take the exclusive lock.

ERRORS
    KeyError (unknown model) -> 409, ShardUnhealthy -> 503 (router marks shard missing).
    ValueError (unknown op, bad payload) -> 400. Anything else -> FastAPI's default 500.

REQUEST LIFECYCLE (one search, seen from the shard)
    router POST /search {"terms", "k", "stats", "vector", "model"} -> ``rpc`` (thread
    pool) -> ``dispatch`` -> ``ShardStore.search`` under the read lock -> reply
    {"bm25": [[doc_id, score], ...], "dense": [[doc_id, score], ...], "shard": id}.
    The shard never fuses or merges; that is the router's job (it sees all shards).

HOW TO READ THIS FILE
    ``build_store`` = process startup (load MY slice of the corpus, replay the WAL,
    preload encoders); ``create_app`` = the HTTP surface (two routes);
    ``__main__`` = run it under uvicorn. All real logic lives in store.py.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from src.config import ARTIFACTS_DIR, PRIMARY_MODEL


def build_store():
    """Build this process's ShardStore from env vars (runs once, at startup).

    Returns a ready ShardStore: base corpus slice indexed, WAL replayed, encoders loaded.
    """
    # Imports inside the function: importing this module (e.g. in tests that pass their own
    # store to create_app) must not drag in the corpus loader and heavy ML libraries.
    from src.beir_data import load_corpus
    from src.index.sharding import shard_of
    from src.service.store import ShardStore

    sid, n = int(os.environ.get("SHARD_ID", "0")), int(os.environ.get("N_SHARDS", "1"))
    models = [m for m in os.environ.get("SHARD_MODELS", PRIMARY_MODEL).split(",") if m]
    state = Path(os.environ.get("STATE_DIR", ARTIFACTS_DIR / f"shard{sid}"))
    t0 = time.time()
    # Keep only the docs this shard owns: crc32(doc_id) % N == my id. The router uses the same
    # shard_of() to route writes and fetches, so both sides always agree on ownership.
    # (Every shard reads the full corpus file and filters; simple, fine at 57k passages.)
    corpus = {d: t for d, t in load_corpus().items() if shard_of(d, n) == sid}
    # load(): index the base slice, then replay this shard's WAL from STATE_DIR so writes
    # acked before a restart are back before we serve traffic.
    # APPLY_MODE=sync (default) applies before the ack; APPLY_MODE=async acks after the WAL and
    # applies in the background. ASYNC_MAX_LAG_S / ASYNC_BATCH_MAX tune async mode.
    store = ShardStore(sid, n, state_dir=state, apply_mode=os.environ.get("APPLY_MODE", "sync"),
                       max_lag_s=float(os.environ.get("ASYNC_MAX_LAG_S", "0.5")),
                       batch_max=int(os.environ.get("ASYNC_BATCH_MAX", "32")),
                       embed_workers=int(os.environ.get("ASYNC_EMBED_WORKERS", "8"))).load(corpus, models)
    # EMBED_BACKEND=hash is a fake, instant embedder used in tests; nothing to preload then.
    if os.environ.get("EMBED_BACKEND", "st") != "hash":
        from src.retrieval.dense import get_embedder

        for m in store.dense:  # WHY preload: the first write must not pay a multi-second model load
            get_embedder(m)
    logging.warning("shard %d/%d ready: %d docs, models=%s, replayed=%d, %.1fs",
                    sid, n, len(store.docs), list(store.dense), store.counters["replayed"], time.time() - t0)
    return store


def create_app(store=None):
    """Wrap ``store`` (or a freshly built one) in a FastAPI app.

    Tests pass a small in-memory store; the real process passes nothing and gets
    ``build_store()``. Two routes: GET /health and POST /{op}.
    """
    from fastapi import FastAPI, HTTPException

    from src.service.router import dispatch
    from src.service.store import ShardUnhealthy

    # Cap PyTorch's intra-op threads for THIS process (see TORCH_THREADS in the docstring).
    if "TORCH_THREADS" in os.environ:
        import torch

        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    store = store or build_store()
    app = FastAPI(title=f"FiQA shard {store.shard_id}")
    app.state.store = store

    # GET /health: what cluster.py polls at startup and what a load balancer/orchestrator
    # would probe. Returns 503 when the shard marked itself unhealthy (crash-only recovery:
    # restart it). The router's /admin/status uses POST /health below instead, which
    # always returns 200 with "healthy": true/false so the details stay readable.
    @app.get("/health")
    def health_get():
        h = store.health()
        if not h["healthy"]:
            raise HTTPException(503, h)
        return h

    # One generic RPC route: POST /stats, /search, /docs, /upsert, ... all land here and
    # ``dispatch`` (router.py) picks the ShardStore method. ``payload: dict`` tells FastAPI to
    # read the whole JSON body as a plain dict (no per-op schema validation).
    # WHY plain ``def``: FastAPI runs it in a worker thread, so a slow search or an embedding
    # during a write does not block the event loop and other requests keep being served.
    @app.post("/{op}")
    def rpc(op: str, payload: dict):
        try:
            return dispatch(store, op, payload)
        except ShardUnhealthy as e:
            raise HTTPException(503, str(e))
        # PITFALL: a payload missing a required key (p["terms"]) is also a KeyError, so it
        # surfaces as 409 rather than 400.
        except KeyError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


if __name__ == "__main__":
    # Started as ``python -m src.service.shard_server`` (one process per shard) by cluster.py.
    import uvicorn

    logging.basicConfig(level=logging.WARNING)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8101")), log_level="warning")
