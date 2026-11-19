"""Offline index build: embed the whole corpus once per model and cache it.

WHAT
    ``python -m src.index.build_index --models minilm bge-small`` writes, per model:
        artifacts/emb_<model>.npy        float32 matrix (n_docs, 384), unit rows
        artifacts/emb_<model>.meta.json  doc ids (row order), text hashes, model info
    ``--package`` zips the artifacts into dist/ so they can be attached as a
    release asset (the "pre-built index artifact" from the reproducibility contract).

WHY A CACHE
    Encoding 57K passages takes minutes on CPU. Encoding queries takes ms.
    Shards start in seconds by slicing this matrix instead of re-encoding.
    The text hashes let the migration backfill (Part B3) reuse a cached
    vector only if the passage text is unchanged since the cache was built.

PITFALLS
    * The row order MUST be stored next to the matrix. Never assume dict order.
    * Anything that changes a DOCUMENT vector (model, max_seq_length) must change
      the cache. We store the spec in the meta file and check it on load. The BGE
      query prefix is NOT part of it: it is applied to queries only, never to documents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS_DIR, MODELS, ROOT


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _paths(model_key: str, artifacts: Path = ARTIFACTS_DIR) -> tuple[Path, Path]:
    return artifacts / f"emb_{model_key}.npy", artifacts / f"emb_{model_key}.meta.json"


def build_cache(model_key: str, batch_size: int = 128) -> Path:
    """Embed every corpus passage with ``model_key`` and save matrix + metadata."""
    from src.beir_data import load_corpus
    from src.retrieval.dense import get_embedder

    corpus = load_corpus()
    ids = list(corpus)
    texts = [corpus[i] for i in ids]
    t0 = time.time()
    vecs = get_embedder(model_key).encode_docs(texts, batch_size=batch_size, progress=True)
    npy, meta = _paths(model_key)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    spec = MODELS[model_key]
    meta.write_text(json.dumps({
        "model": spec.hf_name, "max_seq_length": spec.max_seq_length, "dim": spec.dim,
        "seconds": round(time.time() - t0, 1), "ids": ids, "hashes": [text_hash(t) for t in texts],
    }))
    print(f"[build_index] {model_key}: {vecs.shape} in {time.time() - t0:.0f}s -> {npy}")
    return npy


def load_cache(model_key: str, artifacts: Path = ARTIFACTS_DIR):
    """Return (ids, hashes, matrix) or None if the cache does not exist or is stale."""
    npy, meta = _paths(model_key, artifacts)
    if not (npy.exists() and meta.exists()):
        return None
    m = json.loads(meta.read_text())
    spec = MODELS[model_key]
    if m["model"] != spec.hf_name or m["max_seq_length"] != spec.max_seq_length:
        print(f"[build_index] cache for {model_key} is stale (spec changed); ignoring")
        return None
    # WHY mmap: shards each read only their ~1/3 of the rows; mmap avoids loading all.
    return m["ids"], m["hashes"], np.load(npy, mmap_mode="r")


def package(out: Path = ROOT / "dist" / "fiqa_index_artifacts.zip") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ARTIFACTS_DIR.glob("emb_*")):
            z.write(p, arcname=f"artifacts/{p.name}")
    print(f"[build_index] packaged -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--package", action="store_true", help="zip artifacts into dist/")
    args = ap.parse_args()
    for mk in args.models:
        if load_cache(mk) is None:
            build_cache(mk)
        else:
            print(f"[build_index] {mk}: cache exists, skipping")
    if args.package:
        package()


if __name__ == "__main__":
    main()
