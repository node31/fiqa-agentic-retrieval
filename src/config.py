"""Central configuration: paths, model registry and tunable defaults.

WHAT
    One place for every constant the rest of the code reads: where data and
    artifacts live, which embedding models exist, and default search knobs.

WHY
    Scattered magic numbers make experiments irreproducible. When DESIGN.md says
    "RRF k=60, candidate depth 100", that must be true in exactly one place.

HOW IT FITS
    Imported by everything. Values can be overridden with environment variables
    (useful inside Docker), e.g. ``FIQA_DATA_DIR``, ``ARTIFACTS_DIR``.

PITFALLS
    * Query prefixes: BGE models were trained with an instruction prefix on the
      *query* side only. Forgetting it costs a few nDCG points; adding it to
      documents too is also wrong. The registry keeps that knowledge here.
    * ``max_seq_length`` silently truncates long passages. It is part of the
      model identity for reproducibility, so it lives in the registry too.

SEE ALSO
    docs/topics/02_dense_retrieval.md, examples/05_bge_query_prefix.py

HOW TO READ THIS FILE (top to bottom)
    1. paths       ROOT, DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR (+ .env loading)
    2. models      ModelSpec + the MODELS registry (which embedding models exist)
    3. search      fusion / BM25 knobs used by Part A retrieval
    4. service     sharding, ports and timeouts for Part B
    5. agent/LLM   backend choice and token prices for Part C
    6. SEED        one random seed for everything that samples
    Most values follow the pattern ``os.environ.get("NAME", default)``: the
    environment wins if set, otherwise the default here is used. Note the
    values are read ONCE, when this module is first imported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root: this file is <root>/src/config.py, parents[1] = <root>

# WHY: loading .env here means every entry point (CLI, server, scripts) sees the
# same settings without each one remembering to call load_dotenv().
# override=False: a variable already set in the real environment beats the .env file.
try:  # pragma: no cover - optional dependency at import time
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

# Input dataset folder: corpus.jsonl, queries.jsonl, qrels/<split>.tsv.
DATA_DIR = Path(os.environ.get("FIQA_DATA_DIR", ROOT / "data" / "raw" / "fiqa"))
# Generated files: built indexes, cached embeddings, service/shard state (cluster*/, shard*/).
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
# WHY overridable: toy runs on the tiny dataset must not overwrite the real FiQA results.
# (Set all three, e.g. FIQA_DATA_DIR=tests/fixtures/tinyfiqa plus scratch ARTIFACTS_DIR/RESULTS_DIR.)
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


@dataclass(frozen=True)  # frozen = immutable: a spec can't be edited by accident after creation
class ModelSpec:
    """Everything needed to reproduce one embedding model's vectors.

    Attributes:
        key: short name used in file names, API calls and flags ("minilm").
        hf_name: Hugging Face model id.
        dim: embedding dimension (both our models are 384-d).
        query_prefix: text prepended to queries only (BGE instruction).
        max_seq_length: tokens kept per passage; the rest is truncated.

    An "embedding model" turns a text into a fixed-length list of numbers
    (a vector, here 384 numbers) so that texts with similar MEANING get nearby
    vectors; dense search = "find passages whose vector is closest to the query's".
    """

    key: str
    hf_name: str
    dim: int
    query_prefix: str = ""
    max_seq_length: int = 256


# The registry: short key -> full spec. Code looks models up by key (MODELS["minilm"]),
# so adding a model = adding one entry here.
MODELS: dict[str, ModelSpec] = {
    # No query_prefix: MiniLM encodes queries and passages the same way.
    "minilm": ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384),
    "bge-small": ModelSpec(
        "bge-small",
        "BAAI/bge-small-en-v1.5",
        384,
        # WHY: from the BGE model card. Used for short-query -> passage retrieval.
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}

# Primary model serves traffic first; B3 migrates to the secondary ("upgrade").
# Values are MODELS keys. PRIMARY_MODEL is also the default --model in src/eval.py.
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "minilm")
SECONDARY_MODEL = os.environ.get("SECONDARY_MODEL", "bge-small")

# Cross-encoder used for optional reranking: it reads query AND passage together
# (more accurate, much slower), so it only re-scores the top few candidates.
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Search defaults (tuned on dev only; see results/bench.json) -----------
CANDIDATE_DEPTH = 100  # per-component list depth fed into fusion
# RRF ("reciprocal rank fusion") scores a doc as sum of 1 / (RRF_K + rank) over the
# lists. e.g. rank 1 -> 1/61, rank 2 -> 1/62: a large k flattens the gap between ranks.
RRF_K = 60  # Cormack et al. 2009 default; see docs/topics/05_hybrid_fusion.md
# BM25 keyword-scoring knobs (standard textbook values): k1 = how fast repeated
# words stop adding score, b = how strongly long documents are penalised (0..1).
BM25_K1 = 1.2
BM25_B = 0.75
# "linear" = min-max normalise each list's scores to 0..1, then alpha*dense + (1-alpha)*bm25.
DEFAULT_FUSION = "linear"  # tuned on dev: beats RRF here (results/ablation_fusion_minilm.json)
DEFAULT_ALPHA = 0.7  # weight on dense; BM25 is much weaker than dense on FiQA, so dense dominates
# PITFALL: src/eval.py's --fusion/--alpha defaults are rrf/0.5, NOT these values;
# src/search.py does use these.

# --- Service defaults --------------------------------------------------------
# NOTE: N_SHARDS and SHARD_BASE_PORT document the defaults, but at the time of writing
# no module imports them: shard_server.py reads the N_SHARDS env var itself and
# cluster.py has its own n_shards=3 / base_port=8101 arguments (shard i -> port 8101 + i).
N_SHARDS = int(os.environ.get("N_SHARDS", "3"))  # the corpus is split into this many shard processes
# The router stops waiting for a slow shard after this many seconds and returns a
# "partial" result (missing_shards lists who didn't answer) instead of hanging.
SHARD_TIMEOUT_S = float(os.environ.get("SHARD_TIMEOUT_S", "0.5"))
ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
SHARD_BASE_PORT = int(os.environ.get("SHARD_BASE_PORT", "8101"))

# --- Agent / LLM ----------------------------------------------------------
# "fake" = a deterministic offline stand-in (no network, no cost), so tests and
# demos run anywhere; set a real backend in .env to call an actual LLM.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "fake")
COPILOT_MODEL = os.environ.get("COPILOT_MODEL", "claude-haiku-4.5")
# WHY placeholders: gpt-6-sol pricing was unknown when this was written.
# Cost numbers are "tokens x these prices"; edit .env to get real dollars.
PRICE_PER_1M_INPUT = float(os.environ.get("PRICE_PER_1M_INPUT", "1.00"))
PRICE_PER_1M_OUTPUT = float(os.environ.get("PRICE_PER_1M_OUTPUT", "4.00"))

# Fixed random seed (e.g. eval.py's warmup-query sample) so reruns pick the same items.
SEED = 42
