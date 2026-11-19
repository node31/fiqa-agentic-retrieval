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
    docs/02_dense_retrieval.md, examples/05_bge_query_prefix.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# WHY: loading .env here means every entry point (CLI, server, scripts) sees the
# same settings without each one remembering to call load_dotenv().
try:  # pragma: no cover - optional dependency at import time
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

DATA_DIR = Path(os.environ.get("FIQA_DATA_DIR", ROOT / "data" / "raw" / "fiqa"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to reproduce one embedding model's vectors.

    Attributes:
        key: short name used in file names, API calls and flags ("minilm").
        hf_name: Hugging Face model id.
        dim: embedding dimension (both our models are 384-d).
        query_prefix: text prepended to queries only (BGE instruction).
        max_seq_length: tokens kept per passage; the rest is truncated.
    """

    key: str
    hf_name: str
    dim: int
    query_prefix: str = ""
    max_seq_length: int = 256


MODELS: dict[str, ModelSpec] = {
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
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "minilm")
SECONDARY_MODEL = os.environ.get("SECONDARY_MODEL", "bge-small")

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Search defaults (tuned on dev only; see results/bench.json) -----------
CANDIDATE_DEPTH = 100  # per-component list depth fed into fusion
RRF_K = 60  # Cormack et al. 2009 default; see docs/05_hybrid_fusion.md
BM25_K1 = 1.2
BM25_B = 0.75
DEFAULT_FUSION = "rrf"
DEFAULT_ALPHA = 0.5  # weight on dense for linear fusion

# --- Service defaults --------------------------------------------------------
N_SHARDS = int(os.environ.get("N_SHARDS", "3"))
SHARD_TIMEOUT_S = float(os.environ.get("SHARD_TIMEOUT_S", "0.5"))
ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
SHARD_BASE_PORT = int(os.environ.get("SHARD_BASE_PORT", "8101"))

# --- Agent / LLM ----------------------------------------------------------
LLM_BACKEND = os.environ.get("LLM_BACKEND", "fake")
COPILOT_MODEL = os.environ.get("COPILOT_MODEL", "claude-haiku-4.5")
# WHY placeholders: gpt-6-sol pricing was unknown when this was written.
# Cost numbers are "tokens x these prices"; edit .env to get real dollars.
PRICE_PER_1M_INPUT = float(os.environ.get("PRICE_PER_1M_INPUT", "1.00"))
PRICE_PER_1M_OUTPUT = float(os.environ.get("PRICE_PER_1M_OUTPUT", "4.00"))

SEED = 42
