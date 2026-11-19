"""Shared helpers for the toy examples: a tiny corpus, narrated printing, optional models.

Every toy:
  * runs offline in a few seconds (models are optional; we fall back gracefully),
  * prints WHAT it does and WHY, step by step,
  * ends with a TRY THIS block of experiments to run next.
Run one with:  python examples/02_bm25_by_hand.py
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:  # Windows consoles default to cp1252; make box-drawing/greek letters printable
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# A 10-passage "FiQA in miniature": small enough to check every number by hand.
TOY_DOCS = {
    "d1": "Short selling means borrowing shares and selling them, hoping to buy them back cheaper.",
    "d2": "When you short a stock your loss is unlimited because the price can keep rising.",
    "d3": "Paying off your mortgage early gives a guaranteed return equal to its interest rate.",
    "d4": "Index funds give cheap diversification across the whole stock market.",
    "d5": "A Roth IRA is funded with after-tax money; qualified withdrawals are tax free.",
    "d6": "A traditional 401k lowers taxable income now; withdrawals in retirement are taxed.",
    "d7": "Keep an emergency fund of three to six months of expenses in a savings account.",
    "d8": "Selling a stock for more than you paid creates a capital gain, which is taxed.",
    "d9": "Mortgage interest may be deductible if you itemize, which lowers your tax bill.",
    "d10": "Betting that a share price will fall is called shorting; brokers charge a borrow fee.",
}


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def step(n, text: str) -> None:
    first, *rest = text.split("\n")  # wrap the headline; keep explicit extra lines (already indented) as-is
    print(f"\n--- Step {n}: " + textwrap.fill(first, 72, subsequent_indent="    "))
    for line in rest:
        print(line)


def note(text: str) -> None:
    print(textwrap.fill("  -> " + text, 78, subsequent_indent="     "))


def try_this(*ideas: str) -> None:
    print("\nTRY THIS:")
    for i, idea in enumerate(ideas, 1):
        print(textwrap.fill(f"  {i}. {idea}", 78, subsequent_indent="     "))


def load_st_model(name: str):
    """Return a SentenceTransformer if it is cached locally, else None (toys must work offline)."""
    if os.environ.get("TOYS_NO_MODELS"):
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device="cpu")
    except Exception as e:  # noqa: BLE001 - offline / not cached: degrade gracefully
        print(f"  (model {name} unavailable: {type(e).__name__}; using a fallback)")
        return None
