"""Check the LLM settings with ONE real, uncached call before running the agents (costs a fraction of a cent).

    python scripts/llm_check.py                    # shows the settings and what is wrong with them; no call
    python scripts/llm_check.py --confirm-spend    # ...then makes one tiny Azure call and prints its tokens

WHY THIS EXISTS
    The agents read the Azure settings from the environment, and src/config.py fills the environment from .env
    with override=False: a variable that is already set in your shell (an old ``export AZURE_OPENAI_API_KEY=...``
    in ~/.zshrc, say) silently beats the value in .env. Answers are also cached on disk (.llm_cache/azure/), so a
    question you asked before is answered without any request. Either can make a run look as if it "did not call
    Azure". This script prints where each setting came from, refuses placeholder values, bypasses the cache, and
    turns the usual errors into a sentence about what to fix.

    >>> problems({"AZURE_OPENAI_ENDPOINT": "https://x.example/openai/v1", "AZURE_OPENAI_DEPLOYMENT": "<name>",
    ...           "AZURE_OPENAI_API_KEY": ""})
    ['AZURE_OPENAI_ENDPOINT should end with /openai/v1/ (it ends with ...openai/v1)', 'AZURE_OPENAI_DEPLOYMENT still looks like a placeholder: <name>', 'AZURE_OPENAI_API_KEY is empty']
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY", "AZURE_REASONING_EFFORT"]
HINTS = {
    401: "the key is wrong, or the key belongs to a different resource than the endpoint",
    403: "the key is valid but has no access to this resource or deployment",
    404: "the endpoint path or the deployment name is wrong (the deployment name is the one you chose in Foundry, "
         "and the endpoint must end with /openai/v1/)",
    400: "the request was refused; if the message mentions 'reasoning', the model is not a reasoning model: "
         "set AZURE_REASONING_EFFORT=none in .env",
    429: "rate limited or out of quota on this deployment; wait a minute or raise its TPM limit",
}


def problems(values: dict) -> list[str]:
    """Plain-words list of what is wrong with the settings (empty = they look usable)."""
    out = []
    for name in NAMES[:3]:
        v = values.get(name) or ""
        if not v:
            out.append(f"{name} is empty")
        elif "<" in v or ">" in v:
            out.append(f"{name} still looks like a placeholder: {v}")
    ep = values.get("AZURE_OPENAI_ENDPOINT") or ""
    if ep and "<" not in ep and not ep.endswith("/openai/v1/"):
        out.insert(0, f"AZURE_OPENAI_ENDPOINT should end with /openai/v1/ (it ends with ...{ep[-9:]})")
    return out


def ping(llm) -> tuple[bool, str]:
    """One tiny call that bypasses the disk cache. (True, "OK: ...") or (False, "what failed and the likely cause")."""
    use_cache, llm.use_cache = llm.use_cache, False   # a check must reach the service, never the disk cache
    t0 = time.perf_counter()
    try:
        res = llm.complete("You are a health check. Reply with the single word OK.", "Are you there?",
                           max_output_tokens=200)
    except Exception as e:  # noqa: BLE001 - every failure becomes a message about what to fix
        status = getattr(e, "status_code", None)
        msg = f"FAILED after {time.perf_counter() - t0:.1f} s: {type(e).__name__}: {str(e)[:300]}"
        if status in HINTS:
            msg += f"\n  likely cause: {HINTS[status]}"
        elif "Connection" in type(e).__name__:
            msg += "\n  likely cause: the endpoint host cannot be reached (typo in the URL, VPN, or no network)"
        return False, msg + "\n  python scripts/llm_check.py shows where each setting came from"
    finally:
        llm.use_cache = use_cache
    msg = (f"OK: {llm.model} answered {res.text.strip()[:40]!r} in {res.latency_s:.1f} s, "
           f"{res.input_tokens} input + {res.output_tokens} output tokens (${res.cost_usd:.6f} at the prices in .env)")
    if res.output_tokens and not res.text.strip():
        msg += ("\n  note: no text came back, only hidden reasoning tokens; if the agents' answers come back empty, "
                "lower AZURE_REASONING_EFFORT")
    return True, msg


def show(name: str, value: str | None, source: str) -> str:
    if not value:
        return f"  {name:24s} (not set)"
    if name.endswith("KEY"):
        value = f"...{value[-4:]} ({len(value)} chars)" if len(value) > 8 else "(too short to be a key)"
    return f"  {name:24s} {value}   [from {source}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm-spend", action="store_true", help="make the one real call (a fraction of a cent)")
    a = ap.parse_args(argv)

    shell = {n: os.environ.get(n) for n in NAMES}          # before src.config loads .env
    try:
        from dotenv import dotenv_values
        dotfile = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    except ImportError:
        dotfile = {}
    import src.config  # noqa: F401  - fills os.environ from .env (override=False), exactly as the agents do

    print(f".env: {ROOT / '.env'} ({'found' if (ROOT / '.env').exists() else 'MISSING: copy .env.example to .env'})")
    values = {}
    for n in NAMES:
        values[n] = os.environ.get(n)
        src = "your shell (it overrides .env)" if shell[n] else ".env" if dotfile.get(n) else "default"
        print(show(n, values[n], src))
        if shell[n] and dotfile.get(n) and shell[n] != dotfile.get(n):
            print(f"    ! your shell's {n} differs from the one in .env, and the shell wins: run `unset {n}`")
    found = problems(values)
    for p in found:
        print(f"  ! {p}")
    if found:
        print("fix the settings above in .env, then run this again")
        return 2
    if not a.confirm_spend:
        print("settings look usable; add --confirm-spend to make one real call")
        return 0

    from src.agent.llm import get_llm

    ok, msg = ping(get_llm("azure", confirm_spend=True))
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
