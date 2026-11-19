"""LLM access layer: one tiny interface, three backends, budgets, caching, cost accounting.

INTERFACE
    ``llm.complete(system, user, max_output_tokens, budget) -> LLMResult``
    Text in, text out. Agents ask the model to answer with ONE JSON object
    (an "action"), and ``parse_json`` extracts it. That is the whole tool-calling
    protocol (docs/11_llm_tool_calling.md, docs/adr/0008-json-action-protocol.md).
    Native function-calling APIs are this same idea with the JSON schema enforced
    server-side. Using plain JSON keeps one code path across all three backends.

BACKENDS
    fake     scripted, deterministic, free. Tests, toys and the --dry-run cost estimate.
    copilot  GitHub Copilot CLI in prompt mode (``copilot -p``) with NO tools, NO
             MCP servers and NO repo access (empty temp cwd). Free for development.
             Latency is dominated by process start-up (~30-40 s), so it is useless
             for latency numbers.
    azure    Azure AI Foundry deployment via the OpenAI Responses API. COSTS MONEY.
             Refuses to construct without ``confirm_spend=True``.

GUARDRAILS (docs/13_multi_agent_orchestration.md)
    ``Budget`` is shared by every agent working on ONE user query. It caps the
    number of LLM calls and total tokens. A call that would exceed it raises
    ``BudgetExceeded`` BEFORE any spend, and the orchestrator degrades gracefully.

CACHE
    Every response is stored under .llm_cache/<backend>/<sha256>.json, keyed on
    backend + model + prompts + max tokens. Re-running an eval costs nothing and
    is deterministic. A cache hit still reports the ORIGINAL token counts, so
    "cost per query" is the cost of the system, not of our cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import COPILOT_MODEL, PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT, ROOT

CACHE_DIR = ROOT / ".llm_cache"


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens)


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Tokens -> dollars with the (placeholder) prices from config/.env.

    >>> round(cost_usd(1_000_000, 0), 2) == round(PRICE_PER_1M_INPUT, 2)
    True
    """
    return input_tokens / 1e6 * PRICE_PER_1M_INPUT + output_tokens / 1e6 * PRICE_PER_1M_OUTPUT


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English. Good enough for budgets and dry runs."""
    return max(1, len(text) // 4)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Per-query guardrail shared across agents. Thread-safe (workers run in parallel)."""

    def __init__(self, max_calls: int = 6, max_tokens: int = 12_000):
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def reserve(self, est_input: int, max_output: int) -> None:
        """Raise BEFORE spending if this call could break the budget (worst case = max_output)."""
        with self._lock:
            if self.calls + 1 > self.max_calls:
                raise BudgetExceeded(f"call cap {self.max_calls} reached")
            if self.tokens + est_input + max_output > self.max_tokens:
                raise BudgetExceeded(f"token budget {self.max_tokens} would be exceeded")
            self.calls += 1  # WHY reserve the call slot now: parallel workers must not overshoot

    def record(self, res: LLMResult) -> None:
        with self._lock:
            self.input_tokens += res.input_tokens
            self.output_tokens += res.output_tokens


class LLM:
    """Base class: caching + budget + timing around a backend-specific ``_call``."""

    backend = "base"
    model = ""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def complete(self, system: str, user: str, max_output_tokens: int = 300,
                 budget: Budget | None = None) -> LLMResult:
        if budget:
            budget.reserve(estimate_tokens(system + user), max_output_tokens)
        key = hashlib.sha256(json.dumps([self.backend, self.model, system, user, max_output_tokens]).encode()).hexdigest()
        path = CACHE_DIR / self.backend / f"{key}.json"
        if self.use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            res = LLMResult(d["text"], d["input_tokens"], d["output_tokens"], d["latency_s"], cached=True)
        else:
            t0 = time.perf_counter()
            text, tin, tout = self._call(system, user, max_output_tokens)
            res = LLMResult(text, tin, tout, time.perf_counter() - t0)
            if self.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(res.__dict__), encoding="utf-8")
        if budget:
            budget.record(res)
        return res

    def _call(self, system: str, user: str, max_output_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError


class FakeLLM(LLM):
    """Deterministic LLM: ``responder(system, user) -> text``. Token counts are estimated."""

    backend = "fake"

    def __init__(self, responder: Callable[[str, str], str], latency_s: float = 0.0):
        super().__init__(use_cache=False)
        self.responder, self.latency_s = responder, latency_s

    def _call(self, system, user, max_output_tokens):
        if self.latency_s:
            time.sleep(self.latency_s)
        text = self.responder(system, user)
        return text, estimate_tokens(system + user), estimate_tokens(text)


class CopilotLLM(LLM):
    """GitHub Copilot CLI as a plain text-completion backend (free for development)."""

    backend = "copilot"

    def __init__(self, model: str = COPILOT_MODEL, timeout_s: float = 180, **kw):
        super().__init__(**kw)
        self.model, self.timeout_s = model, timeout_s

    def _call(self, system, user, max_output_tokens):
        prompt = (f"{system}\n\n---\n{user}\n\n(Answer directly. Do not use tools. "
                  f"Keep the answer under {max_output_tokens} tokens.)")
        with tempfile.TemporaryDirectory() as tmp:  # WHY empty cwd: the CLI must not see the repo
            cmd = ["copilot", "-p", prompt, "-s", "--model", self.model, "--available-tools",
                   "--no-custom-instructions", "--disable-builtin-mcps", "--no-ask-user"]
            if os.name == "nt":
                cmd[0] = "copilot.cmd"
            out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                                 timeout=self.timeout_s)
        if out.returncode != 0:
            raise RuntimeError(f"copilot CLI failed: {out.stderr[-500:]}")
        # WHY estimate tokens: the CLI's usage file also counts its own ~26K-token harness
        # prompt, which our system would not pay. We account for OUR prompt and answer only.
        return out.stdout.strip(), estimate_tokens(system + user), estimate_tokens(out.stdout)


class AzureLLM(LLM):
    """Azure AI Foundry deployment through the OpenAI Responses API. Costs money."""

    backend = "azure"

    def __init__(self, confirm_spend: bool = False, **kw):
        if not confirm_spend:
            raise PermissionError("AzureLLM spends real money: pass confirm_spend=True (CLI: --confirm-spend)")
        super().__init__(**kw)
        from openai import OpenAI

        self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        # PITFALL: never log or print the key. OpenAI() reads it only from this argument.
        self.client = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
                             api_key=os.environ["AZURE_OPENAI_API_KEY"], timeout=60, max_retries=2)
        self.reasoning = os.environ.get("AZURE_REASONING_EFFORT", "low")

    def _call(self, system, user, max_output_tokens):
        kw = {"reasoning": {"effort": self.reasoning}} if self.reasoning != "none" else {}
        r = self.client.responses.create(model=self.model, instructions=system, input=user,
                                         max_output_tokens=max_output_tokens, **kw)
        u = r.usage
        return r.output_text or "", u.input_tokens, u.output_tokens


def get_llm(backend: str | None = None, confirm_spend: bool = False, responder=None) -> LLM:
    backend = backend or os.environ.get("LLM_BACKEND", "fake")
    if backend == "fake":
        from src.agent.fake_policies import heuristic_responder

        return FakeLLM(responder or heuristic_responder)
    if backend == "copilot":
        return CopilotLLM()
    if backend == "azure":
        return AzureLLM(confirm_spend=confirm_spend)
    raise ValueError(f"unknown LLM backend {backend!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates ```json fences and chatter).

    >>> parse_json('Sure!\\n```json\\n{"action": "finish", "doc_ids": []}\\n```')
    {'action': 'finish', 'doc_ids': []}
    """
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    depth = 0
    in_str = esc = False
    for i, ch in enumerate(text[start:], start):  # WHY scan: find the matching closing brace
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in: {text[:200]!r}")
