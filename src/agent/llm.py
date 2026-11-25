"""LLM access layer: one tiny interface, three backends, budgets, caching, cost accounting.

INTERFACE
    ``llm.complete(system, user, max_output_tokens, budget) -> LLMResult``
    Text in, text out. Agents ask the model to answer with ONE JSON object
    (an "action"), and ``parse_json`` extracts it. That is the whole tool-calling
    protocol (docs/topics/11_llm_tool_calling.md).
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

GUARDRAILS (docs/topics/13_multi_agent_orchestration.md)
    ``Budget`` is shared by every agent working on ONE user query. It caps the
    number of LLM calls and total tokens. A call that would exceed it raises
    ``BudgetExceeded`` BEFORE any spend, and the orchestrator degrades gracefully.

CACHE
    Every response is stored under .llm_cache/<backend>/<sha256>.json, keyed on
    backend + model + prompts + max tokens. Re-running an eval costs nothing and
    is deterministic. A cache hit still reports the ORIGINAL token counts, so
    "cost per query" is the cost of the system, not of our cache.

HOW TO READ THIS FILE (top to bottom)
    1. ``LLMResult`` / ``cost_usd`` / ``estimate_tokens``: the data one call returns
       and how tokens turn into dollars.
    2. ``Budget``: the per-query spending cap.
    3. ``LLM.complete``: the ONE method every agent calls. The life of one call:
         a. budget.reserve(...)   assume the worst case (full max_output_tokens) and
                                  refuse up front if that could break the cap;
         b. cache lookup          sha256 of (backend, model, prompts, max tokens) ->
                                  a file under .llm_cache/; a hit skips the backend;
         c. self._call(...)       only on a miss: the backend-specific part;
         d. budget.record(res)    add the ACTUAL tokens used.
    4. The three backends (``FakeLLM``, ``CopilotLLM``, ``AzureLLM``); each only
       implements ``_call(system, user, max_output_tokens) -> (text, in_tok, out_tok)``.
    5. ``get_llm``: picks a backend from an argument or the LLM_BACKEND env var.
    6. ``parse_json``: pulls the model's JSON "action" out of its free-text reply.

JARGON
    token          the unit LLMs read/write and bill by (~4 English characters).
    system prompt  the fixed instructions ("you are a planner, reply with JSON ...").
    user prompt    the per-call input (the question plus search results so far).
    tool call      the model does NOT run anything itself. It writes JSON such as
                   {"action": "search", "query": "..."}; OUR code validates it, runs
                   the search, and pastes the results into the next prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import COPILOT_MODEL, PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT, ROOT

# One sub-folder per backend, so fake/copilot/azure answers never mix.
CACHE_DIR = ROOT / ".llm_cache"


@dataclass
class LLMResult:
    """What one ``complete`` call returns: the reply text plus what it cost and how long it took.

    ``cached=True`` means the answer came from disk; the token counts are still those of
    the original (paid) call, so cost reports describe the system, not the cache.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False
    # Input tokens the PROVIDER served from its own prompt cache (Azure/OpenAI report this in
    # usage.input_tokens_details.cached_tokens and bill them at a discount). Different from ``cached``
    # above, which is OUR disk cache. 0 for backends that do not report it.
    cached_input_tokens: int = 0

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


# WHY a dedicated exception: callers catch it and degrade (skip the remaining LLM steps and
# return the results they already have) instead of failing the whole request.
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Per-query guardrail shared across agents. Thread-safe (workers run in parallel).

    A "token budget" is a hard cap on input + output tokens for ONE user query, and the
    "call cap" (``max_calls``) bounds how many LLM calls that query may make. Both exist so
    a confused agent cannot loop forever or run up a bill.
    """

    def __init__(self, max_calls: int = 6, max_tokens: int = 12_000):
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.input_tokens = self.output_tokens = self.cached_input_tokens = 0
        # WHY a lock: check-then-increment in reserve() must be atomic, or two threads could
        # both see "1 call left" and both proceed.
        self._lock = threading.Lock()

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def reserve(self, est_input: int, max_output: int) -> None:
        """Raise BEFORE spending if this call could break the budget (worst case = max_output)."""
        # WHY worst case: we only learn the real output length after paying for it, so we
        # check against the most the call is ALLOWED to produce (max_output).
        # Note: input_tokens/output_tokens are NOT pre-charged here; record() adds the actual
        # numbers afterwards. Only the call counter is reserved up front.
        with self._lock:
            if self.calls + 1 > self.max_calls:
                raise BudgetExceeded(f"call cap {self.max_calls} reached")
            if self.tokens + est_input + max_output > self.max_tokens:
                raise BudgetExceeded(f"token budget {self.max_tokens} would be exceeded")
            self.calls += 1  # WHY reserve the call slot now: parallel workers must not overshoot

    def record(self, res: LLMResult) -> None:
        """Add the ACTUAL tokens of a finished call (cache hits included, see module docstring)."""
        with self._lock:
            self.input_tokens += res.input_tokens
            self.output_tokens += res.output_tokens
            self.cached_input_tokens += res.cached_input_tokens


class LLM:
    """Base class: caching + budget + timing around a backend-specific ``_call``.

    This is the "template method" pattern: ``complete`` is the same for every backend,
    subclasses only fill in ``_call``. Agents depend on this class, never on a backend.
    """

    # Subclasses override these two; both are part of the cache key.
    backend = "base"
    model = ""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        # What actually happened, for the "N requests sent, M from the disk cache" line the CLIs print:
        # a cached answer costs nothing NOW and sends no request, which is easy to mistake for "Azure was not used".
        self.requests_sent = self.cache_hits = 0
        self._count_lock = threading.Lock()

    def complete(self, system: str, user: str, max_output_tokens: int = 300,
                 budget: Budget | None = None) -> LLMResult:
        """One LLM call: reserve budget -> cache lookup -> backend call on a miss -> record tokens.

        Raises ``BudgetExceeded`` before anything is spent if the call could break the budget.
        Backend failures surface as RuntimeError / TimeoutError for the agent to handle.
        """
        # Step a: refuse BEFORE spending. The input size is an estimate (we have not called
        # the model yet), the output is the worst case.
        if budget:
            budget.reserve(estimate_tokens(system + user), max_output_tokens)
        # Step b: the "prompt hash cache". Identical inputs -> identical sha256 -> same file,
        # so a rerun of an eval reads answers from disk: free, fast and reproducible.
        # PITFALL: anything that changes the answer but is NOT in this key (for example
        # AzureLLM's reasoning effort) will silently reuse an old answer. Delete the cache
        # folder after changing such a setting.
        key = hashlib.sha256(json.dumps([self.backend, self.model, system, user, max_output_tokens]).encode()).hexdigest()
        path = CACHE_DIR / self.backend / f"{key}.json"
        if self.use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            res = LLMResult(d["text"], d["input_tokens"], d["output_tokens"], d["latency_s"], cached=True,
                            cached_input_tokens=d.get("cached_input_tokens", 0))
            with self._count_lock:
                self.cache_hits += 1
        else:
            # Step c: cache miss -> really call the backend and time it.
            t0 = time.perf_counter()
            with self._count_lock:
                self.requests_sent += 1
            text, tin, tout, *extra = self._call(system, user, max_output_tokens)  # extra: provider-cached tokens
            res = LLMResult(text, tin, tout, time.perf_counter() - t0, cached_input_tokens=extra[0] if extra else 0)
            if self.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                # res.__dict__ holds exactly the dataclass fields (text, tokens, latency, cached).
                path.write_text(json.dumps(res.__dict__), encoding="utf-8")
        # Step d: charge the ACTUAL tokens. A cache hit is charged too, on purpose: the
        # budget models what the system would cost without our cache.
        if budget:
            budget.record(res)
        return res

    def _call(self, system: str, user: str, max_output_tokens: int) -> tuple[str, int, int]:
        """Backend hook: return (reply text, input tokens, output tokens)."""
        raise NotImplementedError


def usage_line(*llms) -> str:
    """One line on what the LLM objects really did, e.g. for the end of a CLI run.

    >>> class L: backend, model, requests_sent, cache_hits = "azure", "gpt-x", 2, 1
    >>> usage_line(L())
    'LLM azure (gpt-x): 2 request(s) sent to the service, 1 answered from the disk cache (.llm_cache/azure/, no request)'
    """
    seen, parts = set(), []
    for llm in llms:
        llm = getattr(llm, "llm", llm)            # cascade tiers wrap the LLM
        if llm is None or id(llm) in seen:
            continue
        seen.add(id(llm))
        parts.append(f"LLM {llm.backend} ({llm.model or 'scripted'}): {llm.requests_sent} request(s) sent to the service, "
                     f"{llm.cache_hits} answered from the disk cache (.llm_cache/{llm.backend}/, no request)")
    return "; ".join(parts)


class FakeLLM(LLM):
    """Deterministic LLM: ``responder(system, user) -> text``. Token counts are estimated.

    The "responder" is any plain function; see ``fake_policies.heuristic_responder`` for
    the default, or pass a lambda in a test to script exact replies.
    """

    backend = "fake"

    def __init__(self, responder: Callable[[str, str], str], latency_s: float = 0.0):
        # WHY no cache: fake replies are free and instant, and caching them would let a
        # stale file from an old responder leak into a test.
        super().__init__(use_cache=False)
        self.responder, self.latency_s = responder, latency_s

    def _call(self, system, user, max_output_tokens):
        # Optional artificial delay, e.g. to test the orchestrator's timeouts and parallelism.
        if self.latency_s:
            time.sleep(self.latency_s)
        text = self.responder(system, user)
        return text, estimate_tokens(system + user), estimate_tokens(text)


def _copilot_exe() -> list[str]:
    """argv prefix that runs the Copilot CLI with arguments passed through UNCHANGED.

    PITFALL (we hit it): on Windows the npm install exposes ``copilot.cmd``, and cmd.exe cuts an
    argument at its first newline. The model then saw only "You are a planner..." and replied
    "Ready.". So we call the native binary (or node + the JS loader) behind the shim directly.
    Override with COPILOT_BIN.

    Lookup order: COPILOT_BIN -> plain ``copilot`` on non-Windows -> on Windows, the native
    ``copilot.exe`` or ``node npm-loader.js`` next to the npm shim -> ``copilot.exe`` on PATH.
    """
    # 1. Explicit override wins (useful when the install layout differs from what we expect).
    if os.environ.get("COPILOT_BIN"):
        return [os.environ["COPILOT_BIN"]]
    # 2. Linux/macOS: no .cmd shim, arguments with newlines pass through fine.
    if os.name != "nt":
        return ["copilot"]
    # 3. Windows: find the npm shim only to locate the real program installed next to it.
    shim = shutil.which("copilot.cmd")
    if shim:
        base = Path(shim).parent / "node_modules" / "@github"
        native = base / "copilot-win32-x64" / "copilot.exe"
        if native.exists():
            return [str(native)]
        loader = base / "copilot" / "npm-loader.js"
        if loader.exists() and shutil.which("node"):
            return [shutil.which("node"), str(loader)]
    # 4. A standalone (non-npm) install that put copilot.exe directly on PATH.
    exe = shutil.which("copilot.exe")
    if exe:
        return [exe]
    raise FileNotFoundError("Copilot CLI not found; set COPILOT_BIN to the copilot executable")


class CopilotLLM(LLM):
    """GitHub Copilot CLI as a plain text-completion backend (free for development).

    Each call starts the CLI as a "subprocess" (a separate program launched by Python via
    ``subprocess.run``), passes the whole prompt as one command-line argument, and reads
    the reply from its stdout. Slow (process start-up), but free and good enough to measure
    QUALITY. Never use it for latency numbers.
    """

    backend = "copilot"

    def __init__(self, model: str = COPILOT_MODEL, timeout_s: float = 180, **kw):
        super().__init__(**kw)
        self.model, self.timeout_s = model, timeout_s

    def _call(self, system, user, max_output_tokens):
        # WHY one glued prompt: the CLI has no separate "system" channel, so we concatenate
        # system + user. The CLI also has no max-output-tokens flag; the length limit is a
        # polite request in the text, not an enforced cap.
        prompt = (f"{system}\n\n---\n{user}\n\n(Answer directly. Do not use tools. "
                  f"Keep the answer under {max_output_tokens} tokens.)")
        # WHY empty cwd: the CLI must not see the repo. PITFALL (Windows): a CLI helper process can
        # outlive the call and keep the dir open, so deleting it raises; that must not fail the call.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            # Flags, in plain words: -p = one-shot prompt mode; -s = print only the answer;
            # --available-tools with no names = the model gets NO tools (it cannot read files
            # or run commands); no custom instructions, no built-in MCP servers, and never stop
            # to ask the user a question (nobody is there to answer).
            cmd = [*_copilot_exe(), "-p", prompt, "-s", "--model", self.model, "--available-tools",
                   "--no-custom-instructions", "--disable-builtin-mcps", "--no-ask-user"]
            # Translate subprocess-specific errors into the generic ones the agents already
            # catch (TimeoutError, RuntimeError), so agents need no copilot-specific code.
            try:
                out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                                     timeout=self.timeout_s)
            except subprocess.TimeoutExpired as e:  # PITFALL: NOT a subclass of TimeoutError
                raise TimeoutError(f"copilot CLI timed out after {self.timeout_s}s") from e
            except OSError as e:
                raise RuntimeError(f"copilot CLI could not run: {e!r}") from e
        if out.returncode != 0:
            raise RuntimeError(f"copilot CLI failed: {out.stderr[-500:]}")
        # WHY estimate tokens: the CLI's usage file also counts its own ~26K-token harness
        # prompt, which our system would not pay. We account for OUR prompt and answer only.
        return out.stdout.strip(), estimate_tokens(system + user), estimate_tokens(out.stdout)


# Characters a copy and paste (from a web page, a chat app, a rich-text editor) leaves at the ends of a value.
_INVISIBLE_ENDS = " \t\r\n\u00a0\u200b\u200c\u200d\u2060\ufeff\"'\u201c\u201d\u2018\u2019"


def non_ascii(name: str, value: str) -> str | None:
    """Why ``value`` cannot go into an HTTP request, or None if it can.

    The key travels in the ``Authorization`` header, and HTTP headers carry ASCII only, so httpx refuses a
    non-ASCII key with "'ascii' codec can't encode characters in position N-M" before anything is sent.
    The message names the characters (never the key itself).

    >>> non_ascii("KEY", "abc\u2026def\u00a0g")
    'KEY has 2 non-ASCII characters at positions 3, 7: U+2026 HORIZONTAL ELLIPSIS, U+00A0 NO-BREAK SPACE'
    >>> non_ascii("KEY", "abc") is None
    True
    """
    import unicodedata

    bad = [(i, ch) for i, ch in enumerate(value) if ord(ch) > 127]
    if not bad:
        return None
    names = ", ".join(dict.fromkeys(f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}" for _, ch in bad))
    where = ", ".join(str(i) for i, _ in bad[:8]) + (", ..." if len(bad) > 8 else "")
    return f"{name} has {len(bad)} non-ASCII character{'s' * (len(bad) > 1)} at position{'s' * (len(bad) > 1)} " \
           f"{where}: {names}"


def azure_setting(name: str) -> str:
    """Read one Azure setting, trim invisible characters and quotes off its ends, and refuse non-ASCII inside it.

    >>> os.environ["_T"] = "\u201cabc123\u201d\u00a0"; azure_setting("_T")
    'abc123'
    >>> del os.environ["_T"]
    """
    value = os.environ.get(name, "").strip(_INVISIBLE_ENDS)
    why = non_ascii(name, value)
    if why:
        raise ValueError(f"{why}. Re-type or re-paste the value into .env with a plain-text editor "
                         f"(python scripts/llm_check.py shows where each setting came from)")
    return value


class AzureLLM(LLM):
    """Azure AI Foundry deployment through the OpenAI Responses API. Costs money.

    The "Responses API" is OpenAI's newer request shape (``client.responses.create``): you
    pass ``instructions`` (our system prompt) and ``input`` (our user prompt) and read back
    ``output_text`` plus a ``usage`` block with exact token counts. It replaces the older
    Chat Completions "list of messages" shape. Settings come from env vars (.env).
    """

    backend = "azure"

    def __init__(self, confirm_spend: bool = False, **kw):
        # WHY refuse here, in the constructor: no AzureLLM object can exist without an explicit
        # opt-in, so a forgotten env var or a typo can never spend money by accident.
        if not confirm_spend:
            raise PermissionError("AzureLLM spends real money: pass confirm_spend=True (CLI: --confirm-spend)")
        super().__init__(**kw)
        # WHY import here: the openai package is only needed for this backend, so tests and
        # the fake/copilot backends work without it installed.
        from openai import OpenAI

        # On Azure the "model" name is the DEPLOYMENT name you chose in Foundry.
        self.model = azure_setting("AZURE_OPENAI_DEPLOYMENT")
        # PITFALL: never log or print the key. OpenAI() reads it only from this argument.
        self.client = OpenAI(base_url=azure_setting("AZURE_OPENAI_ENDPOINT"),
                             api_key=azure_setting("AZURE_OPENAI_API_KEY"), timeout=60, max_retries=2)
        # Reasoning models "think" (hidden tokens) before answering; "low" keeps that cheap.
        self.reasoning = os.environ.get("AZURE_REASONING_EFFORT", "low")

    def _call(self, system, user, max_output_tokens):
        # "none" = send no reasoning parameter at all (for models that do not accept it).
        # PITFALL: hidden reasoning tokens count against max_output_tokens. If the model
        # thinks too long, output_text can come back empty, parse_json raises ValueError,
        # and the agent falls back to its non-LLM results.
        kw = {"reasoning": {"effort": self.reasoning}} if self.reasoning != "none" else {}
        r = self.client.responses.create(model=self.model, instructions=system, input=user,
                                         max_output_tokens=max_output_tokens, **kw)
        # Unlike the other backends, these token counts are EXACT (billed by the service).
        # cached_tokens: the part of the input the service served from its prompt cache. It only kicks
        # in for prompts of 1,024+ tokens that share an identical prefix with a recent call, which is why
        # every prompt here puts the fixed system prompt first and appends new searches at the END.
        u = r.usage
        details = getattr(u, "input_tokens_details", None)
        return r.output_text or "", u.input_tokens, u.output_tokens, int(getattr(details, "cached_tokens", 0) or 0)


def get_llm(backend: str | None = None, confirm_spend: bool = False, responder=None) -> LLM:
    """Factory: build the backend named by ``backend`` or the LLM_BACKEND env var (default "fake").

    Choosing azure needs BOTH LLM_BACKEND=azure (or backend="azure") AND confirm_spend=True.
    ``responder`` only applies to the fake backend (a custom scripted policy).
    """
    # WHY default "fake": the safe, free choice. Real backends must be asked for explicitly.
    backend = backend or os.environ.get("LLM_BACKEND", "fake")
    if backend == "fake":
        # Imported lazily: the fake policies are only needed for this backend.
        from src.agent.fake_policies import heuristic_responder

        return FakeLLM(responder or heuristic_responder)
    if backend == "copilot":
        return CopilotLLM()
    if backend == "azure":
        return AzureLLM(confirm_spend=confirm_spend)
    raise ValueError(f"unknown LLM backend {backend!r}")


# Matches a Markdown code fence (```json ... ``` or ``` ... ```) and captures its body.
# re.S lets "." match newlines; ".*?" is non-greedy, so it stops at the FIRST closing fence.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates ```json fences and chatter).

    >>> parse_json('Sure!\\n```json\\n{"action": "finish", "doc_ids": []}\\n```')
    {'action': 'finish', 'doc_ids': []}

    Models are told to reply with JSON only, but they often add "Sure!" or wrap it in a
    code fence anyway. Raises ValueError (callers catch it) if no complete object is found.
    """
    # 1. If there is a code fence, look only inside the first one.
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # 2. The object starts at the first "{" (anything before it is chatter).
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    # 3. Walk forward counting brace depth until it returns to 0: that "}" closes the object.
    # WHY not a regex like r"\{.*\}": a greedy one would swallow trailing chatter containing
    # "}", a non-greedy one would stop inside a nested object.
    # PITFALL: braces INSIDE strings ({"query": "a } b"}) must not count, so we track whether
    # we are inside a "..." string (in_str) and whether the previous char was a backslash
    # (esc), so an escaped quote \" does not end the string.
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
                # json.loads still validates the slice; malformed JSON raises ValueError
                # (json.JSONDecodeError is a subclass), which callers already handle.
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in: {text[:200]!r}")
