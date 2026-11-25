"""The enforcement point: what the router's HTTP layer asks before and after every request.

The router (src/service/router.py, create_app) calls into ONE object, ``Policy``, so every security
decision lives here, next to the toggle that controls it:

    request ─► identify ──► rate limit ──► guard the question ──► Router.search(tenant=...) ──► post-check
               (identity)   (rate_limit)   (guard_harmful,          (tenant_filter: the shards     (authz, guard_scope,
                401                         guard_injection) 403     search only your docs)          guard_injection on
                                                                                                    passages, pii_redaction)
                                                                                         └──────────► audit (every outcome)

A refusal is raised as ``Denied(status, detail)``; the router turns it into an HTTP error, and the
audit line is written either way (a refused request is exactly what an auditor wants to see).

    >>> import tempfile, pathlib
    >>> from src.security import tokens
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> p = Policy(SecurityConfig(d / "s.json"), AuditLog(d / "a.jsonl"))
    >>> p.identify(None, "1.2.3.4").user              # identity off: anyone may call, as "anonymous"
    'anonymous'
    >>> _ = p.config.set({"identity": "on", "tenant_filter": "on"})
    >>> p.identify(None, "1.2.3.4")
    Traceback (most recent call last):
    ...
    src.security.policy.Denied: (401, 'missing bearer token')
    >>> alice = p.identify("Bearer " + tokens.mint({"sub": "alice", "tenant": "acme", "role": "user"}), "1.2.3.4")
    >>> p.search_tenant(alice)                          # the shards will only see acme's docs
    'acme'
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.security import guardrails, pii, tokens
from src.security.audit import AuditLog
from src.security.config import SecurityConfig
from src.security.ratelimit import TokenBuckets
from src.security.tenants import ALL


class Denied(Exception):
    """A refused request: HTTP ``status`` and a JSON-able ``detail``."""

    def __init__(self, status: int, detail):
        super().__init__(status, detail)
        self.status, self.detail = status, detail


@dataclass
class Caller:
    user: str
    tenant: str | None  # None = anonymous (identity off, no token)
    role: str
    ip: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def key(self) -> str:
        """Who a rate limit counts against: the user, or the IP address for anonymous callers."""
        return f"user:{self.user}" if self.tenant else f"ip:{self.ip}"


@dataclass
class Verdict:
    """What the checks decided for one request; ends up in the response and the audit line."""
    findings: list[dict] = field(default_factory=list)
    blocked: str | None = None  # the rule that refused the request, if any

    def note(self, finding: dict | None, mode: str) -> dict | None:
        if finding:
            self.findings.append({**finding, "mode": mode})
        return finding


class Policy:
    def __init__(self, config: SecurityConfig, audit: AuditLog, rate: TokenBuckets | None = None,
                 scope_min: float | None = None):
        self.config, self.audit_log = config, audit
        self.rate = rate or TokenBuckets()
        self.scope_min = scope_min

    # ------------------------------------------------------------------ 1. identity
    def identify(self, authorization: str | None, ip: str = "", must_be_valid: bool = False) -> Caller:
        """Who is calling. Raises Denied(401) when identity is on (or ``must_be_valid``) and the token is
        missing or bad. With identity off a missing token means "anonymous"; a token that IS sent is
        still verified, so a client can test its token before an admin turns identity on."""
        strict = must_be_valid or self.config.on("identity")
        if not authorization:
            if strict:
                raise Denied(401, "missing bearer token")
            return Caller("anonymous", None, "user", ip)
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Denied(401, "expected 'Authorization: Bearer <token>'")
        try:
            c = tokens.verify(token.strip())
        except tokens.InvalidToken as e:
            raise Denied(401, f"invalid token: {e}") from None
        return Caller(c["sub"], c["tenant"], c["role"], ip)

    # ------------------------------------------------------------------ 2. authorisation
    @staticmethod
    def require_admin(caller: Caller) -> None:
        """Admin endpoints ALWAYS need an admin token, whatever the toggles say. Otherwise anyone could
        call POST /admin/security and switch every protection off."""
        if not caller.is_admin:
            raise Denied(403, "admin role required")

    def search_tenant(self, caller: Caller) -> str | None:
        """The tenant the shards should filter to, or None for "everything" (filter off, or an admin)."""
        if self.config.on("tenant_filter") and caller.tenant and caller.tenant != ALL:
            return caller.tenant
        return None

    def write_scope(self, caller: Caller) -> tuple[str | None, str | None]:
        """(owner for a NEW doc, tenant to enforce on an existing doc). Enforcement needs authz on."""
        owner = caller.tenant if caller.tenant and caller.tenant != ALL else None
        enforce = owner if self.config.on("authz") else None
        return owner, enforce

    # ------------------------------------------------------------------ 3. rate limit
    def check_rate(self, caller: Caller, v: Verdict) -> None:
        mode = self.config.get("rate_limit")
        if mode == "off" or self.rate.allow(caller.key):
            return
        v.note({"rule": "rate_limit", "key": caller.key}, mode)
        if mode == "block":
            v.blocked = "rate_limit"
            raise Denied(429, {"error": "rate limit exceeded", "findings": v.findings})

    # ------------------------------------------------------------------ 4. guard the question
    def check_question(self, query: str, v: Verdict) -> None:
        """Harmful intent and prompt injection in the user's text. Block mode raises Denied(403)."""
        for name, check in (("guard_harmful", guardrails.check_harmful),
                            ("guard_injection", guardrails.check_injection)):
            mode = self.config.get(name)
            if mode != "off" and v.note(check(query), mode) and mode == "block":
                v.blocked = v.findings[-1]["rule"]
                raise Denied(403, {"error": "request refused by a guardrail", "findings": v.findings})

    # ------------------------------------------------------------------ 5. check the answer
    def post_search(self, caller: Caller, out: dict, v: Verdict) -> dict:
        """Apply authz, scope, passage-injection and PII rules to a Router.search response (in place)."""
        results = out["results"]
        # authz, defence in depth: even if the shards were asked to filter, never return a doc owned by
        # another tenant. With tenant_filter OFF this is the ONLY protection, and it shows why filtering
        # after ranking is the wrong place: the caller gets fewer than k results (sometimes none).
        removed = 0
        if self.config.on("authz") and caller.tenant and caller.tenant != ALL:
            keep = [r for r in results if r.get("tenant") == caller.tenant]
            removed, results = len(results) - len(keep), keep
        # Topic scope: the best dense score says how close the question is to anything in the corpus.
        mode = self.config.get("guard_scope")
        if mode != "off":
            f = guardrails.check_scope(out.get("top_dense"), out.get("model", ""), self.scope_min)
            if v.note(f, mode) and mode == "block":
                v.blocked, results = "topic_scope", []
            # Mode "skipped", never "block": the question passes, but the response shows the check did not run.
            v.note(guardrails.scope_skipped(out.get("top_dense"), out.get("model", ""), self.scope_min), "skipped")
        # Passages are untrusted (forum posts). Log mode flags them; block mode cuts the instruction-like
        # text out before the passage reaches anyone (an agent's LLM, typically).
        mode = self.config.get("guard_injection")
        if mode != "off":
            for r in results:
                if "text" in r and guardrails.find_injection(r["text"]):
                    v.note({"rule": "prompt_injection", "where": "passage", "doc_id": r["doc_id"]}, mode)
                    if mode == "block":
                        r["text"], _ = guardrails.neutralize(r["text"])
        # PII: redact what is returned. The index keeps the original text (search quality is unchanged);
        # only what leaves the service is masked.
        redacted: dict[str, int] = {}
        if self.config.on("pii_redaction"):
            for r in results:
                if r.get("text"):
                    r["text"], counts = pii.redact(r["text"])
                    for kind, n in counts.items():
                        redacted[kind] = redacted.get(kind, 0) + n
        out["results"] = results
        out["security"] = {"user": caller.user, "tenant": caller.tenant, "filtered_to": self.search_tenant(caller),
                           "removed_by_authz": removed, "pii_redacted": redacted, "findings": v.findings,
                           **({"blocked": v.blocked} if v.blocked else {})}
        return out

    def active(self) -> bool:
        """True if any toggle is not off (then responses carry a ``security`` block)."""
        return any(val != "off" for val in self.config.values.values())

    # ------------------------------------------------------------------ 6. audit
    def audit(self, caller: Caller | None, action: str, decision: str, status: int, t0: float,
              always: bool = False, **details) -> None:
        """Write one audit line if audit is on (``always``: toggle changes are logged regardless).
        The question is stored REDACTED: the audit log must not become a second copy of personal data."""
        if not (always or self.config.on("audit")):
            return
        if "query" in details:
            details["query"] = pii.redact(details["query"])[0]
        self.audit_log.append({
            "action": action, "user": caller.user if caller else None, "tenant": caller.tenant if caller else None,
            "role": caller.role if caller else None, "ip": caller.ip if caller else None,
            "decision": decision, "status": status, "latency_ms": round((time.perf_counter() - t0) * 1e3, 2),
            **details})
