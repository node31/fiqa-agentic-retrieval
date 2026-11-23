"""The toggles: one per security feature and guardrail, flipped at runtime by an admin, saved to disk.

    feature          values            what "on" means (docs/design/SECURITY.md has the details)
    identity         off | on          every request must carry a valid signed token (else 401)
    authz            off | on          role and tenant checks: admin-only endpoints, writes only to your own
                                       tenant's docs, search results checked against your tenant
    tenant_filter    off | on          search only your tenant's documents, inside the shards, before ranking
    pii_redaction    off | on          redact emails, phones, cards, ... in returned passage text
    audit            off | on          append every request to the hash-chained audit log
    guard_harmful    off | log | block harmful-intent questions ("how do I launder money")
    guard_injection  off | log | block prompt-injection text in the question and in returned passages
    guard_scope      off | log | block questions far from anything in the corpus (off-topic)
    rate_limit       off | log | block requests per caller per minute

    "log" = shadow mode: nothing is blocked, but the decision is reported and audited. That is how you
    measure a guardrail's false-alarm rate on real traffic before letting it refuse anyone.

Defaults: everything off, so the service behaves exactly as before (and every eval number stays the same).
Env vars SECURITY_<NAME>=on|log|block set the starting values; a state file, once written, wins (like the
router's active_model), so a restart keeps what the admin chose.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ON_OFF = ("off", "on")
GUARD = ("off", "log", "block")
FEATURES: dict[str, tuple[str, ...]] = {
    "identity": ON_OFF, "authz": ON_OFF, "tenant_filter": ON_OFF, "pii_redaction": ON_OFF, "audit": ON_OFF,
    "guard_harmful": GUARD, "guard_injection": GUARD, "guard_scope": GUARD, "rate_limit": GUARD,
}
NEEDS_IDENTITY = ("authz", "tenant_filter")  # without a caller identity there is no tenant to check against


class SecurityConfig:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.values = {k: os.environ.get(f"SECURITY_{k.upper()}", "off") for k in FEATURES}
        for k, v in self.values.items():
            if v not in FEATURES[k]:
                raise ValueError(f"SECURITY_{k.upper()}={v!r}: expected one of {FEATURES[k]}")
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text())
            self.values.update({k: v for k, v in saved.items() if k in FEATURES and v in FEATURES[k]})

    def get(self, name: str) -> str:
        return self.values[name]

    def on(self, name: str) -> bool:
        """True when the feature is active. authz/tenant_filter also need identity to be on."""
        if name in NEEDS_IDENTITY and self.values["identity"] != "on":
            return False
        return self.values[name] != "off"

    def set(self, changes: dict[str, str]) -> dict[str, tuple[str, str]]:
        """Validate and apply {feature: value}; returns {feature: (old, new)} for what changed."""
        for k, v in changes.items():
            if k not in FEATURES:
                raise ValueError(f"unknown feature {k!r}; known: {', '.join(FEATURES)}")
            if v not in FEATURES[k]:
                raise ValueError(f"{k}: {v!r} is not one of {FEATURES[k]}")
        with self._lock:
            diff = {k: (self.values[k], v) for k, v in changes.items() if self.values[k] != v}
            self.values.update(changes)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.values, indent=1))
        return diff

    def describe(self) -> dict:
        """The admin view: each feature's value, allowed values, and whether it is actually in effect."""
        return {k: {"value": self.values[k], "allowed": list(FEATURES[k]), "effective": self.on(k),
                    **({"note": "needs identity on"} if k in NEEDS_IDENTITY and self.values[k] == "on"
                       and not self.on(k) else {})}
                for k in FEATURES}
