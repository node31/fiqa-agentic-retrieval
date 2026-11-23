"""Identity tokens: a minimal JWT (HS256) written out by hand, so every step is visible.

WHAT A TOKEN IS
    Three base64url pieces joined by dots:  header.payload.signature
        header    {"alg": "HS256", "typ": "JWT"}
        payload   {"sub": "alice", "tenant": "acme", "role": "user", "iat": 1700000000, "exp": 1700003600}
        signature HMAC-SHA256(secret, "header.payload")
    Anyone can READ the payload (it is only base64), but nobody without the secret can CHANGE it: editing
    "tenant": "acme" to "globex" changes the bytes the signature covers, and verify() rejects it.

    >>> t = mint({"sub": "alice", "tenant": "acme", "role": "user"}, "s3cret", ttl_s=60, now=1000)
    >>> verify(t, "s3cret", now=1010)["tenant"]
    'acme'
    >>> verify(t, "wrong-secret", now=1010)
    Traceback (most recent call last):
    ...
    src.security.tokens.InvalidToken: bad signature

PITFALLS
    * Compare signatures with hmac.compare_digest (constant time), never ==, or the time a comparison
      takes leaks how many leading bytes matched.
    * Pin the algorithm. The classic JWT bug is trusting the header's "alg" (e.g. "none"); we refuse
      anything but HS256.
    * The secret must be long and random in production. AUTH_SECRET's default is for local demos only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

DEV_SECRET = "dev-only-secret-change-me"  # used when AUTH_SECRET is unset; fine on a laptop, never in production
ROLES = ("user", "admin")


class InvalidToken(Exception):
    """The token is malformed, forged, expired or uses an algorithm we don't accept."""


def secret() -> str:
    return os.environ.get("AUTH_SECRET") or DEV_SECRET


def _b64(data: bytes) -> str:
    # base64url without padding, as the JWT spec (RFC 7519) requires.
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(signing_input: str, key: str) -> str:
    return _b64(hmac.new(key.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest())


def mint(claims: dict, key: str | None = None, ttl_s: int = 3600, now: float | None = None) -> str:
    """Create a signed token. ``claims`` needs sub, tenant and role; iat/exp are added."""
    now = int(time.time() if now is None else now)
    body = {**claims, "iat": now, "exp": now + ttl_s}
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    pay = _b64(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    return f"{head}.{pay}.{_sign(f'{head}.{pay}', key or secret())}"


def verify(token: str, key: str | None = None, now: float | None = None) -> dict:
    """Return the claims of a valid token, or raise InvalidToken saying why."""
    try:
        head, pay, sig = token.split(".")
        header = json.loads(_unb64(head))
        claims = json.loads(_unb64(pay))
    except Exception as e:  # noqa: BLE001 - any parsing problem is the caller's problem
        raise InvalidToken("malformed token") from e
    if header.get("alg") != "HS256":
        raise InvalidToken(f"algorithm {header.get('alg')!r} not accepted")
    if not hmac.compare_digest(sig, _sign(f"{head}.{pay}", key or secret())):
        raise InvalidToken("bad signature")
    if (time.time() if now is None else now) >= claims.get("exp", 0):
        raise InvalidToken("expired")
    if not claims.get("sub") or not claims.get("tenant") or claims.get("role") not in ROLES:
        raise InvalidToken("missing sub, tenant or role")
    return claims
