"""Rate limiting: one token bucket per caller.

HOW A TOKEN BUCKET WORKS
    Each caller has a bucket holding up to ``burst`` tokens. Every request takes one token; tokens drip back
    at ``rate_per_min / 60`` per second. A burst of up to ``burst`` requests goes through at once, and after
    that the caller is held to the steady rate.

    >>> b = TokenBuckets(rate_per_min=60, burst=2)
    >>> [b.allow("alice", now=0.0) for _ in range(3)]   # 2 in the bucket, the 3rd is refused
    [True, True, False]
    >>> b.allow("alice", now=1.0)                        # one second later one token has dripped back
    True
    >>> b.allow("bob", now=1.0)                          # buckets are per caller
    True

    Callers are keyed by user (from the token) or, without identity, by client IP.
"""
from __future__ import annotations

import threading
import time


class TokenBuckets:
    def __init__(self, rate_per_min: float = 120, burst: int = 20):
        self.rate, self.burst = rate_per_min / 60.0, burst
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last refill time)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._state[key] = (tokens - 1.0 if ok else tokens, now)
            return ok
