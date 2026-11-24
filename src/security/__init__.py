"""Security and LLM guardrails for the retrieval service. Every feature has its own runtime toggle.

Read docs/topics/17_security.md first: threat model, how each piece works, how it is checked.

    tenants.py     who owns which document (odd doc ids -> "acme", even -> "globex")
    tokens.py      signed identity tokens (JWT, HS256), written out by hand
    pii.py         find and redact personal data in text
    audit.py       append-only, hash-chained audit log
    guardrails.py  harmful-intent, prompt-injection and topic-scope checks
    ratelimit.py   token bucket per caller
    config.py      the toggles, persisted, flipped at runtime by an admin
    policy.py      the enforcement point the router's HTTP layer calls
"""
