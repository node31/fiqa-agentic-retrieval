"""Mint a demo identity token (a signed JWT) to call the router as a given user.

    python -m scripts.mint_token alice                # acme user   (tenant of the ODD doc ids)
    python -m scripts.mint_token bob                  # globex user (tenant of the EVEN doc ids)
    python -m scripts.mint_token admin                # admin, every tenant ("*"); needed for /admin/security
    python -m scripts.mint_token carol --tenant acme --role user --ttl 600
    python -m scripts.mint_token alice --show         # also print the decoded claims

Then send it as a header:  curl -H "Authorization: Bearer <token>" ...   or paste it into Swagger's Authorize.

The token is signed with AUTH_SECRET (env), the same secret the router verifies with. Without AUTH_SECRET
both sides fall back to a dev-only default, which is fine on a laptop and never acceptable in production:
anyone who knows the secret can mint an admin token. In a real system a login service (an identity
provider) holds the key and issues tokens after checking a password or SSO; this script stands in for it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security import tokens  # noqa: E402
from src.security.tenants import ALL  # noqa: E402

# The demo users from docs/design/SECURITY.md.
USERS = {"alice": ("acme", "user"), "bob": ("globex", "user"), "admin": (ALL, "admin")}


def token_for(user: str, tenant: str | None = None, role: str | None = None, ttl_s: int = 3600) -> str:
    t, r = USERS.get(user, (None, "user"))
    tenant, role = tenant or t, role or r
    if not tenant:
        raise SystemExit(f"unknown user {user!r}: pass --tenant (acme or globex)")
    return tokens.mint({"sub": user, "tenant": tenant, "role": role}, ttl_s=ttl_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("user", help="alice | bob | admin, or any name with --tenant")
    ap.add_argument("--tenant", help="acme | globex | * (default: the demo user's)")
    ap.add_argument("--role", choices=tokens.ROLES)
    ap.add_argument("--ttl", type=int, default=3600, help="seconds until it expires (default 3600)")
    ap.add_argument("--show", action="store_true", help="also print the decoded claims (to stderr)")
    a = ap.parse_args(argv)
    tok = token_for(a.user, a.tenant, a.role, a.ttl)
    print(tok)
    if a.show:
        print(json.dumps(tokens.verify(tok), indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
