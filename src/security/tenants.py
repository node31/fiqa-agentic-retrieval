"""Which tenant owns which document.

RULE (the demo setup)
    Two customers share one index. A corpus passage with an ODD numeric id belongs to "acme", an EVEN id to
    "globex". Ids that aren't numbers fall back to the parity of crc32(id), so every id has exactly one owner.
    A document written later through /upsert belongs to the WRITER's tenant (when identity is on), and keeps
    that owner for life, deletes included, so another tenant can't take the id over.

    >>> default_tenant("553031"), default_tenant("4")
    ('acme', 'globex')
"""
from __future__ import annotations

import zlib

TENANTS = ("acme", "globex")  # odd ids, even ids
ALL = "*"                     # what an admin's token carries: not limited to one tenant


def default_tenant(doc_id: str) -> str:
    """The owner of a document nobody has claimed: odd -> acme, even -> globex."""
    n = int(doc_id) if doc_id.isdigit() else zlib.crc32(doc_id.encode("utf-8"))
    return TENANTS[0] if n % 2 == 1 else TENANTS[1]
