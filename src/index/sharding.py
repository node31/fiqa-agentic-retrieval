"""Shard assignment: which shard owns a document?

WHAT
    ``shard_of(doc_id, n)`` is a stable hash of the id, modulo the shard count.
    The router uses it to send a write to exactly one shard. The offline
    indexer uses the same function to split the corpus, so reads and writes agree.

WHY HASH PARTITIONING
    Uniform spread without a lookup table, and any process can compute it.
    The trade-off is that changing ``n`` moves about (n-1)/n of all docs.
    Consistent hashing moves only about 1/n. examples/15_hash_partitioning.py
    shows both. For a fixed 3-shard demo, plain modulo is the simplest correct
    choice.

PITFALL (a real, nasty bug)
    Python's built-in ``hash(str)`` is *salted per process* (PYTHONHASHSEED).
    The router and the shards would disagree about ownership, and writes would
    land on the wrong shard. CRC32 is deterministic everywhere.
"""
from __future__ import annotations

import zlib


def shard_of(doc_id: str, n_shards: int) -> int:
    """Stable shard index for ``doc_id``.

    >>> shard_of("12345", 3) == shard_of("12345", 3)
    True
    """
    return zlib.crc32(doc_id.encode("utf-8")) % n_shards
