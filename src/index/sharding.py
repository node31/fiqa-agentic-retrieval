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

HOW TO READ THIS FILE
    One function. Think of it as the "address book" of the cluster: given a doc id it
    names the one shard that stores (and is the only one allowed to write) that doc.
    Example: with n=3, the moved fraction when growing to 4 shards is ~3/4, because a doc
    stays put only if crc % 3 == crc % 4, which holds for about 1 id in 4.
"""
from __future__ import annotations

import zlib


def shard_of(doc_id: str, n_shards: int) -> int:
    """Stable shard index for ``doc_id``.

    >>> shard_of("12345", 3) == shard_of("12345", 3)
    True
    """
    # crc32 (a 32-bit checksum, normally used to detect corrupted data) turns the id's UTF-8
    # bytes into an unsigned int in [0, 2**32). It is fast, spreads ids roughly evenly, and
    # gives the same answer in every process, on every machine and Python version.
    # `% n_shards` maps that int to 0..n_shards-1.
    # PITFALL: the result depends on n_shards, so changing n means re-splitting the data
    # (see the module docstring for how much moves; there is no live resharding here).
    return zlib.crc32(doc_id.encode("utf-8")) % n_shards
