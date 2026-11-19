# ch07: Part B in miniature: shards, router, versioned writes, a WAL

Guide: `docs/from_zero/07_part_b_distributed.md`.

> 🎯 **Why:** the hard part of a distributed search engine isn't HTTP, it's **correctness**: identical results to a
> single machine, writes that are safe to retry, and no lost writes after a crash. Doing it in one process first means
> any bug is a logic bug, not network noise.

## Steps (each has its own checks: run them with `-k`)
1. `shard_of` (`-k shard_of`): crc32 % n. **Not** Python's `hash()` (why? see `PAPER.md`).
2. `Shard` + `router_search` (`-k router`): each shard reports `stats(terms)` = its N, total length and df per term;
   the router **sums** them (global stats), each shard scores with those global numbers and returns its top-k,
   and the router merges by score. Must equal one BM25 over all docs.
3. The local-IDF bug (`-k local`): with `use_global_stats=False` each shard uses its own stats, and two *identical* docs on
   different shards get *different* scores. The checker builds such a case.
4. `VersionedStore` (`-k versioned`): upsert/delete with versions + tombstones (toy 16's exact sequence).
5. `WAL` (`-k wal`): append JSON lines + fsync; replay; **truncate a torn last line when opening**. The checker crashes it
   twice in a row (the bug we found in the real project, see the guide §7.4).

`python exercises\ch07_shards\mini_shards.py [--full]` runs a demo of all five (`--full`: 3 shards over all 57,638 docs).

## Hints
- Reuse your ch03 `tokenize` and `idf` (`use_chapter("ch03_bm25")` is already at the top of the file).
- A Shard needs, per doc: its tf per term and its length. Score with the GLOBAL `n_docs` and `avgdl = total_len / n_docs`.
- Ties: sort merged results by (-score, doc_id).
- WAL truncation: open in `"r+b"`, read all bytes; if they don't end with `b"\n"`, `truncate()` to just after the last `b"\n"`.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch07_shards` | the 24-doc corpus on 2 and 3 shards = one BM25; the local-IDF bug; toy 16's writes; crash-twice WAL | seconds |
| full | `python -m pytest exercises\ch07_shards --full` | crc32 % 3 splits 57,638 docs evenly; 3 shards + global stats give the single-node top-10 and scores on all 500 dev queries; local stats visibly change rankings | ~3–4 min |
