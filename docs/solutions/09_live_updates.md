# Solutions: live updates

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** A plain removal forgets that the doc ever existed at version v. A delayed older upsert (version < v) would then look 'new' and resurrect the document. The tombstone remembers (v, deleted) so the version check rejects it.

**2.** The first to take the write lock wins; the second is a no-op ('stale_or_duplicate') because the version is not newer. That's OK only if clients guarantee one text per (id, version). It's the client's version contract, same as Elasticsearch external versioning.

**3.** test_freshness_write_searchable_immediately_after_ack fails (a search right after the ack misses the doc: the write is only queued, and the reply says `"visible": false`). The rewritten guarantee is "searchable within 1 s of the ack". `tests/test_integration_async.py::test_acked_write_is_searchable_within_one_second` polls until the doc is visible and asserts the lag stays under 1 s, and back-pressure in `src/service/async_apply.py` is what makes that bound hold under load.

**4.** See docs/solutions/drills/drill_06.md.
