# Solutions: live updates

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** A plain removal forgets that the doc ever existed at version v. A delayed older upsert (version < v) would then look 'new' and resurrect the document. The tombstone remembers (v, deleted) so the version check rejects it.

**2.** The first to take the write lock wins; the second is a no-op ('stale_or_duplicate') because the version is not newer. That's OK only if clients guarantee one text per (id, version). It's the client's version contract, same as Elasticsearch external versioning.

**3.** test_freshness_write_searchable_immediately_after_ack fails (a search right after the ack misses the doc). The rewritten guarantee would be 'searchable within refresh_interval + apply time', with a test that polls until visible and asserts the bound.

**4.** See docs/solutions/drills/drill_06.md.
