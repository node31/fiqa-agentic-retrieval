"""Toy 18: freshness. Synchronous apply (searchable at ack) vs asynchronous apply (eventually).

Concept: the freshness guarantee is a design choice trading write latency against read visibility.
Real code: ShardStore._write (sync apply), tests/test_integration_service.py   Doc: docs/topics/09_live_updates_consistency.md
"""
import queue
import threading
import time

from _common import banner, note, step, try_this

banner("Toy 18: when does a write become searchable?")


class SyncIndex:
    def __init__(self):
        self.docs = set()

    def write(self, d):
        time.sleep(0.010)  # embedding + index update happen BEFORE the ack
        self.docs.add(d)

    def search(self, d):
        return d in self.docs


class AsyncIndex:
    def __init__(self, apply_delay=0.2):
        self.docs, self.q = set(), queue.Queue()
        threading.Thread(target=self._applier, args=(apply_delay,), daemon=True).start()

    def _applier(self, delay):
        while True:
            d = self.q.get()
            time.sleep(delay)  # a background indexer (batching, refresh interval...)
            self.docs.add(d)

    def write(self, d):
        self.q.put(d)  # ack immediately

    def search(self, d):
        return d in self.docs


for name, ix in (("sync", SyncIndex()), ("async (refresh 200 ms)", AsyncIndex())):
    t0 = time.perf_counter()
    ix.write("doc-1")
    ack = (time.perf_counter() - t0) * 1e3
    visible_now = ix.search("doc-1")
    while not ix.search("doc-1"):
        time.sleep(0.005)
    lag = (time.perf_counter() - t0) * 1e3
    step(name, f"ack after {ack:.0f} ms | searchable right after ack: {visible_now} | searchable after {lag:.0f} ms")
note("Our shards apply synchronously: write latency ~10-20 ms (one embedding), and the guarantee is "
     "'searchable by the first search issued after the ack'. Elasticsearch defaults to async with a 1 s refresh "
     "interval: higher write throughput, but 'read your own write' fails for up to 1 s.")
note("tests/test_integration_service.py::test_freshness... PROVES our guarantee end to end, over HTTP and "
     "across 3 shard processes, with no sleeps or retries.")
try_this("Batch 100 writes: which design has higher throughput? Why?",
         "How would you offer BOTH: async by default, plus a ?refresh=wait_for flag per write?",
         "With replicas (not built here), does 'searchable after ack' still hold? What would you need?")
