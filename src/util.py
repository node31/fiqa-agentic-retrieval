"""Small shared helpers (memory measurement, percentiles).

HOW TO READ THIS FILE
    Two independent helpers used by the benchmark code (src/eval.py, the shard
    servers' /admin/status):
      rss_mb()      how much RAM this process uses now, and the most it ever used
      percentile()  p50 / p95 latency from a list of raw timings
"""
from __future__ import annotations

import os
import sys


def rss_mb() -> dict:
    """Current and peak resident memory of THIS process, in MB.

    PITFALL: "peak" is OS-specific. Windows exposes peak working set,
    Linux exposes ru_maxrss (in KB). macOS exposes ru_maxrss in bytes.
    See docs/topics/07_benchmarking_latency_memory.md.

    Jargon:
        RSS ("resident set size") = the bytes of this process that are actually
        in physical RAM right now (not swapped out, not merely reserved).
        Windows calls the same idea the "working set"; its high-water mark is
        the "peak working set".

    Returns ``{"rss_mb": <now>, "peak_rss_mb": <highest ever>}``. The PEAK is
    what matters for "does it fit in N GB": building an index can briefly use
    far more RAM than the process holds once it is idle again.
    """
    # WHY imported here: psutil is only needed when someone measures memory,
    # so importing this module stays cheap.
    import psutil

    mi = psutil.Process(os.getpid()).memory_info()
    # Windows: psutil exposes the peak working set directly (in bytes).
    # Linux/macOS: that attribute doesn't exist, so getattr returns None.
    peak = getattr(mi, "peak_wset", None)
    if peak is None:
        # `resource` is a Unix-only stdlib module (importing it on Windows fails),
        # which is why it is imported only on this branch.
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # PITFALL: ru_maxrss units differ by OS: bytes on macOS ("darwin"),
        # kilobytes on Linux. Normalise both to bytes.
        peak = ru if sys.platform == "darwin" else ru * 1024
    # 2**20 bytes = 1 MiB (reported as "MB" throughout the project).
    return {"rss_mb": round(mi.rss / 2**20, 1), "peak_rss_mb": round(peak / 2**20, 1)}


def percentile(values, p: float) -> float:
    """Nearest-rank percentile (p in [0, 100]). No interpolation, so p95 of 20 values = 19th.

    >>> percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)
    5

    Nearest-rank percentile: sort the values, then return the value at
    rank ceil(p/100 * n) (1-based). The answer is always one of the measured
    values, never an average of two neighbours (numpy's default interpolates,
    so it can give slightly different numbers).
    Example: 20 latencies, p=95 -> rank ceil(0.95 * 20) = 19 -> the 19th
    smallest, i.e. only ONE sample is allowed to be slower than "p95".
    """
    # WHY NaN and not 0: an empty list has no percentile; 0 ms would look like
    # a (suspiciously great) real measurement.
    if not values:
        return float("nan")
    s = sorted(values)
    import math

    # max(1, ...) handles p=0, which would otherwise give rank 0 -> s[-1] (the MAX).
    rank = max(1, math.ceil(p / 100 * len(s)))
    # rank is 1-based, Python lists are 0-based.
    return s[rank - 1]
