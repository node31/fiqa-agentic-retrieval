"""Small shared helpers (memory measurement, percentiles)."""
from __future__ import annotations

import os
import sys


def rss_mb() -> dict:
    """Current and peak resident memory of THIS process, in MB.

    PITFALL: "peak" is OS-specific. Windows exposes peak working set,
    Linux exposes ru_maxrss (in KB). macOS exposes ru_maxrss in bytes.
    See docs/07_benchmarking_latency_memory.md.
    """
    import psutil

    mi = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = ru if sys.platform == "darwin" else ru * 1024
    return {"rss_mb": round(mi.rss / 2**20, 1), "peak_rss_mb": round(peak / 2**20, 1)}


def percentile(values, p: float) -> float:
    """Nearest-rank percentile (p in [0, 100]). No interpolation, so p95 of 20 values = 19th.

    >>> percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)
    5
    """
    if not values:
        return float("nan")
    s = sorted(values)
    import math

    rank = max(1, math.ceil(p / 100 * len(s)))
    return s[rank - 1]
