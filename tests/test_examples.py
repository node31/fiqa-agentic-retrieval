"""Smoke-test every toy in examples/ so the learning material never rots.

Each toy must exit 0 within 60 s. Toys that need a model degrade gracefully
offline (see examples/_common.py::load_st_model).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("toy", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_toy_runs(toy):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    r = subprocess.run([sys.executable, str(toy)], cwd=toy.parent, capture_output=True, text=True,
                       timeout=120, env=env, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "TRY THIS" in r.stdout or "unavailable" in r.stdout or "not available" in r.stdout
