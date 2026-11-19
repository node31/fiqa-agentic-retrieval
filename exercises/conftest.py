"""pytest setup for the checkers. You don't need to edit this.

* Puts this folder (for `common`) and the project repo (for `src`) on sys.path.
* Two speeds:   python -m pytest exercises            → TOY checks only (tiny FiQA, seconds)
                python -m pytest exercises --full     → FULL checks only (57,638 docs, minutes, cached)
                python -m pytest exercises --all      → both
  Run the toy checks while you code, and the full ones in a second terminal when you want the real numbers.
* EXERCISES_SOLUTIONS=1 grades the answer key (solutions/) instead of your files.
"""
import importlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import FIQA_DIR, REPO, SOLUTIONS, have_full_data, use_repo  # noqa: E402

use_repo()

if not (REPO / "src").is_dir():
    raise pytest.UsageError(f"Project code not found at {REPO / 'src'}: the exercises/ folder must stay inside the repo.")

# chapter folder -> module the checker imports. Order matters: later chapters import earlier ones.
MODULES = [("ch01_data", "loader"), ("ch02_metrics", "my_metrics"), ("ch03_bm25", "my_bm25"),
           ("ch04_dense", "my_dense"), ("ch05_fusion", "my_fusion"), ("ch06_pipeline", "mini_pipeline"),
           ("ch07_shards", "mini_shards"), ("ch08_agents", "agent_loop")]

if SOLUTIONS:
    # WHY pre-import: each checker does e.g. `import my_bm25`. Loading the solution module first puts it in
    # sys.modules, so that import returns the SOLUTION instead of the stub next to the checker.
    for folder, module in MODULES:
        sys.path.insert(0, str(HERE / "solutions" / folder))
        importlib.import_module(module)


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="run ONLY the full-FiQA checks (slow, cached)")
    parser.addoption("--all", action="store_true", help="run the toy AND the full checks")


def pytest_configure(config):
    config.addinivalue_line("markers", "full: runs on the full FiQA corpus; selected with --full or --all")


def pytest_report_header(config):
    mode = "ALL (toy + full)" if config.getoption("--all") else "FULL (57,638 docs)" if config.getoption("--full") \
        else "TOY (tiny FiQA). Real data: add --full"
    who = "the ANSWER KEY (solutions/)" if SOLUTIONS else "YOUR code"
    return [f"exercises: checking {who}; mode: {mode}"]


@pytest.fixture
def tiny_dir() -> Path:
    from common import TINY
    return TINY


def pytest_collection_modifyitems(config, items):
    want_full, want_all = config.getoption("--full"), config.getoption("--all")
    keep, drop = [], []
    for item in items:
        is_full = "full" in item.keywords
        if want_all or is_full == want_full:
            keep.append(item)
        else:
            drop.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
    if not have_full_data():
        skip = pytest.mark.skip(reason=f"full FiQA not found at {FIQA_DIR}: run python scripts/download_data.py")
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip)
