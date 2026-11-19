"""Shared paths and helpers for all exercises. You don't need to edit this.

REPO      the project root (this folder's parent): reference code in src/, the guide in docs/, data in data/.
TINY      the 24-document mini FiQA used by the TOY checks (same 3-file format as the real one). Seconds.
FIQA_DIR  the full FiQA download (57,638 docs), used by the FULL checks (`--full`). Minutes, cached.
CACHE     exercises/.cache/: slow results (e.g. your corpus embeddings) saved here so a second run is instant.

Set EXERCISES_SOLUTIONS=1 to make everything (checkers + use_chapter) use the files in solutions/ instead of yours.
"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../fiqa-agentic-retrieval/exercises
REPO = Path(os.environ.get("FIQA_REPO", HERE.parent)).resolve()  # .../fiqa-agentic-retrieval
TINY = REPO / "tests" / "fixtures" / "tinyfiqa"
FIQA_DIR = REPO / "data" / "raw" / "fiqa"
CACHE = HERE / ".cache"
SOLUTIONS = os.environ.get("EXERCISES_SOLUTIONS") == "1"


def chapter_dir(name: str) -> Path:
    """Folder of a chapter's code: yours (default) or the answer key's (EXERCISES_SOLUTIONS=1)."""
    return HERE / "solutions" / name if SOLUTIONS else HERE / name


def use_chapter(*names: str) -> None:
    """Make other chapters' files importable, e.g. use_chapter("ch03_bm25") then `import my_bm25`."""
    for n in names:
        p = str(chapter_dir(n))
        if p not in sys.path:
            sys.path.insert(0, p)


def use_repo() -> None:
    """Make the reference code importable as `src...` (the checkers use it to compare with your code)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def have_full_data() -> bool:
    return (FIQA_DIR / "corpus.jsonl").exists()


def dataset_from_argv(argv: list[str] | None = None) -> tuple[str, Path]:
    """For demos: `python <file>.py` → ("tiny", TINY);  `python <file>.py --full` → ("full", FIQA_DIR)."""
    if "--full" in (sys.argv if argv is None else argv):
        if not have_full_data():
            raise SystemExit(f"Full FiQA not found at {FIQA_DIR}. Download it first: python scripts/download_data.py")
        return "full", FIQA_DIR
    return "tiny", TINY


def cached_npy(name: str, compute, key: str = ""):
    """Return compute() (a numpy array), saved in exercises/.cache/ so later calls are instant.

    `key` is any string that should invalidate the cache when it changes. Pass the SOURCE CODE of the function that
    produced the numbers (inspect.getsource(fn)): edit your function and the cache recomputes automatically.
    Delete exercises/.cache/ to force everything to recompute.
    """
    import numpy as np

    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    path = CACHE / f"{name}-{tag}.npy"
    if path.exists():
        return np.load(path)
    CACHE.mkdir(exist_ok=True)
    arr = np.asarray(compute())
    np.save(path, arr)
    return arr
