"""Guard: no API key or secret-looking string may be committed.

WHY: the Azure key lives in .env (gitignored). This test fails loudly if a
secret ever lands in a tracked file, e.g. a key pasted into a doc or a test.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Azure AI keys are long base62-ish strings; also catch "api_key = '...'" patterns.
PATTERNS = [
    re.compile(r"[A-Za-z0-9]{70,}"),
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9]{24,}"),
]
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".example", ""}
# Never-tracked folders, skipped when we have to walk the tree ourselves (no git available).
UNTRACKED_DIRS = {".git", ".venv", "artifacts", "dist", ".cache", ".pytest_cache", "__pycache__", "raw", "node_modules"}


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """Run git in the repo, or None if git is missing or this isn't a checkout (e.g. inside the Docker image)."""
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return None if r.returncode == 128 else r   # 128 = "not a git repository"


def tracked_files():
    r = _git("ls-files")
    if r is not None and r.returncode == 0:
        return [ROOT / p for p in r.stdout.splitlines()]
    # WHY a fallback: without git (a source zip, the container) the guard must still look at the files.
    return [p for p in ROOT.rglob("*") if p.name != ".env" and not (UNTRACKED_DIRS & set(p.relative_to(ROOT).parts))]


def test_env_file_is_gitignored():
    r = _git("check-ignore", ".env")
    if r is not None:
        assert r.returncode == 0, ".env must be gitignored"
    else:
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in (line.strip() for line in lines), ".env must be listed in .gitignore"


def test_no_secrets_in_tracked_files():
    offenders = []
    for p in tracked_files():
        if p.suffix not in ALLOWED_SUFFIXES or not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in PATTERNS:
            if pat.search(text):
                offenders.append(str(p.relative_to(ROOT)))
                break
    assert not offenders, f"secret-looking strings in: {offenders}"


# Employer / machine details that must not ship: internal package mirrors and hosts, an absolute
# home or work folder, cloud resource names. Kept as fragments so this file does not match itself.
PRIVATE = re.compile("|".join([
    "package" + "feedproxy", "micro" + "soft\\.io", "visual" + "studio\\.com", "dev\\.azure" + "\\.com",
    "share" + "point\\.com", "agency_" + "copilot", "prag" + "upta", r"[A-Za-z]:\\{1,2}Users\\",
    "/Users/[a-z]", "/home/[a-z]", "Dev" + "Box", "Win" + "UI", "foun" + "dary-", "rg-" + "pranav",
]), re.IGNORECASE)
TEXT_SUFFIXES = ALLOWED_SUFFIXES | {".json", ".jsonl", ".csv", ".html", ".ps1", ".sh"}


def test_no_private_details_in_tracked_files():
    offenders = []
    for p in tracked_files():
        if p.suffix not in TEXT_SUFFIXES or not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        m = PRIVATE.search(p.read_text(encoding="utf-8", errors="ignore"))
        if m:
            offenders.append(f"{p.relative_to(ROOT)}: {m.group(0)!r}")
    assert not offenders, f"private details in: {offenders}"
