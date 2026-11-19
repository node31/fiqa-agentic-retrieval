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


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [ROOT / p for p in out.splitlines()]


def test_env_file_is_gitignored():
    r = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, ".env must be gitignored"


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
