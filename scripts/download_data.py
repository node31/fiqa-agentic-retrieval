"""Download the standard BEIR FiQA archive into data/raw/fiqa (idempotent).

Usage: python scripts/download_data.py
Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
(the URL used by the official ``beir`` package's ``util.download_and_unzip``).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    target = RAW / "fiqa" / "corpus.jsonl"
    if target.exists():
        print(f"already present: {target.parent}")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "fiqa.zip"
    print(f"downloading {URL} ...")
    try:
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW)
    except (OSError, zipfile.BadZipFile) as e:  # URLError, SSL errors, timeouts, truncated zip
        # One clear line instead of a traceback: the bootstrap scripts then fall back to the copy
        # attached to the GitHub release. Remove a partial file so a re-run starts clean.
        zpath.unlink(missing_ok=True)
        print(f"download failed: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1
    print(f"done: {RAW / 'fiqa'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
