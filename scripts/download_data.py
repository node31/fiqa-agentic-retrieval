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
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
    print(f"done: {RAW / 'fiqa'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
