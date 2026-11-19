# FiQA Distributed + Agentic Retrieval

A learning-first implementation of a take-home assignment. It covers
retrieval over BEIR/FiQA, a sharded service with live updates and zero-downtime model migration, and
hand-written single-agent and multi-agent retrieval.

> 🚧 Work in progress. Setup and run instructions will be filled in as each part lands.

## Quick start (local, Windows/Linux/macOS)

```bash
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py                      # BEIR FiQA -> data/raw/fiqa
python -m src.beir_data                              # dataset statistics
pytest -q
```

Start learning at [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md).
