# Reproducible CPU-only runtime for the router, shards, indexer and eval.
#   docker compose build          # see docker-compose.yml
# Corporate networks: pass your PyPI mirror, e.g.
#   docker compose build --build-arg PIP_INDEX_URL=https://<mirror>/pypi/simple/
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf \
    PYTHONPATH=/app

WORKDIR /app
# WHY a separate torch step: the default PyPI torch wheel for Linux pulls ~2.5 GB of CUDA libraries.
# The CPU wheel is ~200 MB, and the assignment is CPU-only.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the models into the image so containers start offline and identically every time.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY src ./src
COPY scripts ./scripts
COPY data/build_compound.py data/compound_dev.jsonl data/compound_rewrites.json ./data/
COPY examples ./examples
COPY tests ./tests
COPY Makefile pytest.ini ./

# Data (data/raw) and indexes (artifacts/) are volumes: see docker-compose.yml.
CMD ["python", "-m", "src.service.router"]
