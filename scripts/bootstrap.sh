#!/usr/bin/env bash
# =====================================================================================================
# bootstrap.sh: from a fresh clone to a RUNNING, VERIFIED system with one command (Linux and macOS).
# Windows: scripts\bootstrap.ps1 does the same steps. Full guide: DEPLOY.md.
#
#   ./scripts/bootstrap.sh                    # sync write mode (default), smoke test at the end
#   ./scripts/bootstrap.sh --mode async       # async write mode (ADR 0013)
#   ./scripts/bootstrap.sh --migrate-check    # also prove a live blue/green model migration works
#   ./scripts/bootstrap.sh --build-index      # encode embeddings locally instead of downloading them
#   ./scripts/bootstrap.sh --down             # stop everything (data and caches are kept)
#
# Needs ONLY Docker (with Compose v2). No Python on the host: data download, unzip and the smoke test
# all run inside the project's own image. Safe to re-run: every step skips work that is already done.
#
# Steps
#   0. prerequisites  docker + compose v2 + daemon running, memory/disk/port 8000 warnings
#   1. config         .env (from .env.example), PyPI mirror detection, folders for the bind mounts
#   2. image          docker compose build (pinned versions: constraints.txt)
#   3. data           BEIR FiQA -> data/raw/fiqa       (upstream URL, fallback: GitHub release)
#   4. embeddings     artifacts/emb_*.npy              (GitHub release download, fallback: encode locally)
#   5. start          docker compose up -d --wait      (3 shard containers + router, all healthy)
#   6. verify         scripts/smoke_test.py inside a container (search, freshness, idempotency, ...)
# =====================================================================================================
set -euo pipefail

# Always run from the repo root, wherever the script is called from.
cd "$(dirname "$0")/.."

MODE="${APPLY_MODE:-sync}"
MIGRATE_CHECK=0
BUILD_INDEX=0
DOWN=0
RELEASE_TAG="index-artifacts-v1"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --migrate-check) MIGRATE_CHECK=1; shift ;;
    --build-index) BUILD_INDEX=1; shift ;;
    --down) DOWN=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1"; usage 1 ;;
  esac
done
case "$MODE" in sync|async) ;; *) echo "--mode must be sync or async"; exit 1 ;; esac
export APPLY_MODE="$MODE"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# `docker compose run` for one-off commands inside the project image (profile "tools").
tools() { docker compose --profile tools run --rm -T tools "$@"; }

# ---------------------------------------------------------------------------------------------------
step "0/6 prerequisites"
command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install Docker Desktop (macOS) or Docker Engine (Linux): https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || die "The Docker daemon is not running. Start Docker Desktop (macOS) or: sudo systemctl start docker (Linux). On Linux without sudo for docker: sudo usermod -aG docker \$USER, then log out and in."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required ('docker compose', not 'docker-compose'). Update Docker."
ok "$(docker --version | head -n1); $(docker compose version | head -n1)"
ok "host: $(uname -s) $(uname -m)  (images build natively for amd64 and arm64)"

if [ "$DOWN" -eq 1 ]; then
  step "stopping the stack (data, embeddings and WALs are kept on disk)"
  docker compose --profile tools down
  exit 0
fi

# Memory available to Docker (bytes). 3 shards x ~0.6-1 GB + router ~0.8 GB, x2 during a migration.
MEM_BYTES="$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)"
MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
if [ "$MEM_GB" -lt 6 ]; then
  warn "Docker has only ${MEM_GB} GB of memory. Give it >= 8 GB (Docker Desktop > Settings > Resources > Memory)."
else
  ok "Docker memory: ${MEM_GB} GB"
fi
FREE_GB=$(( $(df -Pk . | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if [ "$FREE_GB" -lt 12 ]; then warn "only ${FREE_GB} GB free disk here; the image + data + caches need ~8-10 GB"; else ok "free disk: ${FREE_GB} GB"; fi

# Port 8000 must be free, unless it is OUR router from an earlier run (then `up` just keeps it).
if (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
  if docker compose ps --status running --services 2>/dev/null | grep -qx router; then
    ok "port 8000: our router is already running (will be updated in place)"
  else
    die "port 8000 is already used by another program. Stop it, then re-run."
  fi
fi

# ---------------------------------------------------------------------------------------------------
step "1/6 configuration"
# .env must exist BEFORE any `compose run tools`: it is bind-mounted, and Docker would otherwise create
# an empty DIRECTORY named .env. It only holds optional LLM settings (Part C); none are needed here.
if [ ! -f .env ]; then cp .env.example .env; ok "created .env from .env.example (LLM keys optional, not needed to run search)"; else ok ".env exists"; fi
mkdir -p data/raw artifacts results/runs
# PyPI mirror: honour PIP_INDEX_URL, else the host's pip config (corporate networks), else pypi.org.
if [ -z "${PIP_INDEX_URL:-}" ]; then
  for p in pip3 pip; do
    if command -v "$p" >/dev/null 2>&1; then
      PIP_INDEX_URL="$("$p" config get global.index-url 2>/dev/null || true)"
      [ -n "$PIP_INDEX_URL" ] && break
    fi
  done
fi
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
ok "PyPI index for the image build: $PIP_INDEX_URL"
ok "write mode: APPLY_MODE=$APPLY_MODE"

# ---------------------------------------------------------------------------------------------------
step "2/6 build the image (first time ~5-10 min: CPU torch + models baked in; later runs use the cache)"
docker compose build
ok "image fiqa-retrieval:latest ready ($(docker image inspect fiqa-retrieval:latest --format '{{.Architecture}}'))"

# GitHub repo slug (owner/name) from the clone's origin, for release downloads. Handles
# https://github.com/o/r(.git), git@github.com:o/r.git and https://user:token@github.com/o/r.
SLUG="$(git config --get remote.origin.url 2>/dev/null | sed -E 's#^.*github\.com[:/]##; s#\.git$##; s#/$##' || true)"

# fetch_release_asset <file> <dest_dir>: 1) public URL, 2) GitHub API with a token (GITHUB_TOKEN /
# GH_TOKEN, or the gh CLI's login) for PRIVATE repos, using only curl/awk. 0 = success.
fetch_release_asset() {
  local name="$1" dest="$2" token api url loc
  [ -n "$SLUG" ] || return 1
  if curl -fsSL -o "$dest/$name" "https://github.com/$SLUG/releases/download/$RELEASE_TAG/$name" 2>/dev/null; then return 0; fi
  rm -f "$dest/$name"
  token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [ -z "$token" ] && command -v gh >/dev/null 2>&1; then token="$(gh auth token 2>/dev/null || true)"; fi
  [ -n "$token" ] || return 1
  api="$(curl -fsSL -H "Authorization: Bearer $token" "https://api.github.com/repos/$SLUG/releases/tags/$RELEASE_TAG" 2>/dev/null)" || return 1
  # The API's JSON is pretty-printed: each asset has its "url" line before its "name" line.
  url="$(printf '%s\n' "$api" | awk -v n="\"name\": \"$name\"" '/"url": ".*\/releases\/assets\//{u=$2} index($0,n){gsub(/[",]/,"",u); print u; exit}')"
  [ -n "$url" ] || return 1
  # PITFALL: `curl -L -H Authorization` would forward the token to the redirect target (a pre-signed
  # storage URL), which rejects requests carrying two auth methods. So: resolve the redirect WITH the
  # token, then download the signed URL WITHOUT it.
  loc="$(curl -fsS -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $token" -H "Accept: application/octet-stream" "$url")" || return 1
  [ -n "$loc" ] || return 1
  curl -fsSL -o "$dest/$name" "$loc" || { rm -f "$dest/$name"; return 1; }
}

# ---------------------------------------------------------------------------------------------------
step "3/6 data: BEIR FiQA (57,638 passages)"
if [ -f data/raw/fiqa/corpus.jsonl ]; then
  ok "already present: data/raw/fiqa"
elif tools python scripts/download_data.py; then
  ok "downloaded from the BEIR mirror"
else
  warn "upstream download failed; trying the copy attached to the GitHub release"
  fetch_release_asset fiqa.zip data/raw || die "could not download FiQA. Check your internet connection / proxy and re-run."
  tools python -c "import zipfile; zipfile.ZipFile('data/raw/fiqa.zip').extractall('data/raw')"
  ok "extracted data/raw/fiqa"
fi

# ---------------------------------------------------------------------------------------------------
step "4/6 embedding caches (MiniLM + bge-small for all passages)"
have_caches() { [ -f artifacts/emb_minilm.npy ] && [ -f artifacts/emb_minilm.meta.json ] && [ -f artifacts/emb_bge-small.npy ] && [ -f artifacts/emb_bge-small.meta.json ]; }
if have_caches; then
  ok "already present: artifacts/emb_*"
else
  if [ "$BUILD_INDEX" -eq 0 ] && fetch_release_asset fiqa_index_artifacts.zip artifacts; then
    tools python -c "import zipfile; zipfile.ZipFile('artifacts/fiqa_index_artifacts.zip').extractall('.')"
    rm -f artifacts/fiqa_index_artifacts.zip
    ok "downloaded pre-built caches from release $RELEASE_TAG"
  else
    [ "$BUILD_INDEX" -eq 1 ] || warn "release download not possible (private repo without the gh CLI?)"
    warn "encoding 2 x 57,638 passages on the CPU: ~30-40 min, once"
    tools python -m src.index.build_index --models minilm bge-small
  fi
  have_caches || die "embedding caches missing after step 4"
  ok "caches ready"
fi

# ---------------------------------------------------------------------------------------------------
step "5/6 start: 3 shard containers + router (waits until every healthcheck passes, ~1-2 min)"
docker compose up -d --wait --wait-timeout 900 || { docker compose ps; docker compose logs --tail 40; die "the stack did not become healthy (logs above)"; }
docker compose ps --format 'table {{.Service}}\t{{.State}}\t{{.Health}}\t{{.Ports}}'

# ---------------------------------------------------------------------------------------------------
step "6/6 verify (inside a container, against http://router:8000)"
SMOKE_ARGS="--url http://router:8000"
[ "$MIGRATE_CHECK" -eq 1 ] && SMOKE_ARGS="$SMOKE_ARGS --migrate"
# shellcheck disable=SC2086
tools python -m scripts.smoke_test $SMOKE_ARGS || die "smoke test failed (see above). Logs: docker compose logs router shard0"

# On Linux, files written by containers (running as root) are root-owned; hand them back to you so a
# plain `rm -rf` of the clone works later. Docker Desktop (macOS/Windows) maps ownership already.
if [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" -ne 0 ]; then
  docker compose --profile tools run --rm -T --user root tools chown -R "$(id -u):$(id -g)" /app/artifacts /app/data/raw /app/results >/dev/null 2>&1 || true
fi

cat <<EOF

$(printf '\033[1;32m')System is up and verified.$(printf '\033[0m')
  Router (the only entry point):  http://localhost:8000      API docs: http://localhost:8000/docs
  Try a search:
    curl -s -X POST localhost:8000/search -H 'content-type: application/json' -d '{"query":"what is short selling?"}'
  Status of every shard:          curl -s localhost:8000/admin/status
  Full dev evaluation (500 q):    docker compose --profile tools run --rm eval
  Logs:                           docker compose logs -f router shard0
  Stop (keeps data):              ./scripts/bootstrap.sh --down
  More (scenarios, migration, load tests, troubleshooting): DEPLOY.md
EOF
