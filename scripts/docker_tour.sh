#!/usr/bin/env bash
# docker_tour.sh: run every Docker step of the manual tests (guide section 3) with one command, check each result,
# write a report with an animated replay per step, and put the stack back exactly as it was.
# Windows: scripts/docker_tour.ps1 does the same steps.
#
#   ./scripts/docker_tour.sh                       # everything except the paid agent step (~2-3 min)
#   ./scripts/docker_tour.sh --azure               # also the agents (3e) on Azure; needs .env filled in
#   ./scripts/docker_tour.sh --skip eval,simple    # leave out some phases
#   ./scripts/docker_tour.sh --only writes,shard   # just these phases
#   ./scripts/docker_tour.sh --no-open             # don't open the report at the end
#
# Needs ONLY Docker and a running stack (./scripts/bootstrap.sh). No Python on the host: the checks run in the
# project's "tools" container, with the repo's scenarios/, scripts/ and src/ mounted read-only.
# Phases, in order: search (3a), eval (3b), writes (3c), migration (3d), simple (3f), agents (3e, only with --azure),
# shard (3g: stops and starts shard1).
# Not part of the tour: 3h (scripts/docker_check.py wipes artifacts/docker) and 3i (the async write mode): the tour
# never changes the write mode, and needs the stack in sync mode (the bootstrap default).
# Before anything changes it records the stack's shape; at the end it starts shard1 if needed, deletes the docs it
# wrote, restores the models, and verifies the shape is identical (also after a failure or Ctrl+C).
# Output: toy_runs/docker_tour/<stamp>/index.html (gitignored).
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PHASES="search eval writes migration simple agents shard"
AZURE=0; OPEN=1; SKIP=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --azure) AZURE=1 ;;
    --no-open) OPEN=0 ;;
    --skip) SKIP="$(echo "${2:-}" | tr ',' ' ')"; shift ;;
    --only) ONLY="$(echo "${2:-}" | tr ',' ' ')"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done
for p in $SKIP $ONLY; do
  case " $PHASES " in *" $p "*) ;; *) echo "unknown phase '$p' (phases: $PHASES)"; exit 2 ;; esac
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
want() {   # want PHASE -> 0 if it should run
  if [ -n "$ONLY" ]; then case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; fi
  case " $SKIP " in *" $1 "*) return 1 ;; *) return 0 ;; esac
}

# ---------------------------------------------------------------------------------------------- prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is not installed (or not on PATH)."; exit 2; }
docker info >/dev/null 2>&1 || { echo "The Docker daemon is not running. Start Docker first."; exit 2; }
[ -f .env ] || cp .env.example .env   # the tools container mounts it

STAMP="$(date +%Y%m%d-%H%M%S)"
REL_DIR="toy_runs/docker_tour/$STAMP"
DIR="$ROOT/$REL_DIR"
mkdir -p "$DIR"
HOST_ROOT="$ROOT"
# Git Bash on Windows: give docker a Windows path, and stop MSYS from rewriting /app/... arguments. No-op elsewhere.
if pwd -W >/dev/null 2>&1; then HOST_ROOT="$(pwd -W)"; export MSYS_NO_PATHCONV=1; fi
MOUNTS=(-v "$HOST_ROOT/scenarios:/app/scenarios:ro" -v "$HOST_ROOT/scripts:/app/scripts:ro" -v "$HOST_ROOT/src:/app/src:ro"
        -v "$HOST_ROOT/toy_runs:/app/toy_runs")
# On Linux, files written by a container belong to its user: run as you, so you can delete toy_runs/ later.
[ "$(uname -s)" = "Linux" ] && MOUNTS+=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
AZURE_ARG=()
if [ "$AZURE" = 1 ]; then
  mkdir -p "$ROOT/.llm_cache"
  MOUNTS+=(-v "$HOST_ROOT/.llm_cache:/app/.llm_cache")   # reuse cached LLM answers: a rerun costs nothing
  AZURE_ARG=(--azure)
fi

# One command inside the project image, against http://router:8000 on the compose network.
tour() { docker compose --profile tools run --rm -T "${MOUNTS[@]}" tools python -m scripts.docker_tour "$@" --dir "$REL_DIR"; }
# The containers' state, read by the shape check (before and after).
save_compose() { docker compose ps --format json > "$DIR/compose_now.json"; }
now() { date +%s; }
# A docker-level step (only the wrapper can do these) goes into the report too.
note() {   # note KEY SECTION TITLE WHAT OK(0/1) SECONDS GOT
  local status=fail mark=FAIL
  [ "$5" = 1 ] && status=pass && mark=PASS
  printf '{"key": "%s", "section": "%s", "title": "%s", "what": "%s", "expected": "the stack is healthy again", "status": "%s", "seconds": %s, "got": "%s", "t": %s}\n' \
    "$1" "$2" "$3" "$4" "$status" "$6" "$7" "$(( $(now) - $6 ))" >> "$DIR/wrapper.jsonl"
  printf '  %-4s  %3s  %s  (%s s)\n' "$mark" "$2" "$3" "$6"
}
# Start shard1 again and wait until every healthcheck passes (it replays its WAL first). APPLY_MODE is pinned to
# sync so `up` never recreates the shards in another mode, whatever this shell has exported.
start_shard1() {
  docker compose start shard1 && APPLY_MODE=sync docker compose up -d --wait --wait-timeout 900
}

# ---------------------------------------------------------------------------------------------- preflight
step "preflight: the stack is up; its shape is recorded so it can be restored and verified at the end"
save_compose
if ! tour preflight; then
  echo; echo "Nothing was changed."; exit 2
fi
SHARD1_STOPPED=0
COMPLETED=0
RC=1

restore() {
  trap - EXIT INT TERM
  step "restore: shard1, the tour's docs and the models; then verify the shape"
  if [ "$SHARD1_STOPPED" = 1 ]; then
    local t0 ok=0
    t0=$(now)
    start_shard1 && ok=1
    note restart_shard1 end "Start shard1 again (the tour stopped early)" "docker compose start shard1, wait until healthy" "$ok" "$(( $(now) - t0 ))" "shard1 healthy again: $ok"
  fi
  save_compose
  tour finish; RC=$?
  [ "$COMPLETED" = 1 ] || { [ "$RC" = 0 ] && RC=130; }
  REPORT="$DIR/index.html"
  if [ -f "$REPORT" ] && [ "$OPEN" = 1 ]; then
    if command -v open >/dev/null 2>&1; then open "$REPORT"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$REPORT" >/dev/null 2>&1 || true; fi
  fi
  echo
  if [ "$COMPLETED" != 1 ]; then echo "The tour stopped early; the stack was restored (the shape check is the last table in the report)."
  elif [ "$RC" = 0 ]; then echo "Every step passed, and the stack is back in its original shape."
  else echo "Some steps failed (see the report). The shape check is the last table in it."; fi
  echo "Report: $REPORT"
  exit "$RC"
}
trap restore EXIT INT TERM   # restore runs on success, on failure and on Ctrl+C

RUN_NOW=(); SKIPPED=()
for p in search eval writes migration simple agents; do want "$p" && RUN_NOW+=("$p"); done
for p in $PHASES; do want "$p" || SKIPPED+=("skip:$p"); done

step "3a-3f: search, the dev eval, writes, migration, the simple scenarios"
tour run ${RUN_NOW[@]+"${RUN_NOW[@]}"} ${SKIPPED[@]+"${SKIPPED[@]}"} ${AZURE_ARG[@]+"${AZURE_ARG[@]}"}

if want shard; then
  step "3g: lose a shard"
  tour run shard_before
  t0=$(now); docker compose stop shard1; ok=$?; SHARD1_STOPPED=1
  note stop_shard1 3g "docker compose stop shard1" "stop the container; the router's calls to it now fail" "$([ $ok = 0 ] && echo 1 || echo 0)" "$(( $(now) - t0 ))" "shard1 stopped"
  tour run shard_down
  t0=$(now)
  if start_shard1; then ok=1; SHARD1_STOPPED=0; else ok=0; fi
  note start_shard1 3g "docker compose start shard1" "start it again; it replays its WAL before /health says healthy" "$ok" "$(( $(now) - t0 ))" "shard1 healthy again: $ok"
  tour run shard_back
fi
COMPLETED=1
# restore() runs from the EXIT trap