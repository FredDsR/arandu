#!/bin/bash
# =============================================================================
# Arandu Phase C RAG eval-chain Common Job Script
#
# Shared logic for the Phase C evaluation stages that have no dedicated cluster
# runner (chunk, kg-link-passages, kg-build-retriever-index,
# generate-non-answerable, retrieve, answer, judge-answers, rag-analysis).
# Sourced by the thin per-stage scripts under scripts/slurm/rag/, never run
# directly.
#
# Contract — the sourcing per-stage script sets:
#   RAG_CLI_ARGS      (required) the `arandu` subcommand + args, e.g.
#                     "answer --id mini-dry-run-qwen"
#   RAG_NEEDS_OLLAMA  "true" to bring up the ollama-gpu sidecar + pull the model
#                     before running the stage (LLM stages); default "false".
#
# Optional environment (defaults shown):
#   PIPELINE_ID            REQUIRED — names the run
#   RAG_OLLAMA_MODEL       qwen3:14b   — model pulled when RAG_NEEDS_OLLAMA=true
#   ARANDU_EMBEDDER_PROVIDER / ARANDU_EMBEDDER_MODEL   — embedder (index/retrieve)
#   ARANDU_<STAGE>_PROVIDER / _MODEL_ID / _BASE_URL    — per-stage LLM wiring
#   MIN_DISK_GB            15
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

RAG_CLI_ARGS="${RAG_CLI_ARGS:?RAG_CLI_ARGS must be set by the per-stage script}"
RAG_NEEDS_OLLAMA="${RAG_NEEDS_OLLAMA:-false}"
RAG_OLLAMA_MODEL="${RAG_OLLAMA_MODEL:-qwen3:14b}"
# Which compose service + profile runs the stage. GPU stages (LLM / embedder)
# use arandu-rag under the rag-gpu profile; pure-CPU stages override to
# arandu-rag-cpu / rag-cpu so they don't wait behind GPU contention.
RAG_SERVICE="${RAG_SERVICE:-arandu-rag}"
RAG_PROFILE="${RAG_PROFILE:-rag-gpu}"

# Directories (bind-mounted into the arandu-rag container)
export ARANDU_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
export ARANDU_HF_CACHE_DIR="${ARANDU_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"
export PIPELINE_ID="${PIPELINE_ID:-}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# ollama-gpu is the only GPU-backed sidecar profile we use on tupi.
OLLAMA_SERVICE="ollama-gpu"
DOCKER_PROFILE="$RAG_PROFILE"

# ---------------------------------------------------------------------------
# Container teardown trap.
#
# The stage launches containers via `docker compose run`/`up`, but those are
# owned by the docker daemon, NOT by this job step's process tree. So when
# SLURM ends the job WITHOUT a clean script exit (a TIME LIMIT timeout or an
# `scancel`), it kills the shell before a normal teardown runs, and the
# containers keep running on the node as ORPHANS: still holding the GPU and
# writing outputs outside any allocation, contending with whatever SLURM
# schedules onto the node next. Observed: judge-answers 799024 TIMEOUT left
# `ollama-gpu-<jobid>` + `<project>-arandu-rag-run-*` alive on tupi2, degrading
# another user's job.
#
# Two things are needed for the trap to actually fire in time:
#   1. The stage command runs in the BACKGROUND and is `wait`-ed on (see below).
#      bash does NOT run a trap while a foreground external command executes; it
#      defers it until that command returns. A foreground `docker compose run`
#      would therefore defer teardown until the container exits, i.e. never on a
#      real timeout. `wait` is interruptible, so backgrounding + wait lets the
#      SIGTERM/SIGINT handler run immediately.
#   2. `#SBATCH --signal=B:TERM@60` (per-stage scripts) makes SLURM send SIGTERM
#      to the batch shell 60s before the limit, well inside KillWait.
#
# Traps are armed LATER (via _rag_arm_traps, just before the first container
# starts), so an early validation `exit 1` does not run `docker compose down`
# while this job has no containers up (which could disturb a co-located job that
# shares the compose project).
_rag_teardown() {
    echo ""
    echo "[cleanup] tearing down containers (profile ${DOCKER_PROFILE})..."
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" \
        down --remove-orphans --timeout 10 2>/dev/null || true
}
_rag_on_signal() {
    # Ignore further INT/TERM while tearing down (a scancel retry or a short
    # KillWait must not kill the shell mid-`down`) and disarm EXIT so teardown
    # runs exactly once. $1 = signal name (for the log), $2 = exit code.
    trap '' INT TERM
    trap - EXIT
    echo ""
    echo "[cleanup] caught ${1} (SLURM timeout/scancel); tearing down..."
    _rag_teardown
    exit "${2}"
}
_rag_arm_traps() {
    trap _rag_teardown EXIT
    trap '_rag_on_signal SIGINT 130' INT   # 128 + SIGINT(2)
    trap '_rag_on_signal SIGTERM 143' TERM # 128 + SIGTERM(15)
}

echo "=============================================="
echo "Arandu Phase C RAG stage"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job Name:      ${SLURM_JOB_NAME:-rag}"
echo "Node:          $(hostname)"
echo "Start Time:    $(date)"
echo "Pipeline ID:   ${PIPELINE_ID:-<unset>}"
echo "CLI:           arandu ${RAG_CLI_ARGS}"
echo "Needs Ollama:  ${RAG_NEEDS_OLLAMA} (model: ${RAG_OLLAMA_MODEL})"
echo "Embedder:      ${ARANDU_EMBEDDER_PROVIDER:-<default>} / ${ARANDU_EMBEDDER_MODEL:-<default>}"
echo "Results Dir:   ${ARANDU_RESULTS_DIR}"
echo "=============================================="

cd "$PROJECT_DIR"

if [ -z "$PIPELINE_ID" ]; then
    echo "ERROR: PIPELINE_ID is required (names the run dir)." >&2
    exit 1
fi
if [ ! -d "$ARANDU_RESULTS_DIR/$PIPELINE_ID" ]; then
    echo "ERROR: run dir $ARANDU_RESULTS_DIR/$PIPELINE_ID not found." >&2
    echo "       Seed transcription first (copy it into the new run id)." >&2
    exit 1
fi

mkdir -p "$OLLAMA_MODELS_DIR" "$ARANDU_HF_CACHE_DIR" logs

# ---------------------------------------------------------------------------
# Disk preflight + cleanup (mirrors kg_common.sh; cluster nodes fill up)
# ---------------------------------------------------------------------------
echo "Pruning unused Docker data to free disk space..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down --remove-orphans 2>/dev/null || true
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
find "$OLLAMA_MODELS_DIR" -name "*-partial" -delete 2>/dev/null || true
find "$OLLAMA_MODELS_DIR" -name "*.tmp" -delete 2>/dev/null || true

MIN_DISK_GB=${MIN_DISK_GB:-15}
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
[ -d "$DOCKER_ROOT" ] || DOCKER_ROOT=/var/lib/docker
AVAIL_KB=$(df --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -d ' ' || true)
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1024 / 1024 ))
echo "Docker storage: $DOCKER_ROOT — ${AVAIL_GB} GB available (min ${MIN_DISK_GB})"
if [ "${AVAIL_KB:-0}" -gt 0 ] && [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
    echo "ERROR: not enough disk on $DOCKER_ROOT (${AVAIL_GB} GB < ${MIN_DISK_GB} GB)." >&2
    exit 1
fi

echo ""
echo "Building ${RAG_SERVICE} image (reuses the kg-extra image)..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build "$RAG_SERVICE"

# From here on containers get started, so arm the teardown traps. (Kept out of
# the early-validation/build path above so a pre-container `exit 1` cannot run
# `docker compose down` when this job has nothing up.)
_rag_arm_traps

# ---------------------------------------------------------------------------
# Ollama sidecar (LLM stages only)
# ---------------------------------------------------------------------------
if [ "$RAG_NEEDS_OLLAMA" = "true" ]; then
    echo ""
    echo "Starting ${OLLAMA_SERVICE} and pulling ${RAG_OLLAMA_MODEL}..."
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE"
    OLLAMA_READY=false
    for i in {1..30}; do
        if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
            OLLAMA_READY=true
            break
        fi
        echo "  Waiting for Ollama... ($i/30)"
        sleep 5
    done
    if [ "$OLLAMA_READY" = false ]; then
        echo "ERROR: Ollama failed to start after 30 attempts" >&2
        exit 1
    fi
    echo "Pulling model: $RAG_OLLAMA_MODEL"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama pull "$RAG_OLLAMA_MODEL"
fi

# ---------------------------------------------------------------------------
# Run the stage. `run --rm` overrides the entrypoint args cleanly (the image's
# ENTRYPOINT is `arandu`), so RAG_CLI_ARGS is the subcommand + flags.
# ---------------------------------------------------------------------------
echo ""
echo "Running: arandu ${RAG_CLI_ARGS}"
echo "=============================================="
# set +e: under `set -e` a failing stage aborts the script HERE, skipping
# the cleanup below and leaking the ollama sidecar on the shared node
# (observed with judge-answers 795114). Capture the rc instead.
#
# Run in the BACKGROUND + `wait` (not foreground): bash defers signal traps
# until a foreground external command returns, so a foreground run would keep
# the SIGTERM teardown from firing until the container exits (never, on a real
# timeout). `wait` is interruptible, so the trap runs immediately; if no signal
# arrives, `wait` returns the container's real exit code.
set +e
# shellcheck disable=SC2086  # intentional word-splitting of the arg string
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" run --rm "$RAG_SERVICE" ${RAG_CLI_ARGS} &
RUN_PID=$!
wait "$RUN_PID"
RUN_RC=$?
set -e

echo "=============================================="
echo "Stage finished (rc=${RUN_RC}) at $(date)"
echo "=============================================="
# Container teardown is handled by the _rag_teardown EXIT trap set above, so it
# runs here on normal exit AND on a SLURM SIGTERM (timeout/scancel). Do not add
# a manual `docker compose down`; it would just double-run the trap.
exit $RUN_RC
