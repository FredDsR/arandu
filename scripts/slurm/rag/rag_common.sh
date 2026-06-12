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
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "/var/lib/docker")
AVAIL_KB=$(df --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "Docker storage: $DOCKER_ROOT — ${AVAIL_GB} GB available (min ${MIN_DISK_GB})"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
    echo "ERROR: not enough disk on $DOCKER_ROOT (${AVAIL_GB} GB < ${MIN_DISK_GB} GB)." >&2
    exit 1
fi

echo ""
echo "Building ${RAG_SERVICE} image (reuses the kg-extra image)..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build "$RAG_SERVICE"

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
set +e
# shellcheck disable=SC2086  # intentional word-splitting of the arg string
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" run --rm "$RAG_SERVICE" ${RAG_CLI_ARGS}
RUN_RC=$?
set -e

echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down 2>/dev/null || true

echo "=============================================="
echo "Stage finished (rc=${RUN_RC}) at $(date)"
echo "=============================================="
exit $RUN_RC
