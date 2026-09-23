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
# Container teardown trap. Shared with the other steps; see
# scripts/slurm/container_teardown.sh for why it exists and for the two
# conditions it needs to fire in time (background + `wait`, and
# `#SBATCH --signal=B:TERM@60` on the per-stage script).
#
# Deploy note: rsyncing this file without container_teardown.sh leaves the job
# unable to start, which is the intended failure. Silently losing the trap
# would mean orphaned GPU containers on the node.
TEARDOWN_LIB="${SLURM_SUBMIT_DIR:-$PROJECT_DIR}/scripts/slurm/container_teardown.sh"
if [ ! -f "$TEARDOWN_LIB" ]; then
    echo "ERROR: $TEARDOWN_LIB not found; refusing to run without the teardown trap." >&2
    echo "       Deploy scripts/slurm/container_teardown.sh alongside this script." >&2
    exit 1
fi
# shellcheck source=scripts/slurm/container_teardown.sh
source "$TEARDOWN_LIB"

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

CONTAINER_LIB="${SLURM_SUBMIT_DIR:-$PROJECT_DIR}/scripts/slurm/container_lib.sh"
if [ ! -f "$CONTAINER_LIB" ]; then
    echo "ERROR: $CONTAINER_LIB not found; refusing to run without container_lib.sh." >&2
    exit 1
fi
# shellcheck source=scripts/slurm/container_lib.sh
source "$CONTAINER_LIB"

# Preflight + cleanup
arandu_preflight_and_clean

# Build image (reuses Dockerfile.kg)
arandu_build_image "arandu-kg:latest" "Dockerfile.kg"

# Initialize isolated pod
arandu_init_pod

# Ollama sidecar (LLM stages only)
if [ "$RAG_NEEDS_OLLAMA" = "true" ]; then
    arandu_start_ollama "$RAG_OLLAMA_MODEL" true "${OLLAMA_NUM_PARALLEL:-3}" "${OLLAMA_CONTEXT_LENGTH:-}"
fi

# RAG GPU: true if stage needs GPU (arandu-rag), false if arandu-rag-cpu
WORKER_GPU=false
if [ "$RAG_PROFILE" = "rag-gpu" ] || [ "$RAG_SERVICE" = "arandu-rag" ]; then
    WORKER_GPU=true
fi

# Run the stage in the pod
# shellcheck disable=SC2086
arandu_run_worker "arandu-kg:latest" "$WORKER_GPU" ${RAG_CLI_ARGS}
RUN_RC=$?

echo "=============================================="
echo "Stage finished (rc=${RUN_RC}) at $(date)"
echo "=============================================="
exit $RUN_RC
