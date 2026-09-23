#!/bin/bash
# =============================================================================
# Arandu CEP QA Generation Common Job Script
#
# This script contains the shared logic for all CEP generation SLURM scripts.
# It should be sourced from partition-specific scripts, not run directly.
#
# Required environment variables (set by partition scripts):
#   ARANDU_QA_WORKERS - Number of parallel workers
#
# Optional environment variables:
#   ARANDU_QA_MODEL_ID - Ollama model to use (default: qwen3:14b)
#   ARANDU_QA_PROVIDER - LLM provider (default: ollama)
#   ARANDU_QA_OLLAMA_URL - Ollama API URL (default: http://ollama:11434/v1)
#   ARANDU_CEP_BLOOM_DISTRIBUTION - JSON pairs/level (default: 3/1/1/1)
#   ARANDU_CEP_LANGUAGE - Language for prompts (default: pt)
#   USE_GPU_OLLAMA - Set to "true" to use GPU-accelerated Ollama (default: false)
#
# NOTE: This job only generates CEP pairs. Validation is a separate step run by
# the judge-qa SLURM job (scripts/slurm/judge/), configured via ARANDU_JUDGE_*.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# QA Generation settings (support override from environment)
export ARANDU_QA_PROVIDER="${ARANDU_QA_PROVIDER:-ollama}"
export ARANDU_QA_MODEL_ID="${ARANDU_QA_MODEL_ID:-qwen3:14b}"
export ARANDU_QA_OLLAMA_URL="${ARANDU_QA_OLLAMA_URL:-http://ollama:11434/v1}"
export ARANDU_QA_WORKERS="${ARANDU_QA_WORKERS:-4}"

# CEP-specific settings
# Bloom pairs/level as JSON integer counts. The per-chunk ladder size is the
# sum of these counts; the default is the thesis 3/1/1/1 split. Set the var
# before sbatch to override (e.g. '{"remember": 2, "understand": 2, "analyze": 1, "evaluate": 1}').
DEFAULT_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'
export ARANDU_CEP_BLOOM_DISTRIBUTION="${ARANDU_CEP_BLOOM_DISTRIBUTION:-$DEFAULT_BLOOM_DISTRIBUTION}"
export ARANDU_CEP_LANGUAGE="${ARANDU_CEP_LANGUAGE:-pt}"

# GPU mode for Ollama (partition scripts set this)
USE_GPU_OLLAMA="${USE_GPU_OLLAMA:-false}"

# Directories
export ARANDU_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
export ARANDU_HF_CACHE_DIR="${ARANDU_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Arandu CEP QA Generation Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-cep-qa-generation}"
echo "Partition:     ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "QA Provider:   $ARANDU_QA_PROVIDER"
echo "QA Model:      $ARANDU_QA_MODEL_ID"
echo "Ollama GPU:    $USE_GPU_OLLAMA"
echo "Bloom dist:    $ARANDU_CEP_BLOOM_DISTRIBUTION"
echo "Workers:       $ARANDU_QA_WORKERS"
echo "Results Dir:   $ARANDU_RESULTS_DIR"
echo "=============================================="
echo "CEP Language:  $ARANDU_CEP_LANGUAGE"
echo "(validation runs separately via the judge-qa job)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

if [ ! -d "$ARANDU_RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $ARANDU_RESULTS_DIR"
    echo "Please run transcription first."
    exit 1
fi

# Create output directories
mkdir -p "$OLLAMA_MODELS_DIR"
mkdir -p logs

# -----------------------------------------------------------------------------
# Export SLURM_JOB_ID for container naming
# -----------------------------------------------------------------------------
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
export PIPELINE_ID="${PIPELINE_ID:-}"

# Set CEP_REBUILD=1 to clear stale CEP outputs + checkpoint first (e.g. to
# regenerate after a corpus dedup so dropped sources do not linger from a prior
# run). Only the exact value "1" enables it (so CEP_REBUILD=0 is a real off).
# Forwarded to the compose command as ${CEP_REBUILD_FLAG:-}.
if [ "${CEP_REBUILD:-}" = "1" ]; then
    export CEP_REBUILD_FLAG="--rebuild"
else
    export CEP_REBUILD_FLAG=""
fi

# -----------------------------------------------------------------------------
# Determine Docker profile based on GPU mode
# -----------------------------------------------------------------------------
if [ "$USE_GPU_OLLAMA" = "true" ]; then
    DOCKER_PROFILE="cep-gpu"
    OLLAMA_SERVICE="ollama-gpu"
else
    DOCKER_PROFILE="cep"
    OLLAMA_SERVICE="ollama"
fi

# -----------------------------------------------------------------------------
# Run CEP QA Generation via Podman with Ollama sidecar
# -----------------------------------------------------------------------------
CONTAINER_LIB="${SLURM_SUBMIT_DIR:-$PROJECT_DIR}/scripts/slurm/container_lib.sh"
if [ ! -f "$CONTAINER_LIB" ]; then
    echo "ERROR: $CONTAINER_LIB not found; refusing to run without container_lib.sh." >&2
    exit 1
fi
# shellcheck source=scripts/slurm/container_lib.sh
source "$CONTAINER_LIB"

# Preflight + cleanup
arandu_preflight_and_clean

# Build image
arandu_build_image "arandu:latest" "Dockerfile"

# Initialize isolated pod
arandu_init_pod

# Start Ollama sidecar if using ollama provider
if [ "$ARANDU_QA_PROVIDER" = "ollama" ]; then
    arandu_start_ollama "$ARANDU_QA_MODEL_ID" "$USE_GPU_OLLAMA" "${ARANDU_QA_WORKERS:-2}"
fi

CEP_CMD=(
    "generate-cep-qa"
    "/app/results"
    "--id" "$PIPELINE_ID"
    "--workers" "${ARANDU_QA_WORKERS:-2}"
)
[ -n "${CEP_REBUILD_FLAG:-}" ] && CEP_CMD+=("$CEP_REBUILD_FLAG")

arandu_run_worker "arandu:latest" false "${CEP_CMD[@]}"
CEP_RC=$?

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Arandu CEP QA Generation Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Results Dir:   $ARANDU_RESULTS_DIR"

# Count generated files
CEP_COUNT=$(find "$ARANDU_RESULTS_DIR" -name "*_cep_qa.json" 2>/dev/null | wc -l)
echo "CEP Records:   $CEP_COUNT files generated"
echo "=============================================="
