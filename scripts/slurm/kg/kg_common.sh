#!/bin/bash
# =============================================================================
# Arandu Knowledge Graph Construction Common Job Script
#
# This script contains the shared logic for all KG construction SLURM scripts.
# It should be sourced from partition-specific scripts, not run directly.
#
# Optional environment variables:
#   ARANDU_KG_MODEL_ID - Ollama model to use (default: llama3.1:8b)
#   ARANDU_KG_PROVIDER - LLM provider (default: ollama)
#   ARANDU_KG_OLLAMA_URL - Ollama API URL (default: http://ollama:11434/v1)
#   ARANDU_KG_BACKEND - KGC backend (default: atlas)
#   ARANDU_KG_TEMPERATURE - LLM temperature (default: 0.5)
#   ARANDU_KG_LANGUAGE - Language for KG extraction (default: pt)
#   USE_GPU_OLLAMA - Set to "true" to use GPU-accelerated Ollama (default: false)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# KG Construction settings (support override from environment)
export ARANDU_KG_PROVIDER="${ARANDU_KG_PROVIDER:-ollama}"
export ARANDU_KG_MODEL_ID="${ARANDU_KG_MODEL_ID:-llama3.1:8b}"
export ARANDU_KG_OLLAMA_URL="${ARANDU_KG_OLLAMA_URL:-http://ollama:11434/v1}"
export ARANDU_KG_BACKEND="${ARANDU_KG_BACKEND:-atlas}"
export ARANDU_KG_TEMPERATURE="${ARANDU_KG_TEMPERATURE:-0.5}"
export ARANDU_KG_LANGUAGE="${ARANDU_KG_LANGUAGE:-pt}"

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
echo "Arandu KG Construction Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-kg-construction}"
echo "Partition:     ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "KG Provider:   $ARANDU_KG_PROVIDER"
echo "KG Model:      $ARANDU_KG_MODEL_ID"
echo "Ollama GPU:    $USE_GPU_OLLAMA"
echo "Backend:       $ARANDU_KG_BACKEND"
echo "Temperature:   $ARANDU_KG_TEMPERATURE"
echo "Language:      $ARANDU_KG_LANGUAGE"
echo "Results Dir:   $ARANDU_RESULTS_DIR"
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

# -----------------------------------------------------------------------------
# Determine Docker profile based on GPU mode
# -----------------------------------------------------------------------------
if [ "$USE_GPU_OLLAMA" = "true" ]; then
    DOCKER_PROFILE="kg-gpu"
    OLLAMA_SERVICE="ollama-gpu"
else
    DOCKER_PROFILE="kg"
    OLLAMA_SERVICE="ollama"
fi

# -----------------------------------------------------------------------------
# Run KG Construction via Podman with Ollama sidecar
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

# Build image (uses Dockerfile.kg with atlas-rag)
arandu_build_image "arandu-kg:latest" "Dockerfile.kg"

# Initialize isolated pod
arandu_init_pod

# Start Ollama sidecar if using ollama provider
if [ "$ARANDU_KG_PROVIDER" = "ollama" ]; then
    arandu_start_ollama "$ARANDU_KG_MODEL_ID" "$USE_GPU_OLLAMA" 3
fi

echo "Starting KG construction process..."
echo "=============================================="

arandu_run_worker "arandu-kg:latest" false \
    build-kg /app/results --id "$PIPELINE_ID"
KG_RC=$?

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Arandu KG Construction Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Results Dir:   $ARANDU_RESULTS_DIR"

# Show graph statistics if atlas output exists
ATLAS_GRAPH=$(find "$ARANDU_RESULTS_DIR" -path "*/kg/atlas_output/*.graphml" -print -quit 2>/dev/null)
if [ -n "$ATLAS_GRAPH" ]; then
    echo "Atlas Output:  graphml files created"
fi

# Count individual graphs
GRAPH_COUNT=$(find "$ARANDU_RESULTS_DIR" -name "*.graphml" 2>/dev/null | wc -l)
echo "Total Graphs:  $GRAPH_COUNT files generated"
echo "=============================================="
