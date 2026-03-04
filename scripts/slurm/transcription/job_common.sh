#!/bin/bash
# =============================================================================
# Arandu Common Job Script
#
# This script contains the shared logic for all SLURM partition scripts.
# It should be sourced from partition-specific scripts, not run directly.
#
# Required environment variables (set by partition scripts):
#   WORKERS - Number of parallel workers
#   ARANDU_MODEL_ID - Whisper model to use
#
# Optional environment variables:
#   ARANDU_LANGUAGE - Language code for transcription (default: pt)
#   ARANDU_RESULTS_DIR - Custom results directory (default: $PROJECT_DIR/results)
#   PIPELINE_ID - Pipeline run ID for versioned results layout (default: auto-resolved)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# Support both old (WORKERS, CATALOG_FILE) and new (ARANDU_*) variable names
# for backward compatibility
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Support both ARANDU_WORKERS and WORKERS (prioritize ARANDU_*)
if [ -n "${ARANDU_WORKERS:-}" ]; then
    WORKERS="${ARANDU_WORKERS}"
fi
WORKERS="${WORKERS:-1}"

# Support both ARANDU_CATALOG_FILE and CATALOG_FILE
if [ -n "${ARANDU_CATALOG_FILE:-}" ]; then
    CATALOG_FILE="${ARANDU_CATALOG_FILE}"
fi
CATALOG_FILE="${CATALOG_FILE:-catalog.csv}"

# Support custom results directory (final destination after job completes)
FINAL_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"

USE_CPU="${USE_CPU:-false}"
USE_ROCM="${USE_ROCM:-false}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Arandu Job Started"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "Results Dir:   $FINAL_RESULTS_DIR"
echo "Model:         $ARANDU_MODEL_ID"
echo "Language:      ${ARANDU_LANGUAGE:-pt}"
echo "Workers:       $WORKERS"
echo "Catalog:       $CATALOG_FILE"
echo "=============================================="

# -----------------------------------------------------------------------------
# Check GPU availability
# -----------------------------------------------------------------------------
echo ""
echo "Checking GPU availability..."

if [ "$USE_ROCM" = "true" ]; then
    echo "ROCm mode enabled for AMD GPU"
    if [ -e /dev/kfd ]; then
        echo "AMD GPU device found (/dev/kfd)"
        rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not available on host"
    else
        echo "Warning: /dev/kfd not found, AMD GPU may not be available"
    fi
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    GPU_AVAILABLE=true
else
    echo "No NVIDIA GPU detected, will use CPU mode"
    GPU_AVAILABLE=false
    USE_CPU=true
fi

# -----------------------------------------------------------------------------
# Setup working directory
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Create cache directory for Hugging Face models
mkdir -p cache/huggingface

# -----------------------------------------------------------------------------
# Setup working directories
# -----------------------------------------------------------------------------
echo ""
echo "Setting up working directories..."

WORK_INPUT_DIR="$PROJECT_DIR/input"
WORK_RESULTS_DIR="$FINAL_RESULTS_DIR"
WORK_CREDENTIALS_DIR="$PROJECT_DIR"
WORK_HF_CACHE_DIR="$PROJECT_DIR/cache/huggingface"
mkdir -p "$FINAL_RESULTS_DIR"

echo "  Input Dir:       $WORK_INPUT_DIR"
echo "  Results Dir:     $WORK_RESULTS_DIR"
echo "  Credentials Dir: $WORK_CREDENTIALS_DIR"
echo "  HF Cache Dir:    $WORK_HF_CACHE_DIR"

# -----------------------------------------------------------------------------
# Export environment variables for Docker Compose
# Export both old and new variable names for backward compatibility
# -----------------------------------------------------------------------------
export SLURM_JOB_ID
export PIPELINE_ID="${PIPELINE_ID:-}"

# Export both WORKERS and ARANDU_WORKERS
export WORKERS
export ARANDU_WORKERS="$WORKERS"

export ARANDU_MODEL_ID

# Set default language to Portuguese (Brazilian)
export ARANDU_LANGUAGE="${ARANDU_LANGUAGE:-pt}"

# Export both CATALOG_FILE and ARANDU_CATALOG_FILE
export CATALOG_FILE
export ARANDU_CATALOG_FILE="$CATALOG_FILE"

# Export both old and new path variables
export INPUT_DIR="$WORK_INPUT_DIR"
export ARANDU_INPUT_DIR="$WORK_INPUT_DIR"

export RESULTS_DIR="$WORK_RESULTS_DIR"
export ARANDU_RESULTS_DIR="$WORK_RESULTS_DIR"

export CREDENTIALS_DIR="$WORK_CREDENTIALS_DIR"
export ARANDU_CREDENTIALS_DIR="$WORK_CREDENTIALS_DIR"

export HF_CACHE_DIR="$WORK_HF_CACHE_DIR"
export ARANDU_HF_CACHE_DIR="$WORK_HF_CACHE_DIR"

# Set quantization flag (enabled by default for GPU, disabled for CPU)
if [ "$USE_CPU" = "true" ]; then
    export ARANDU_QUANTIZE=false
    export ARANDU_FORCE_CPU=true
    export QUANTIZE_FLAG=""
    export CPU_FLAG="--cpu"
else
    export ARANDU_QUANTIZE=false
    export ARANDU_FORCE_CPU=false
    export QUANTIZE_FLAG="--quantize"
    export CPU_FLAG=""
fi

# -----------------------------------------------------------------------------
# Build Docker image (if needed)
# -----------------------------------------------------------------------------
echo ""
echo "Pruning Docker build cache, unused images, and volumes to free disk space..."
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

echo ""
echo "Building Docker image..."

# Use -f flag to specify docker-compose.yml location from PROJECT_DIR
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

if [ "$USE_ROCM" = "true" ]; then
    docker compose -f "$COMPOSE_FILE" --profile rocm build arandu-rocm
elif [ "$USE_CPU" = "true" ]; then
    docker compose -f "$COMPOSE_FILE" --profile cpu build arandu-cpu
else
    docker compose -f "$COMPOSE_FILE" --profile gpu build arandu
fi

# -----------------------------------------------------------------------------
# Run transcription
# -----------------------------------------------------------------------------
echo ""
echo "Starting transcription process..."
echo "=============================================="

if [ "$USE_ROCM" = "true" ]; then
    echo "Running with AMD ROCm support..."
    docker compose -f "$COMPOSE_FILE" --profile rocm up arandu-rocm --abort-on-container-exit
elif [ "$USE_CPU" = "true" ]; then
    echo "Running in CPU mode..."
    docker compose -f "$COMPOSE_FILE" --profile cpu up arandu-cpu --abort-on-container-exit
else
    echo "Running with NVIDIA GPU support..."
    docker compose -f "$COMPOSE_FILE" --profile gpu up arandu --abort-on-container-exit
fi

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" down

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Arandu Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Results Dir:   $FINAL_RESULTS_DIR"
echo "=============================================="
