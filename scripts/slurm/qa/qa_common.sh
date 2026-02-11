#!/bin/bash
# =============================================================================
# G-Transcriber QA Generation Common Job Script
#
# This script contains the shared logic for all QA generation SLURM scripts.
# It should be sourced from partition-specific scripts, not run directly.
#
# Required environment variables (set by partition scripts):
#   GTRANSCRIBER_QA_WORKERS - Number of parallel workers
#
# Optional environment variables:
#   GTRANSCRIBER_QA_MODEL_ID - Ollama model to use (default: llama3.1:8b)
#   GTRANSCRIBER_QA_PROVIDER - LLM provider (default: ollama)
#   GTRANSCRIBER_QA_OLLAMA_URL - Ollama API URL (default: http://ollama:11434/v1)
#   GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT - Questions per document (default: 10)
#   USE_GPU_OLLAMA - Set to "true" to use GPU-accelerated Ollama (default: false)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# QA Generation settings (support override from environment)
export GTRANSCRIBER_QA_PROVIDER="${GTRANSCRIBER_QA_PROVIDER:-ollama}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-llama3.1:8b}"
export GTRANSCRIBER_QA_OLLAMA_URL="${GTRANSCRIBER_QA_OLLAMA_URL:-http://ollama:11434/v1}"
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT="${GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT:-10}"
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-4}"

# GPU mode for Ollama (partition scripts set this)
USE_GPU_OLLAMA="${USE_GPU_OLLAMA:-false}"

# Directories
export GTRANSCRIBER_RESULTS_DIR="${GTRANSCRIBER_RESULTS_DIR:-$PROJECT_DIR/results}"
export GTRANSCRIBER_HF_CACHE_DIR="${GTRANSCRIBER_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "G-Transcriber QA Generation Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-qa-generation}"
echo "Partition:     ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "QA Provider:   $GTRANSCRIBER_QA_PROVIDER"
echo "QA Model:      $GTRANSCRIBER_QA_MODEL_ID"
echo "Ollama GPU:    $USE_GPU_OLLAMA"
echo "Questions/Doc: $GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT"
echo "Workers:       $GTRANSCRIBER_QA_WORKERS"
echo "Results Dir:   $GTRANSCRIBER_RESULTS_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

if [ ! -d "$GTRANSCRIBER_RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $GTRANSCRIBER_RESULTS_DIR"
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
    DOCKER_PROFILE="qa-gpu"
    OLLAMA_SERVICE="ollama-gpu"
else
    DOCKER_PROFILE="qa"
    OLLAMA_SERVICE="ollama"
fi

# -----------------------------------------------------------------------------
# Run QA Generation via Docker with Ollama sidecar
# -----------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# Clean up any orphan containers from previous runs to avoid conflicts
echo ""
echo "Cleaning up any orphan containers from previous runs..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down --remove-orphans 2>/dev/null || true

echo ""
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build gtranscriber-qa

echo ""
echo "Starting Ollama sidecar ($OLLAMA_SERVICE) and pulling model..."
echo "=============================================="

# Start Ollama in background and wait for it to be healthy
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
        echo "Ollama is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 5
done

# Pull the model if using Ollama provider
if [ "$GTRANSCRIBER_QA_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Pulling model: $GTRANSCRIBER_QA_MODEL_ID"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama pull "$GTRANSCRIBER_QA_MODEL_ID"
fi

echo ""
echo "Starting QA generation process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up gtranscriber-qa --abort-on-container-exit

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "G-Transcriber QA Generation Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Results Dir:   $GTRANSCRIBER_RESULTS_DIR"

# Count generated files
QA_COUNT=$(find "$GTRANSCRIBER_RESULTS_DIR" -name "*_qa.json" 2>/dev/null | wc -l)
echo "QA Records:    $QA_COUNT files generated"
echo "=============================================="
