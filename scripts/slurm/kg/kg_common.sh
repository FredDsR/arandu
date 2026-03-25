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
# Run KG Construction via Docker with Ollama sidecar
# -----------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# Clean up any orphan containers from previous runs to avoid conflicts
echo ""
echo "Cleaning up any orphan containers from previous runs..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down --remove-orphans 2>/dev/null || true

echo ""
echo "Pruning Docker build cache, unused images, and volumes to free disk space..."
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

echo ""
echo "Cleaning up partial Ollama downloads and unused models..."
# Remove partial/interrupted model downloads (blobs/sha256-*-partial)
find "$OLLAMA_MODELS_DIR" -name "*-partial" -delete 2>/dev/null || true
# Remove orphaned temp files left by interrupted pulls
find "$OLLAMA_MODELS_DIR" -name "*.tmp" -delete 2>/dev/null || true

docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE" 2>/dev/null || true
OLLAMA_UP=false
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
        OLLAMA_UP=true
        break
    fi
    echo "  Waiting for Ollama... ($i/30)"
    sleep 5
done
if [ "$OLLAMA_UP" = true ]; then
    REQUIRED_MODELS=("$ARANDU_KG_MODEL_ID")
    INSTALLED=$(docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list 2>/dev/null | tail -n +2 | awk '{print $1}') || true
    for model in $INSTALLED; do
        is_required=false
        for req in "${REQUIRED_MODELS[@]}"; do
            [ "$model" = "$req" ] && is_required=true && break
        done
        if [ "$is_required" = false ]; then
            echo "  Removing unused model: $model"
            docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama rm "$model" 2>/dev/null || true
        fi
    done
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down 2>/dev/null || true
fi

# Fail fast if disk space is critically low (need ~15 GB for build + model)
MIN_DISK_GB=${MIN_DISK_GB:-15}
AVAIL_KB=$(df --output=avail "$PROJECT_DIR" 2>/dev/null | tail -1 | tr -d ' ')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "Available disk space: ${AVAIL_GB} GB (minimum: ${MIN_DISK_GB} GB)"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
    echo "ERROR: Not enough disk space (${AVAIL_GB} GB < ${MIN_DISK_GB} GB). Aborting."
    echo "Tip: manually run 'docker system prune -af --volumes' on the node."
    exit 1
fi

echo ""
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build arandu-kg

echo ""
echo "Starting Ollama sidecar ($OLLAMA_SERVICE) and pulling model..."
echo "=============================================="

# Start Ollama in background and wait for it to be healthy
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
OLLAMA_READY=false
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
        echo "Ollama is ready!"
        OLLAMA_READY=true
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 5
done

if [ "$OLLAMA_READY" = false ]; then
    echo "ERROR: Ollama failed to start after 30 attempts"
    exit 1
fi

# Pull the model if using Ollama provider
if [ "$ARANDU_KG_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Pulling model: $ARANDU_KG_MODEL_ID"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama pull "$ARANDU_KG_MODEL_ID"
fi

echo ""
echo "Starting KG construction process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up arandu-kg --abort-on-container-exit

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
