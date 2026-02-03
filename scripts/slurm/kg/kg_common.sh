#!/bin/bash
# =============================================================================
# G-Transcriber Knowledge Graph Construction Common Job Script
#
# This script contains the shared logic for all KG construction SLURM scripts.
# It should be sourced from partition-specific scripts, not run directly.
#
# Required environment variables (set by partition scripts):
#   GTRANSCRIBER_KG_WORKERS - Number of parallel workers
#
# Optional environment variables:
#   GTRANSCRIBER_KG_MODEL_ID - Ollama model to use (default: llama3.1:8b)
#   GTRANSCRIBER_KG_PROVIDER - LLM provider (default: ollama)
#   GTRANSCRIBER_KG_OLLAMA_URL - Ollama API URL (default: http://ollama:11434/v1)
#   GTRANSCRIBER_KG_MERGE_GRAPHS - Merge graphs into corpus graph (default: true)
#   GTRANSCRIBER_KG_OUTPUT_FORMAT - Output format (default: graphml)
#   GTRANSCRIBER_KG_LANGUAGE - Language for KG extraction (default: pt)
#   USE_GPU_OLLAMA - Set to "true" to use GPU-accelerated Ollama (default: false)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# KG Construction settings (support override from environment)
export GTRANSCRIBER_KG_PROVIDER="${GTRANSCRIBER_KG_PROVIDER:-ollama}"
export GTRANSCRIBER_KG_MODEL_ID="${GTRANSCRIBER_KG_MODEL_ID:-llama3.1:8b}"
export GTRANSCRIBER_KG_OLLAMA_URL="${GTRANSCRIBER_KG_OLLAMA_URL:-http://ollama:11434/v1}"
export GTRANSCRIBER_KG_MERGE_GRAPHS="${GTRANSCRIBER_KG_MERGE_GRAPHS:-true}"
export GTRANSCRIBER_KG_OUTPUT_FORMAT="${GTRANSCRIBER_KG_OUTPUT_FORMAT:-graphml}"
export GTRANSCRIBER_KG_LANGUAGE="${GTRANSCRIBER_KG_LANGUAGE:-pt}"
export GTRANSCRIBER_KG_WORKERS="${GTRANSCRIBER_KG_WORKERS:-8}"

# GPU mode for Ollama (partition scripts set this)
USE_GPU_OLLAMA="${USE_GPU_OLLAMA:-false}"

# Directories
export GTRANSCRIBER_RESULTS_DIR="${GTRANSCRIBER_RESULTS_DIR:-$PROJECT_DIR/results}"
export GTRANSCRIBER_KG_DIR="${GTRANSCRIBER_KG_DIR:-$PROJECT_DIR/knowledge_graphs}"
export GTRANSCRIBER_HF_CACHE_DIR="${GTRANSCRIBER_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "G-Transcriber KG Construction Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-kg-construction}"
echo "Partition:     ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "KG Provider:   $GTRANSCRIBER_KG_PROVIDER"
echo "KG Model:      $GTRANSCRIBER_KG_MODEL_ID"
echo "Ollama GPU:    $USE_GPU_OLLAMA"
echo "Merge Graphs:  $GTRANSCRIBER_KG_MERGE_GRAPHS"
echo "Output Format: $GTRANSCRIBER_KG_OUTPUT_FORMAT"
echo "Language:      $GTRANSCRIBER_KG_LANGUAGE"
echo "Workers:       $GTRANSCRIBER_KG_WORKERS"
echo "Results Dir:   $GTRANSCRIBER_RESULTS_DIR"
echo "KG Output:     $GTRANSCRIBER_KG_DIR"
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
mkdir -p "$GTRANSCRIBER_KG_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"
mkdir -p logs

# -----------------------------------------------------------------------------
# Export SLURM_JOB_ID for container naming
# -----------------------------------------------------------------------------
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

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
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build gtranscriber-kg

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
if [ "$GTRANSCRIBER_KG_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Pulling model: $GTRANSCRIBER_KG_MODEL_ID"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama pull "$GTRANSCRIBER_KG_MODEL_ID"
fi

echo ""
echo "Starting KG construction process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up gtranscriber-kg --abort-on-container-exit

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
echo "G-Transcriber KG Construction Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "KG Output:     $GTRANSCRIBER_KG_DIR"

# Show graph statistics if corpus graph exists
if [ -f "$GTRANSCRIBER_KG_DIR/corpus_graph.graphml" ]; then
    echo "Corpus Graph:  corpus_graph.graphml created"
fi

# Count individual graphs
GRAPH_COUNT=$(find "$GTRANSCRIBER_KG_DIR" -name "*.graphml" 2>/dev/null | wc -l)
echo "Total Graphs:  $GRAPH_COUNT files generated"
echo "=============================================="
