#!/bin/bash
#SBATCH --job-name=gtranscriber-kg
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber Knowledge Graph Construction SLURM Job Script
#
# Builds knowledge graphs from transcription results using AutoSchemaKG.
# Includes an Ollama sidecar container for local LLM inference.
#
# Required: Transcription results must exist in GTRANSCRIBER_RESULTS_DIR
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# KG Construction settings (GTRANSCRIBER_KG_ prefix)
export GTRANSCRIBER_KG_PROVIDER="${KG_PROVIDER:-ollama}"
export GTRANSCRIBER_KG_MODEL_ID="${KG_MODEL:-llama3.1:8b}"
# Use ollama service name within Docker network (sidecar container)
export GTRANSCRIBER_KG_OLLAMA_URL="${KG_OLLAMA_URL:-http://ollama:11434/v1}"
export GTRANSCRIBER_KG_MERGE_GRAPHS="${KG_MERGE_GRAPHS:-true}"
export GTRANSCRIBER_KG_OUTPUT_FORMAT="${KG_OUTPUT_FORMAT:-graphml}"
export GTRANSCRIBER_KG_LANGUAGE="${KG_LANGUAGE:-pt}"
export GTRANSCRIBER_KG_WORKERS="${WORKERS:-8}"

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
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "KG Provider:   $GTRANSCRIBER_KG_PROVIDER"
echo "KG Model:      $GTRANSCRIBER_KG_MODEL_ID"
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
# Run KG Construction via Docker with Ollama sidecar
# -----------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

echo ""
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --profile kg build gtranscriber-kg

echo ""
echo "Starting Ollama sidecar and pulling model..."
echo "=============================================="

# Start Ollama in background and wait for it to be healthy
docker compose -f "$COMPOSE_FILE" --profile kg up -d ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T ollama ollama list &>/dev/null; then
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
    docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull "$GTRANSCRIBER_KG_MODEL_ID"
fi

echo ""
echo "Starting KG construction process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile kg up gtranscriber-kg --abort-on-container-exit

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile kg down

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
