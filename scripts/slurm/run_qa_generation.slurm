#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber QA Generation SLURM Job Script
#
# Generates synthetic QA pairs from transcription results using LLM.
# Includes an Ollama sidecar container for local LLM inference.
#
# Required: Transcription results must exist in GTRANSCRIBER_RESULTS_DIR
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# QA Generation settings (GTRANSCRIBER_QA_ prefix)
export GTRANSCRIBER_QA_PROVIDER="${QA_PROVIDER:-ollama}"
export GTRANSCRIBER_QA_MODEL_ID="${QA_MODEL:-llama3.1:8b}"
# Use ollama service name within Docker network (sidecar container)
export GTRANSCRIBER_QA_OLLAMA_URL="${QA_OLLAMA_URL:-http://ollama:11434/v1}"
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT="${QUESTIONS_PER_DOCUMENT:-10}"
export GTRANSCRIBER_QA_WORKERS="${WORKERS:-4}"

# Directories
export GTRANSCRIBER_RESULTS_DIR="${GTRANSCRIBER_RESULTS_DIR:-$PROJECT_DIR/results}"
export GTRANSCRIBER_QA_DIR="${GTRANSCRIBER_QA_DIR:-$PROJECT_DIR/qa_dataset}"
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
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "QA Provider:   $GTRANSCRIBER_QA_PROVIDER"
echo "QA Model:      $GTRANSCRIBER_QA_MODEL_ID"
echo "Questions/Doc: $GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT"
echo "Workers:       $GTRANSCRIBER_QA_WORKERS"
echo "Results Dir:   $GTRANSCRIBER_RESULTS_DIR"
echo "QA Output:     $GTRANSCRIBER_QA_DIR"
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
mkdir -p "$GTRANSCRIBER_QA_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"
mkdir -p logs

# -----------------------------------------------------------------------------
# Export SLURM_JOB_ID for container naming
# -----------------------------------------------------------------------------
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# -----------------------------------------------------------------------------
# Run QA Generation via Docker with Ollama sidecar
# -----------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

echo ""
echo "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --profile qa build gtranscriber-qa

echo ""
echo "Starting Ollama sidecar and pulling model..."
echo "=============================================="

# Start Ollama in background and wait for it to be healthy
docker compose -f "$COMPOSE_FILE" --profile qa up -d ollama

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
if [ "$GTRANSCRIBER_QA_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Pulling model: $GTRANSCRIBER_QA_MODEL_ID"
    docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull "$GTRANSCRIBER_QA_MODEL_ID"
fi

echo ""
echo "Starting QA generation process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile qa up gtranscriber-qa --abort-on-container-exit

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile qa down

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "G-Transcriber QA Generation Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "QA Output:     $GTRANSCRIBER_QA_DIR"

# Count generated files
QA_COUNT=$(find "$GTRANSCRIBER_QA_DIR" -name "qa_*.json" 2>/dev/null | wc -l)
echo "QA Records:    $QA_COUNT files generated"
echo "=============================================="
