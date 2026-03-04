#!/bin/bash
#SBATCH --job-name=arandu-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Arandu Evaluation SLURM Job Script
#
# Evaluates the quality of QA datasets and knowledge graphs.
# This is a CPU-based job that computes metrics without LLM inference.
#
# Required:
#   - Pipeline results in ARANDU_RESULTS_DIR (versioned layout)
#   - QA outputs at results/{pipeline_id}/qa/outputs/
#   - (Optional) KG outputs at results/{pipeline_id}/kg/outputs/
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Evaluation settings (ARANDU_EVAL_ prefix)
export ARANDU_EVAL_METRICS="${EVAL_METRICS:-qa,entity,relation,semantic}"
export ARANDU_EVAL_EMBEDDING_MODEL="${EVAL_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Directories
export ARANDU_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
export ARANDU_EVAL_DIR="${ARANDU_EVAL_DIR:-$PROJECT_DIR/evaluation}"
export ARANDU_HF_CACHE_DIR="${ARANDU_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export PIPELINE_ID="${PIPELINE_ID:-}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Arandu Evaluation Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-evaluation}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "Metrics:       $ARANDU_EVAL_METRICS"
echo "Embed Model:   $ARANDU_EVAL_EMBEDDING_MODEL"
echo "Results Dir:   $ARANDU_RESULTS_DIR"
echo "Pipeline ID:   ${PIPELINE_ID:-<auto>}"
echo "Eval Output:   $ARANDU_EVAL_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Check results directory exists
if [ ! -d "$ARANDU_RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $ARANDU_RESULTS_DIR"
    echo "Please run transcription first."
    exit 1
fi

# Check QA dataset exists inside versioned results layout
QA_COUNT=$(find "$ARANDU_RESULTS_DIR" -path "*/qa/outputs/qa_*.json" 2>/dev/null | wc -l)
if [ "$QA_COUNT" -eq 0 ]; then
    echo "Error: No QA records found in $ARANDU_RESULTS_DIR/*/qa/outputs/"
    echo "Please run QA generation first."
    exit 1
fi
echo "Found $QA_COUNT QA record files"

# Check knowledge graph exists inside versioned results layout (optional)
KG_PATH=$(find "$ARANDU_RESULTS_DIR" -path "*/kg/outputs/corpus_graph.graphml" -print -quit 2>/dev/null)
if [ -n "$KG_PATH" ]; then
    echo "Found corpus knowledge graph: $KG_PATH"
else
    echo "Warning: Corpus graph not found in $ARANDU_RESULTS_DIR/*/kg/outputs/"
    echo "Graph-based metrics will be skipped."
fi

# Create output directory
mkdir -p "$ARANDU_EVAL_DIR"
mkdir -p logs

# -----------------------------------------------------------------------------
# Export SLURM_JOB_ID for container naming
# -----------------------------------------------------------------------------
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# -----------------------------------------------------------------------------
# Run Evaluation via Docker
# -----------------------------------------------------------------------------
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

echo ""
echo "Pruning Docker build cache, unused images, and volumes to free disk space..."
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

echo ""
echo "Building Docker image..."
docker compose -f "$COMPOSE_FILE" --profile evaluate build arandu-eval

echo ""
echo "Starting evaluation process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile evaluate up arandu-eval --abort-on-container-exit

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile evaluate down

# -----------------------------------------------------------------------------
# Job Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Arandu Evaluation Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Eval Output:   $ARANDU_EVAL_DIR"

# Show report if exists
REPORT_FILE="$ARANDU_EVAL_DIR/report.json"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "Evaluation Report Summary:"
    echo "--------------------------"
    # Extract key metrics using python (available in most environments)
    python3 -c "
import json
with open('$REPORT_FILE') as f:
    data = json.load(f)
    print(f\"  Overall Score: {data.get('overall_score', 'N/A'):.3f}\")
    print(f\"  Total Documents: {data.get('total_documents', 'N/A')}\")
    print(f\"  Total QA Pairs: {data.get('total_qa_pairs', 'N/A')}\")
    if data.get('qa_f1_score'):
        print(f\"  QA F1 Score: {data['qa_f1_score']:.3f}\")
" 2>/dev/null || echo "  (Could not parse report)"
fi

echo "=============================================="
