#!/bin/bash
#SBATCH --job-name=gtranscriber-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber Evaluation SLURM Job Script
#
# Evaluates the quality of QA datasets and knowledge graphs.
# This is a CPU-based job that computes metrics without LLM inference.
#
# Required:
#   - QA dataset in GTRANSCRIBER_QA_DIR
#   - Transcription results in GTRANSCRIBER_RESULTS_DIR
#   - (Optional) Knowledge graph in GTRANSCRIBER_KG_DIR
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Evaluation settings (GTRANSCRIBER_EVAL_ prefix)
export GTRANSCRIBER_EVAL_METRICS="${EVAL_METRICS:-qa,entity,relation,semantic}"
export GTRANSCRIBER_EVAL_EMBEDDING_MODEL="${EVAL_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Directories
export GTRANSCRIBER_RESULTS_DIR="${GTRANSCRIBER_RESULTS_DIR:-$PROJECT_DIR/results}"
export GTRANSCRIBER_QA_DIR="${GTRANSCRIBER_QA_DIR:-$PROJECT_DIR/qa_dataset}"
export GTRANSCRIBER_KG_DIR="${GTRANSCRIBER_KG_DIR:-$PROJECT_DIR/knowledge_graphs}"
export GTRANSCRIBER_EVAL_DIR="${GTRANSCRIBER_EVAL_DIR:-$PROJECT_DIR/evaluation}"
export GTRANSCRIBER_HF_CACHE_DIR="${GTRANSCRIBER_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "G-Transcriber Evaluation Job Started"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Job Name:      ${SLURM_JOB_NAME:-evaluation}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Start Time:    $(date)"
echo "Project Dir:   $PROJECT_DIR"
echo "=============================================="
echo "Metrics:       $GTRANSCRIBER_EVAL_METRICS"
echo "Embed Model:   $GTRANSCRIBER_EVAL_EMBEDDING_MODEL"
echo "QA Dir:        $GTRANSCRIBER_QA_DIR"
echo "Results Dir:   $GTRANSCRIBER_RESULTS_DIR"
echo "KG Dir:        $GTRANSCRIBER_KG_DIR"
echo "Eval Output:   $GTRANSCRIBER_EVAL_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Check QA dataset exists
if [ ! -d "$GTRANSCRIBER_QA_DIR" ]; then
    echo "Error: QA dataset directory not found: $GTRANSCRIBER_QA_DIR"
    echo "Please run QA generation first."
    exit 1
fi

QA_COUNT=$(find "$GTRANSCRIBER_QA_DIR" -name "qa_*.json" 2>/dev/null | wc -l)
if [ "$QA_COUNT" -eq 0 ]; then
    echo "Error: No QA records found in $GTRANSCRIBER_QA_DIR"
    exit 1
fi
echo "Found $QA_COUNT QA record files"

# Check transcription results exist
if [ ! -d "$GTRANSCRIBER_RESULTS_DIR" ]; then
    echo "Warning: Results directory not found: $GTRANSCRIBER_RESULTS_DIR"
    echo "Some metrics may not be available."
fi

# Check knowledge graph exists (optional)
KG_PATH="$GTRANSCRIBER_KG_DIR/corpus_graph.graphml"
if [ -f "$KG_PATH" ]; then
    echo "Found corpus knowledge graph"
else
    echo "Warning: Corpus graph not found at $KG_PATH"
    echo "Graph-based metrics will be skipped."
    KG_PATH=""
fi

# Create output directory
mkdir -p "$GTRANSCRIBER_EVAL_DIR"
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
echo "Building Docker image..."
docker compose -f "$COMPOSE_FILE" --profile evaluate build gtranscriber-eval

echo ""
echo "Starting evaluation process..."
echo "=============================================="

docker compose -f "$COMPOSE_FILE" --profile evaluate up gtranscriber-eval --abort-on-container-exit

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
echo "G-Transcriber Evaluation Job Completed"
echo "=============================================="
echo "End Time:      $(date)"
echo "Eval Output:   $GTRANSCRIBER_EVAL_DIR"

# Show report if exists
REPORT_FILE="$GTRANSCRIBER_EVAL_DIR/report.json"
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
