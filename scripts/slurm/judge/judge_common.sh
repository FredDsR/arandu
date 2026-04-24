#!/bin/bash
# =============================================================================
# Arandu Judge Common Job Script
#
# Shared logic for all judge SLURM scripts (transcription + QA). Source this
# from partition-specific scripts; do not run directly.
#
# Required environment variables:
#   PIPELINE_ID                     - Pipeline directory under results/ (e.g. test-cep-01)
#
# Optional environment variables:
#   JUDGE_SUBCOMMAND                - "judge-transcription" (default) or "judge-qa"
#   ARANDU_JUDGE_VALIDATOR_MODEL    - Validator model (default: qwen3:14b)
#   ARANDU_JUDGE_VALIDATOR_PROVIDER - Validator provider (default: ollama)
#   ARANDU_JUDGE_VALIDATOR_BASE_URL - Validator base URL (default: sidecar URL)
#   ARANDU_JUDGE_LANGUAGE           - Prompt language (default: pt)
#   ARANDU_JUDGE_TEMPERATURE        - LLM sampling temperature (default: 0.3)
#   USE_GPU_OLLAMA                  - "true" to use ollama-gpu sidecar (default: false)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Validator settings — default to the local Ollama sidecar. Setting these
# explicitly shields the job from whatever lives in the repo's .env (which
# may point OPENAI_API_KEY / ARANDU_LLM_BASE_URL at a cloud provider).
export ARANDU_JUDGE_VALIDATOR_MODEL="${ARANDU_JUDGE_VALIDATOR_MODEL:-qwen3:14b}"
export ARANDU_JUDGE_VALIDATOR_PROVIDER="${ARANDU_JUDGE_VALIDATOR_PROVIDER:-ollama}"
export ARANDU_JUDGE_VALIDATOR_BASE_URL="${ARANDU_JUDGE_VALIDATOR_BASE_URL:-http://ollama:11434/v1}"
export ARANDU_JUDGE_LANGUAGE="${ARANDU_JUDGE_LANGUAGE:-pt}"
export ARANDU_JUDGE_TEMPERATURE="${ARANDU_JUDGE_TEMPERATURE:-0.3}"

# Which CLI subcommand to run inside the arandu-judge container
JUDGE_SUBCOMMAND="${JUDGE_SUBCOMMAND:-judge-transcription}"

# GPU mode for Ollama (partition scripts set this)
USE_GPU_OLLAMA="${USE_GPU_OLLAMA:-false}"

# Directories
export ARANDU_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
export ARANDU_HF_CACHE_DIR="${ARANDU_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"

# PIPELINE_ID is required — the judge runs against a specific pipeline's outputs.
: "${PIPELINE_ID:?PIPELINE_ID env var is required (e.g. 'PIPELINE_ID=test-cep-01 sbatch ...')}"
export PIPELINE_ID

# -----------------------------------------------------------------------------
# Assemble the CLI invocation based on the requested subcommand
# -----------------------------------------------------------------------------
case "$JUDGE_SUBCOMMAND" in
    judge-transcription)
        INPUT_DIR_HOST="$ARANDU_RESULTS_DIR/$PIPELINE_ID/transcription/outputs"
        INPUT_DIR_CONTAINER="/app/results/$PIPELINE_ID/transcription/outputs"
        OUTPUT_FILE_CONTAINER="/app/results/$PIPELINE_ID/transcription/judgements.json"
        JUDGE_CMD=(
            "$JUDGE_SUBCOMMAND"
            "$INPUT_DIR_CONTAINER"
            "--language" "$ARANDU_JUDGE_LANGUAGE"
            "--output" "$OUTPUT_FILE_CONTAINER"
        )
        ;;
    judge-qa)
        INPUT_DIR_HOST="$ARANDU_RESULTS_DIR/$PIPELINE_ID/cep/outputs"
        INPUT_DIR_CONTAINER="/app/results/$PIPELINE_ID/cep/outputs"
        OUTPUT_FILE_CONTAINER="/app/results/$PIPELINE_ID/cep/judgements.json"
        JUDGE_CMD=(
            "$JUDGE_SUBCOMMAND"
            "$INPUT_DIR_CONTAINER"
            "--provider" "$ARANDU_JUDGE_VALIDATOR_PROVIDER"
            "--model" "$ARANDU_JUDGE_VALIDATOR_MODEL"
            "--base-url" "$ARANDU_JUDGE_VALIDATOR_BASE_URL"
            "--language" "$ARANDU_JUDGE_LANGUAGE"
            "--output" "$OUTPUT_FILE_CONTAINER"
        )
        ;;
    *)
        echo "Error: Unknown JUDGE_SUBCOMMAND '$JUDGE_SUBCOMMAND'"
        echo "Supported: judge-transcription, judge-qa"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Job Information
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Arandu Judge Job Started"
echo "=============================================="
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Job Name:          ${SLURM_JOB_NAME:-arandu-judge}"
echo "Partition:         ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:              $(hostname)"
echo "Start Time:        $(date)"
echo "Project Dir:       $PROJECT_DIR"
echo "Pipeline ID:       $PIPELINE_ID"
echo "=============================================="
echo "Subcommand:        $JUDGE_SUBCOMMAND"
echo "Validator Model:   $ARANDU_JUDGE_VALIDATOR_MODEL"
echo "Validator Provider:$ARANDU_JUDGE_VALIDATOR_PROVIDER"
echo "Validator URL:     $ARANDU_JUDGE_VALIDATOR_BASE_URL"
echo "Language:          $ARANDU_JUDGE_LANGUAGE"
echo "Ollama GPU:        $USE_GPU_OLLAMA"
echo "Input Dir:         $INPUT_DIR_HOST"
echo "=============================================="

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

if [ ! -d "$INPUT_DIR_HOST" ]; then
    echo "Error: Input directory not found at $INPUT_DIR_HOST"
    echo "Run the upstream step for pipeline '$PIPELINE_ID' first."
    exit 1
fi

mkdir -p "$OLLAMA_MODELS_DIR"
mkdir -p logs

export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# -----------------------------------------------------------------------------
# Docker profile
# -----------------------------------------------------------------------------
if [ "$USE_GPU_OLLAMA" = "true" ]; then
    DOCKER_PROFILE="judge-gpu"
    OLLAMA_SERVICE="ollama-gpu"
else
    DOCKER_PROFILE="judge"
    OLLAMA_SERVICE="ollama"
fi

COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# -----------------------------------------------------------------------------
# Clean up from previous runs
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up any orphan containers from previous runs..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down --remove-orphans 2>/dev/null || true

# -----------------------------------------------------------------------------
# Build + start ollama, pull model (only when using the ollama provider)
# -----------------------------------------------------------------------------
echo ""
echo "Building arandu-judge image..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build arandu-judge

if [ "$ARANDU_JUDGE_VALIDATOR_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Starting Ollama sidecar ($OLLAMA_SERVICE)..."
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE"

    echo "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
            echo "Ollama is ready."
            break
        fi
        echo "  Waiting... ($i/30)"
        sleep 5
    done

    echo ""
    echo "Pulling model: $ARANDU_JUDGE_VALIDATOR_MODEL"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" \
        ollama pull "$ARANDU_JUDGE_VALIDATOR_MODEL"
fi

# -----------------------------------------------------------------------------
# Run the judge
# -----------------------------------------------------------------------------
echo ""
echo "Starting judge process..."
echo "CLI: arandu ${JUDGE_CMD[*]}"
echo "=============================================="

set +e
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" \
    run --rm arandu-judge "${JUDGE_CMD[@]}"
JUDGE_EXIT=$?
set -e

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
echo ""
echo "Cleaning up containers..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Arandu Judge Job Completed"
echo "=============================================="
echo "End Time:       $(date)"
echo "Subcommand:     $JUDGE_SUBCOMMAND"
echo "Judgements:     $ARANDU_RESULTS_DIR/$PIPELINE_ID/$(dirname "${OUTPUT_FILE_CONTAINER#/app/results/$PIPELINE_ID/}")/$(basename "$OUTPUT_FILE_CONTAINER")"
echo "Exit Code:      $JUDGE_EXIT"
echo "=============================================="

exit $JUDGE_EXIT
