#!/bin/bash
# =============================================================================
# Arandu Emic Judge Common Job Script
#
# Shared logic for the `arandu emic-judge` SLURM scripts. Source this from a
# partition-specific script; do not run directly.
#
# Runs the ordinal `emic_validity` criterion over the CEP pairs of a run and
# writes per-source scores to results/$PIPELINE_ID/emic_judge/outputs/.
#
# Required environment variables:
#   PIPELINE_ID                    - Pipeline directory under results/
#
# Optional environment variables:
#   EMIC_SCOPE                     - all (default) | approved
#   EMIC_RERUN                     - 1 to discard the checkpoint and re-score
#   ARANDU_EMIC_JUDGE_MODEL_ID     - Model (default: qwen3:14b)
#   ARANDU_EMIC_JUDGE_PROVIDER     - Provider (default: ollama)
#   ARANDU_EMIC_JUDGE_BASE_URL     - Base URL (default: sidecar URL)
#   ARANDU_EMIC_JUDGE_TEMPERATURE  - Sampling temperature (default: 0.1)
#   ARANDU_EMIC_JUDGE_WORKERS      - Client-side concurrency (default: 4)
#   USE_GPU_OLLAMA                 - "true" for the ollama-gpu sidecar
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# LLM settings default to the local Ollama sidecar. Setting these explicitly
# shields the job from whatever lives in the repo's .env (which may point
# OPENAI_API_KEY / ARANDU_LLM_BASE_URL at a cloud provider).
#
# NOTE: the model is a METHODOLOGICAL parameter here, not a budget knob. These
# scores are the study's measurement of emic validity and the human annotation
# round reports agreement with them, so a run feeding the agreement study must
# pin the same model the dissertation describes. Do not swap it to "go faster".
export ARANDU_EMIC_JUDGE_MODEL_ID="${ARANDU_EMIC_JUDGE_MODEL_ID:-qwen3:14b}"
export ARANDU_EMIC_JUDGE_PROVIDER="${ARANDU_EMIC_JUDGE_PROVIDER:-ollama}"
export ARANDU_EMIC_JUDGE_BASE_URL="${ARANDU_EMIC_JUDGE_BASE_URL:-http://ollama:11434/v1}"
export ARANDU_EMIC_JUDGE_TEMPERATURE="${ARANDU_EMIC_JUDGE_TEMPERATURE:-0.1}"
export ARANDU_EMIC_JUDGE_LANGUAGE="${ARANDU_EMIC_JUDGE_LANGUAGE:-pt}"

# Scope. "all" (the default, and the decided methodology) scores every pair and
# records its judge-qa verdict, enabling the emic-validity x approval
# cross-tabulation. "approved" scores only canonically-approved pairs.
EMIC_SCOPE="${EMIC_SCOPE:-all}"
case "$EMIC_SCOPE" in
    all|approved) ;;
    *)
        echo "Error: EMIC_SCOPE must be 'all' or 'approved' (got '$EMIC_SCOPE')" >&2
        exit 1
        ;;
esac

# Resume vs rerun. Default resumes: sources already checkpointed are skipped,
# so a re-submission after a wall hit only scores the remainder.
EMIC_RERUN="${EMIC_RERUN:-0}"
case "${EMIC_RERUN,,}" in
    1|true|yes|on) EMIC_RERUN_FLAG="--rerun" ;;
    0|false|no|off|"") EMIC_RERUN_FLAG="--resume" ;;
    *)
        echo "Error: EMIC_RERUN must be 0/1, true/false, yes/no, or on/off (got '$EMIC_RERUN')" >&2
        exit 1
        ;;
esac

USE_GPU_OLLAMA="${USE_GPU_OLLAMA:-false}"

export ARANDU_RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
export ARANDU_HF_CACHE_DIR="${ARANDU_HF_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$PROJECT_DIR/cache/ollama}"

: "${PIPELINE_ID:?PIPELINE_ID env var is required (e.g. 'PIPELINE_ID=thesis-run-01 sbatch ...')}"
export PIPELINE_ID
# Isolate this job's compose project.
#
# Without this every arandu job on the node shares one project (named after the
# deploy directory), so `docker compose --profile emic-gpu down` reaches any
# service in that project matching the profile. The ollama sidecars are listed
# under the emic profiles, so an emic job starting or finishing next to a live
# judge-qa job on the same tupi node would stop that job's sidecar: its
# remaining LLM calls turn into null scores, and its checkpoint records them as
# done, so its own --resume never retries them.
#
# Scoping the project to the job id makes teardown affect only our containers.
# The stage container and its sidecar share this value, so `depends_on` and the
# internal network still resolve.
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-arandu-emic-${SLURM_JOB_ID:-local}}"

INPUT_DIR_HOST="$ARANDU_RESULTS_DIR/$PIPELINE_ID/cep/outputs"
OUTPUT_DIR_HOST="$ARANDU_RESULTS_DIR/$PIPELINE_ID/emic_judge/outputs"

EMIC_CMD=(
    "emic-judge"
    "--id" "$PIPELINE_ID"
    "--scope" "$EMIC_SCOPE"
    "$EMIC_RERUN_FLAG"
)

echo "=============================================="
echo "Arandu Emic Judge Job Started"
echo "=============================================="
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Job Name:       ${SLURM_JOB_NAME:-arandu-emic}"
echo "Partition:      ${SLURM_JOB_PARTITION:-N/A}"
echo "Node:           $(hostname)"
echo "Start Time:     $(date)"
echo "Pipeline ID:    $PIPELINE_ID"
echo "=============================================="
echo "Scope:          $EMIC_SCOPE"
echo "Mode:           ${EMIC_RERUN_FLAG#--}"
echo "Model:          $ARANDU_EMIC_JUDGE_MODEL_ID"
echo "Provider:       $ARANDU_EMIC_JUDGE_PROVIDER"
echo "Base URL:       $ARANDU_EMIC_JUDGE_BASE_URL"
echo "Temperature:    $ARANDU_EMIC_JUDGE_TEMPERATURE"
echo "Workers:        ${ARANDU_EMIC_JUDGE_WORKERS:-<default>}"
echo "Ollama slots:   ${OLLAMA_NUM_PARALLEL:-<compose default>}"
echo "Ollama ctx:     ${OLLAMA_CONTEXT_LENGTH:-<compose default>}"
echo "Ollama GPU:     $USE_GPU_OLLAMA"
echo "Compose proj:   $COMPOSE_PROJECT_NAME"
echo "Input Dir:      $INPUT_DIR_HOST"
echo "Output Dir:     $OUTPUT_DIR_HOST"
echo "=============================================="

cd "$PROJECT_DIR"

if [ ! -d "$INPUT_DIR_HOST" ]; then
    echo "Error: CEP outputs not found at $INPUT_DIR_HOST" >&2
    echo "       Run generate-cep-qa (and judge-qa) for '$PIPELINE_ID' first." >&2
    exit 1
fi

mkdir -p "$OLLAMA_MODELS_DIR" "$ARANDU_HF_CACHE_DIR" logs

export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

if [ "$USE_GPU_OLLAMA" = "true" ]; then
    DOCKER_PROFILE="emic-gpu"
    OLLAMA_SERVICE="ollama-gpu"
else
    DOCKER_PROFILE="emic"
    OLLAMA_SERVICE="ollama"
fi

COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"


# Deploy note: rsyncing this file without container_teardown.sh leaves the job
# unable to start, which is the intended failure. Silently losing the trap would
# mean orphaned GPU containers on the node.
TEARDOWN_LIB="${SLURM_SUBMIT_DIR:-$PROJECT_DIR}/scripts/slurm/container_teardown.sh"
if [ ! -f "$TEARDOWN_LIB" ]; then
    echo "ERROR: $TEARDOWN_LIB not found; refusing to run without the teardown trap." >&2
    echo "       Deploy scripts/slurm/container_teardown.sh alongside this script." >&2
    exit 1
fi
# shellcheck source=scripts/slurm/container_teardown.sh
source "$TEARDOWN_LIB"

# ---------------------------------------------------------------------------
# Disk preflight + cleanup (mirrors rag_common.sh; cluster nodes fill up)
# ---------------------------------------------------------------------------
echo ""
echo "Pruning unused Docker data to free disk space..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" down --remove-orphans 2>/dev/null || true
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
find "$OLLAMA_MODELS_DIR" -name "*-partial" -delete 2>/dev/null || true
find "$OLLAMA_MODELS_DIR" -name "*.tmp" -delete 2>/dev/null || true

MIN_DISK_GB=${MIN_DISK_GB:-15}
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
[ -d "$DOCKER_ROOT" ] || DOCKER_ROOT=/var/lib/docker
AVAIL_KB=$(df --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -d ' ' || true)
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1024 / 1024 ))
echo "Docker storage: $DOCKER_ROOT: ${AVAIL_GB} GB available (min ${MIN_DISK_GB})"
if [ "${AVAIL_KB:-0}" -gt 0 ] && [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
    echo "ERROR: not enough disk on $DOCKER_ROOT (${AVAIL_GB} GB < ${MIN_DISK_GB} GB)." >&2
    exit 1
fi

echo ""
echo "Building arandu-emic image..."
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" build arandu-emic

# From here on containers get started, so arm the teardown traps. (Kept out of
# the validation/build path above so a pre-container `exit 1` cannot run
# `docker compose down` when this job has nothing up.)
arandu_arm_teardown_traps

if [ "$ARANDU_EMIC_JUDGE_PROVIDER" = "ollama" ]; then
    echo ""
    echo "Starting Ollama sidecar ($OLLAMA_SERVICE)..."
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" up -d "$OLLAMA_SERVICE"

    OLLAMA_READY=false
    for i in {1..30}; do
        if docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" ollama list &>/dev/null; then
            OLLAMA_READY=true
            break
        fi
        echo "  Waiting for Ollama... ($i/30)"
        sleep 5
    done
    if [ "$OLLAMA_READY" = false ]; then
        echo "ERROR: Ollama failed to start after 30 attempts" >&2
        exit 1
    fi

    echo "Pulling model: $ARANDU_EMIC_JUDGE_MODEL_ID"
    docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE" \
        ollama pull "$ARANDU_EMIC_JUDGE_MODEL_ID"
fi

echo ""
echo "Running: arandu ${EMIC_CMD[*]}"
echo "=============================================="
# Background + `wait` (not foreground): bash defers signal traps until a
# foreground external command returns, so a foreground run would keep the
# SIGTERM teardown from firing until the container exits (never, on a real
# timeout). See scripts/slurm/container_teardown.sh.
set +e
docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" \
    run --rm arandu-emic "${EMIC_CMD[@]}" &
RUN_PID=$!
wait "$RUN_PID"
EMIC_EXIT=$?
set -e

echo "=============================================="
echo "Arandu Emic Judge Job Completed"
echo "=============================================="
echo "End Time:       $(date)"
echo "Scope:          $EMIC_SCOPE"
echo "Scores in:      $OUTPUT_DIR_HOST"
echo "Exit Code:      $EMIC_EXIT"
echo "=============================================="
# Container teardown is handled by the EXIT trap armed above, so it runs here on
# normal exit AND on a SLURM SIGTERM (timeout/scancel). Do not add a manual
# `docker compose down`; it would just double-run the trap.

exit $EMIC_EXIT
