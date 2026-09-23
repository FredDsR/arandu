#!/bin/bash
# =============================================================================
# Arandu SLURM Container Library (Podman Pods)
#
# Shared container orchestration for all SLURM jobs on the PCAD cluster.
# Replaces Docker Compose with native Podman pods:
#   - Each job runs in an isolated Podman pod (named arandu-pod-$SLURM_JOB_ID).
#   - Ollama sidecar and worker container share the pod's network namespace,
#     so they communicate over localhost (127.0.0.1) and via the alias 'ollama'.
#   - GPU acceleration uses CDI (--device nvidia.com/gpu=all).
#   - Teardown traps stop the pod cleanly on normal exit, timeout, or scancel.
#   - Images are stored in user scratch (/scratch/$USER/.containers/storage).
# =============================================================================

ARANDU_JOB_ID="${SLURM_JOB_ID:-local}"
ARANDU_POD_NAME="arandu-pod-${ARANDU_JOB_ID}"
OLLAMA_CONTAINER_NAME="ollama-${ARANDU_JOB_ID}"
WORKER_CONTAINER_NAME="arandu-worker-${ARANDU_JOB_ID}"

# -----------------------------------------------------------------------------
# Teardown Trap Functions
# -----------------------------------------------------------------------------

arandu_teardown() {
    echo ""
    echo "[cleanup] tearing down Podman pod (${ARANDU_POD_NAME})..."
    podman pod rm -f "${ARANDU_POD_NAME}" 2>/dev/null || true
}

arandu_on_signal() {
    local sig="$1"
    local exit_code="$2"
    trap '' INT TERM
    trap - EXIT
    echo ""
    echo "[cleanup] caught ${sig} (SLURM timeout or cancellation); tearing down..."
    arandu_teardown
    exit "${exit_code}"
}

arandu_arm_teardown_traps() {
    trap arandu_teardown EXIT
    trap 'arandu_on_signal SIGINT 130' INT
    trap 'arandu_on_signal SIGTERM 143' TERM
}

# -----------------------------------------------------------------------------
# Disk Preflight & Cleanup
# -----------------------------------------------------------------------------

arandu_preflight_and_clean() {
    local min_gb="${MIN_DISK_GB:-15}"
    echo ""
    echo "Cleaning unused Podman containers and dangling images..."
    podman container prune -f 2>/dev/null || true
    podman image prune -f 2>/dev/null || true

    if [ -n "${OLLAMA_MODELS_DIR:-}" ] && [ -d "$OLLAMA_MODELS_DIR" ]; then
        find "$OLLAMA_MODELS_DIR" -name "*-partial" -delete 2>/dev/null || true
        find "$OLLAMA_MODELS_DIR" -name "*.tmp" -delete 2>/dev/null || true
    fi

    local storage_root
    storage_root=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "/scratch/$USER")
    [ -d "$storage_root" ] || storage_root="/scratch/$USER"

    local avail_kb avail_gb
    avail_kb=$(df --output=avail "$storage_root" 2>/dev/null | tail -1 | tr -d ' ' || true)
    avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
    echo "Podman storage: $storage_root: ${avail_gb} GB available (min ${min_gb} GB)"

    if [ "${avail_kb:-0}" -gt 0 ] && [ "$avail_gb" -lt "$min_gb" ]; then
        echo "ERROR: not enough disk on $storage_root (${avail_gb} GB < ${min_gb} GB)." >&2
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Image Build
# -----------------------------------------------------------------------------

arandu_build_image() {
    local image_tag="$1"
    local dockerfile="$2"
    local context_dir="${3:-$PROJECT_DIR}"

    echo ""
    echo "Building ${image_tag} image (Dockerfile: ${dockerfile})..."
    podman build -t "$image_tag" -f "${context_dir}/${dockerfile}" "$context_dir"
}

# -----------------------------------------------------------------------------
# Pod Initialization
# -----------------------------------------------------------------------------

arandu_init_pod() {
    echo ""
    echo "Initializing Podman pod: ${ARANDU_POD_NAME}..."
    podman pod rm -f "${ARANDU_POD_NAME}" 2>/dev/null || true
    podman pod create \
        --name "${ARANDU_POD_NAME}" \
        --shm-size 16g \
        --add-host ollama:127.0.0.1
    arandu_arm_teardown_traps
}

# -----------------------------------------------------------------------------
# Ollama Sidecar Lifecycle
# -----------------------------------------------------------------------------

arandu_start_ollama() {
    local model_id="$1"
    local use_gpu="${2:-false}"
    local parallel="${3:-3}"
    local ctx_len="${4:-}"

    echo ""
    echo "Starting Ollama sidecar (GPU: ${use_gpu}, model: ${model_id})..."

    local gpu_args=()
    if [ "$use_gpu" = "true" ]; then
        gpu_args+=("--device" "nvidia.com/gpu=all")
    fi

    local env_args=(
        "-e" "OLLAMA_HOST=0.0.0.0"
        "-e" "OLLAMA_KEEP_ALIVE=5m"
        "-e" "OLLAMA_NUM_PARALLEL=${parallel}"
    )
    if [ -n "$ctx_len" ]; then
        env_args+=("-e" "OLLAMA_CONTEXT_LENGTH=${ctx_len}")
    fi

    podman run -d \
        --pod "${ARANDU_POD_NAME}" \
        --name "${OLLAMA_CONTAINER_NAME}" \
        "${gpu_args[@]}" \
        "${env_args[@]}" \
        -v "${OLLAMA_MODELS_DIR}:/root/.ollama:rw" \
        ollama/ollama:latest

    local ready=false
    echo "Waiting for Ollama to become ready..."
    for i in {1..30}; do
        if podman exec "${OLLAMA_CONTAINER_NAME}" ollama list &>/dev/null; then
            ready=true
            break
        fi
        echo "  Waiting for Ollama... ($i/30)"
        sleep 5
    done

    if [ "$ready" = false ]; then
        echo "ERROR: Ollama failed to start after 30 attempts" >&2
        exit 1
    fi

    echo "Pulling model: ${model_id}..."
    podman exec "${OLLAMA_CONTAINER_NAME}" ollama pull "${model_id}"
}

# -----------------------------------------------------------------------------
# Worker Container Execution
# -----------------------------------------------------------------------------

arandu_run_worker() {
    local image_tag="$1"
    local use_gpu="$2"
    shift 2
    local cmd=("$@")

    local gpu_args=()
    if [ "$use_gpu" = "true" ]; then
        gpu_args+=("--device" "nvidia.com/gpu=all")
    fi

    # Forward all pipeline and authentication env vars
    local env_args=()
    while IFS= read -r var; do
        [ -n "$var" ] && env_args+=("-e" "$var")
    done < <(compgen -v | grep -E '^(ARANDU_|OPENAI_|NVIDIA_|CUDA_|HF_HOME|TRANSFORMERS_CACHE|PIPELINE_ID)')

    # Mounts
    local vol_args=(
        "-v" "${ARANDU_RESULTS_DIR}:/app/results:rw"
        "-v" "${ARANDU_HF_CACHE_DIR}:/app/.cache/huggingface:rw"
    )
    if [ -n "${ARANDU_INPUT_DIR:-}" ] && [ -d "$ARANDU_INPUT_DIR" ]; then
        vol_args+=("-v" "${ARANDU_INPUT_DIR}:/app/input:ro")
    fi
    if [ -n "${ARANDU_CREDENTIALS_DIR:-}" ] && [ -d "$ARANDU_CREDENTIALS_DIR" ]; then
        vol_args+=("-v" "${ARANDU_CREDENTIALS_DIR}:/app/credentials:rw")
    fi

    echo ""
    echo "Running worker in pod ${ARANDU_POD_NAME}: arandu ${cmd[*]}"
    echo "=============================================="

    # Run in background and wait so traps can intercept SIGTERM from SLURM
    set +e
    podman run --rm \
        --pod "${ARANDU_POD_NAME}" \
        --name "${WORKER_CONTAINER_NAME}" \
        "${gpu_args[@]}" \
        "${vol_args[@]}" \
        "${env_args[@]}" \
        "$image_tag" \
        "${cmd[@]}" &
    local run_pid=$!
    wait "$run_pid"
    local run_rc=$?
    set -e

    return "$run_rc"
}
