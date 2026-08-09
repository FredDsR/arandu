#!/bin/bash
# =============================================================================
# Shared container-teardown trap for SLURM job scripts.
#
# Source this from a `<step>_common.sh`, then call `arandu_arm_teardown_traps`
# once, just before the first container starts.
#
# WHY THIS EXISTS
#
# Containers started with `docker compose run` / `up` are owned by the docker
# daemon, NOT by the job step's process tree. When SLURM ends a job WITHOUT a
# clean script exit (a TIME LIMIT timeout or an `scancel`), it kills the shell
# before any normal teardown runs and the containers keep going on the node as
# ORPHANS: still holding the GPU, still writing outputs, outside any
# allocation, contending with whatever SLURM schedules next. Observed with
# judge-answers 799024, whose TIMEOUT left `ollama-gpu-<jobid>` and
# `<project>-arandu-rag-run-*` alive on tupi2 and degraded another user's job;
# it could not be killed from the head node (ssh is blocked by
# pam_slurm_adopt) and needed admin intervention.
#
# TWO THINGS ARE REQUIRED for the trap to actually fire in time:
#
#   1. The stage command must run in the BACKGROUND and be `wait`-ed on. bash
#      does NOT run a trap while a foreground external command executes; it
#      defers it until that command returns. A foreground `docker compose run`
#      would therefore defer teardown until the container exits, i.e. never on
#      a real timeout. `wait` is interruptible, so backgrounding + `wait` lets
#      the SIGTERM/SIGINT handler run immediately:
#
#          docker compose ... run --rm "$SERVICE" $ARGS &
#          RUN_PID=$!
#          wait "$RUN_PID"
#          RUN_RC=$?
#
#   2. The partition script must carry `#SBATCH --signal=B:TERM@60` so SLURM
#      sends SIGTERM to the batch shell 60s before the limit, well inside
#      KillWait.
#
# Arm the traps LATE (after validation and image build, just before the first
# container starts) so an early `exit 1` does not run `docker compose down`
# while this job has nothing up, which could disturb a co-located job sharing
# the compose project.
#
# Required variables at teardown time: COMPOSE_FILE, DOCKER_PROFILE.
#
# Extracted from scripts/slurm/rag/rag_common.sh (PR #152), which still carries
# its own copy; see the tracking task on unifying the remaining common scripts.
# =============================================================================

arandu_teardown_containers() {
    echo ""
    echo "[cleanup] tearing down containers (profile ${DOCKER_PROFILE})..."
    docker compose -f "$COMPOSE_FILE" --profile "$DOCKER_PROFILE" \
        down --remove-orphans --timeout 10 2>/dev/null || true
}

# $1 = signal name (for the log), $2 = exit code.
arandu_on_teardown_signal() {
    # Ignore further INT/TERM while tearing down (a scancel retry or a short
    # KillWait must not kill the shell mid-`down`) and disarm EXIT so teardown
    # runs exactly once.
    trap '' INT TERM
    trap - EXIT
    echo ""
    echo "[cleanup] caught ${1} (SLURM timeout/scancel); tearing down..."
    arandu_teardown_containers
    exit "${2}"
}

arandu_arm_teardown_traps() {
    trap arandu_teardown_containers EXIT
    trap 'arandu_on_teardown_signal SIGINT 130' INT   # 128 + SIGINT(2)
    trap 'arandu_on_teardown_signal SIGTERM 143' TERM # 128 + SIGTERM(15)
}
