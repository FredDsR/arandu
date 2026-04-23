#!/bin/bash
# Watch a SLURM job and notify when it finishes.
#
# Usage: ./scripts/slurm/watch-job.sh <job_id> [interval_seconds]
#
# Examples:
#   ./scripts/slurm/watch-job.sh 774591
#   ./scripts/slurm/watch-job.sh 774591 60

set -euo pipefail

JOB_ID="${1:?Usage: $0 <job_id> [interval_seconds]}"
INTERVAL="${2:-300}"
SSH_HOST="fdsreckziegel@pcad.inf.ufrgs.br"

echo "Watching SLURM job $JOB_ID (checking every ${INTERVAL}s)..."

while true; do
    OUTPUT=$(ssh "$SSH_HOST" "squeue -j $JOB_ID -h -o '%T'" 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(date): SSH failed, retrying..."
    elif [ -z "$OUTPUT" ]; then
        notify-send "SLURM Job $JOB_ID" "Job finished (no longer in queue)" 2>/dev/null || true
        echo "$(date): Job $JOB_ID finished!"
        break
    else
        echo "$(date): Job $JOB_ID — $OUTPUT"
    fi

    sleep "$INTERVAL"
done
