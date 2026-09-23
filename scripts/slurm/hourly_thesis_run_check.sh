#!/usr/bin/env bash

set -euo pipefail

SSH_HOST="fdsreckziegel@pcad.inf.ufrgs.br"
REMOTE_DIR="~/etno-kgc-preprocessing"
LOG_PATH="logs/thesis-run-01-hourly-monitor.log"
JOB_IDS="798077,798078,798079,798080"
TARGET_MINUTE=55
INTERVAL_SECONDS=3600

mkdir -p "$(dirname "$LOG_PATH")"

run_check() {
    local now output stop_reason=""
    now="$(date -Is)"

    output="$(
        ssh "$SSH_HOST" "
            cd $REMOTE_DIR || exit 1
            printf 'TIME\t%s\n' \"\$(date -Is)\"
            printf '\nSQUEUE\n'
            squeue -u fdsreckziegel -o '%.10i %.12P %.30j %.10T %.10M %.10L %R'
            printf '\nSACCT\n'
            sacct -j $JOB_IDS --format=JobID,JobName%30,Partition,State,Start,End,Elapsed,ExitCode -X -P
            printf '\nLOGS\n'
            find logs -maxdepth 1 -type f \\
                \\( -name '*798077*' -o -name '*798078*' -o -name '*798079*' -o -name '*798080*' \\) \\
                | sort
            printf '\nTAILS\n'
            for f in \$(find logs -maxdepth 1 -type f \\
                \\( -name '*798077*' -o -name '*798078*' -o -name '*798079*' -o -name '*798080*' \\) \\
                | sort); do
                printf '\n===== %s =====\n' \"\$f\"
                tail -n 40 \"\$f\"
            done
        " 2>&1
    )"

    {
        printf '\n===== LOCAL_CHECK %s =====\n' "$now"
        printf '%s\n' "$output"
    } >> "$LOG_PATH"

    if printf '%s\n' "$output" | grep -Eq '^798080\|.*\|COMPLETED\|'; then
        stop_reason="rag-analysis completed"
    elif printf '%s\n' "$output" | grep -Eq '^(798077|798078|798079|798080)\|.*\|(FAILED|CANCELLED|TIMEOUT)\|'; then
        stop_reason="terminal state detected"
    fi

    if [[ -n "$stop_reason" ]]; then
        printf '\n===== MONITOR_STOP %s %s =====\n' "$(date -Is)" "$stop_reason" >> "$LOG_PATH"
        notify-send "thesis-run-01 monitor" "$stop_reason" >/dev/null 2>&1 || true
        return 1
    fi

    return 0
}

seconds_until_next_target() {
    local minute second delay
    minute="$(date +%M)"
    second="$(date +%S)"
    delay=$(( (10#$TARGET_MINUTE - 10#$minute) * 60 - 10#$second ))
    if (( delay <= 0 )); then
        delay=$(( delay + 3600 ))
    fi
    printf '%s\n' "$delay"
}

initial_delay="$(seconds_until_next_target)"
printf '===== MONITOR_START %s next_check_in=%ss log=%s =====\n' \
    "$(date -Is)" "$initial_delay" "$LOG_PATH" >> "$LOG_PATH"
sleep "$initial_delay"

while true; do
    if ! run_check; then
        break
    fi
    sleep "$INTERVAL_SECONDS"
done
