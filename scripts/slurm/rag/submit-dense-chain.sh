#!/bin/bash
# =============================================================================
# Submit the dense arms of an existing run as one SLURM dependency chain:
#
#   retrieve-dense -> answer x N_ANSWER_JOBS -> judge-answers x N_JUDGE_JOBS
#
# answer and judge-answers resume from their checkpoints, so each stage can run
# as several jobs: the first one waits for the previous stage, and every
# further job of the stage waits for the job before it to end in any state
# (afterany), picking up what a job cut by the 24h wall left. A job with
# nothing left to do exits after the container start-up.
#
# A stage starts only on success of the stage before it: every answer job needs
# retrieve-dense to succeed (afterok), and every judge-answers job needs the
# last answer job to succeed (afterok), on top of their own afterany links.
# When that cannot be met (retrieve or the last answer job failed or timed
# out), SLURM cancels the waiting jobs (--kill-on-invalid-dep), and no judge
# job runs over a partial answer set.
#
# Before submitting, the checkpoints and run metadata of the three stages are
# copied to *.bak-<BACKUP_TAG> (existing backups are kept).
#
# Run from the repository root on the cluster login node:
#   PIPELINE_ID=thesis-run-02 bash scripts/slurm/rag/submit-dense-chain.sh
#
# Optional environment (defaults shown):
#   DENSE_ARMS     "dense_mpnet128 dense_bgem3_256"   dense arms to retrieve
#   N_ANSWER_JOBS  1              answer jobs in the chain
#   N_JUDGE_JOBS   2              judge-answers jobs in the chain
#   BACKUP_TAG     pre-dense      suffix of the checkpoint backups
#   DRY_RUN        (unset)        "1" prints the sbatch commands instead of
#                                 submitting them (backups are still written)
# =============================================================================

set -euo pipefail

PIPELINE_ID="${PIPELINE_ID:?PIPELINE_ID is required}"
DENSE_ARMS="${DENSE_ARMS:-dense_mpnet128 dense_bgem3_256}"
N_ANSWER_JOBS="${N_ANSWER_JOBS:-1}"
N_JUDGE_JOBS="${N_JUDGE_JOBS:-2}"
BACKUP_TAG="${BACKUP_TAG:-pre-dense}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"
RESULTS_DIR="${ARANDU_RESULTS_DIR:-$PROJECT_DIR/results}"
RUN_DIR="$RESULTS_DIR/$PIPELINE_ID"

if [ ! -f scripts/slurm/rag/rag_common.sh ]; then
    echo "ERROR: run from the repository root." >&2
    exit 1
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir $RUN_DIR not found." >&2
    exit 1
fi
mkdir -p logs

for arm in $DENSE_ARMS; do
    if [ -d "$RUN_DIR/retrieve/outputs/$arm" ]; then
        echo "WARNING: $RUN_DIR/retrieve/outputs/$arm already exists; retrieve resumes it." >&2
    fi
done

for f in retrieve/retrieve_checkpoint.json retrieve/run_metadata.json \
         answers/answer_checkpoint.json answers/run_metadata.json \
         judge_answers/judge_answers_checkpoint.json judge_answers/run_metadata.json; do
    src="$RUN_DIR/$f"
    dst="$src.bak-$BACKUP_TAG"
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found; is $PIPELINE_ID a finished run?" >&2
        exit 1
    fi
    if [ -f "$dst" ]; then
        echo "backup kept:    $dst"
    else
        cp -p "$src" "$dst"
        echo "backup written: $dst"
    fi
done

export PIPELINE_ID DENSE_ARMS

submit() {
    if [ "${DRY_RUN:-}" = "1" ]; then
        echo "sbatch --parsable --kill-on-invalid-dep=yes $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable --kill-on-invalid-dep=yes "$@"
    fi
}

RET=$(submit scripts/slurm/rag/retrieve-dense.slurm)
echo "retrieve-dense: $RET"
ANS=$(submit --dependency=afterok:"$RET" scripts/slurm/rag/answer.slurm)
echo "answer:         $ANS"
for _ in $(seq 2 "$N_ANSWER_JOBS"); do
    ANS=$(submit --dependency=afterok:"$RET",afterany:"$ANS" scripts/slurm/rag/answer.slurm)
    echo "answer:         $ANS (resume)"
done
# Every judge job needs the last answer job to have succeeded. In
# --dependency, a comma joins conditions that must all hold.
PREV=$(submit --dependency=afterok:"$ANS" scripts/slurm/rag/judge-answers.slurm)
echo "judge-answers:  $PREV"
for _ in $(seq 2 "$N_JUDGE_JOBS"); do
    PREV=$(submit --dependency=afterok:"$ANS",afterany:"$PREV" scripts/slurm/rag/judge-answers.slurm)
    echo "judge-answers:  $PREV (resume)"
done
