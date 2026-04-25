#!/bin/bash
#SBATCH --job-name=arandu-judge-qa
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge_qa_tupi_%j.out
#SBATCH --error=logs/judge_qa_tupi_%j.err

# =============================================================================
# Arandu CEP QA Judge — TUPI Partition
#
# Runs `arandu judge-qa` against
#     results/$PIPELINE_ID/cep/outputs
# using qwen3:14b on the GPU-accelerated Ollama sidecar.
#
# By default the run resumes — pairs that already carry a validation
# payload are skipped. To force a fresh pass over every sampled pair,
# set JUDGE_REJUDGE=1 at submit time.
#
# Usage:
#     # Default: resume from where the previous run left off
#     PIPELINE_ID=test-cep-01 sbatch scripts/slurm/judge/qa/tupi.slurm
#
#     # Force re-evaluation of every sampled pair
#     PIPELINE_ID=test-cep-01 JUDGE_REJUDGE=1 \
#         sbatch scripts/slurm/judge/qa/tupi.slurm
# =============================================================================

# Declare this wrapper's intent
export JUDGE_SUBCOMMAND=judge-qa

# Partition-specific settings (validator model default lives in judge_common.sh)
export USE_GPU_OLLAMA=true

source "${SLURM_SUBMIT_DIR}/scripts/slurm/judge/judge_common.sh"
