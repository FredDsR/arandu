#!/bin/bash
#SBATCH --job-name=judge-qa
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/judge_qa_tupi_%j.out
#SBATCH --error=logs/judge_qa_tupi_%j.err

# =============================================================================
# Arandu CEP QA Judge — TUPI Partition
#
# Runs `arandu judge-qa` against
#     results/$PIPELINE_ID/cep/outputs
# using qwen3:14b on the GPU-accelerated Ollama sidecar.
#
# Usage:
#     PIPELINE_ID=test-cep-01 sbatch scripts/slurm/judge/qa/tupi.slurm
# =============================================================================

# Declare this wrapper's intent
export JUDGE_SUBCOMMAND=judge-qa

# Partition-specific settings
export ARANDU_JUDGE_VALIDATOR_MODEL="${ARANDU_JUDGE_VALIDATOR_MODEL:-qwen3:14b}"
export USE_GPU_OLLAMA=true

source "${SLURM_SUBMIT_DIR}/scripts/slurm/judge/judge_common.sh"
