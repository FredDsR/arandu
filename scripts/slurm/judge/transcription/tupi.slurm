#!/bin/bash
#SBATCH --job-name=arandu-judge-transcription
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/judge_transcription_tupi_%j.out
#SBATCH --error=logs/judge_transcription_tupi_%j.err

# =============================================================================
# Arandu Transcription Judge — TUPI Partition
#
# Runs `arandu judge-transcription` against
#     results/$PIPELINE_ID/transcription/outputs
# using qwen3:14b on the GPU-accelerated Ollama sidecar.
#
# Usage:
#     PIPELINE_ID=test-cep-01 sbatch scripts/slurm/judge/transcription/tupi.slurm
# =============================================================================

# Declare this wrapper's intent
export JUDGE_SUBCOMMAND=judge-transcription

# Partition-specific settings (validator model default lives in judge_common.sh)
export USE_GPU_OLLAMA=true

source "${SLURM_SUBMIT_DIR}/scripts/slurm/judge/judge_common.sh"
