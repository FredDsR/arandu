#!/bin/bash
#SBATCH --job-name=arandu-judge-transcription
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge_transcription_tupi_%j.out
#SBATCH --error=logs/judge_transcription_tupi_%j.err
#SBATCH --signal=B:TERM@60

# =============================================================================
# Arandu Transcription Judge — TUPI Partition
#
# Runs `arandu judge-transcription` against
#     results/$PIPELINE_ID/transcription/outputs
# using qwen3:14b on the GPU-accelerated Ollama sidecar.
#
# By default the run resumes — already-judged records are skipped. To force
# a fresh pass over everything, set JUDGE_REJUDGE=1 at submit time.
#
# Usage:
#     # Default: resume from where the previous run left off
#     PIPELINE_ID=test-cep-01 sbatch scripts/slurm/judge/transcription/tupi.slurm
#
#     # Force re-evaluation of every record (after model/prompt change)
#     PIPELINE_ID=test-cep-01 JUDGE_REJUDGE=1 \
#         sbatch scripts/slurm/judge/transcription/tupi.slurm
# =============================================================================

# Declare this wrapper's intent
export JUDGE_SUBCOMMAND=judge-transcription

# Partition-specific settings (validator model default lives in judge_common.sh)
export USE_GPU_OLLAMA=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../judge_common.sh"
