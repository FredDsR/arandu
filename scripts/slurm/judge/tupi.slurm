#!/bin/bash
#SBATCH --job-name=arandu-judge
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/judge_tupi_%x_%j.out
#SBATCH --error=logs/judge_tupi_%x_%j.err

# =============================================================================
# Arandu Transcription-Judge SLURM Job Script - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Validator: qwen3:14b on GPU-accelerated Ollama
#
# Usage:
#     PIPELINE_ID=test-cep-01 sbatch scripts/slurm/judge/tupi.slurm
# =============================================================================

# Partition-specific settings
export ARANDU_JUDGE_VALIDATOR_MODEL="${ARANDU_JUDGE_VALIDATOR_MODEL:-qwen3:14b}"
export USE_GPU_OLLAMA=true

# Source the common judge script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/judge/judge_common.sh"
