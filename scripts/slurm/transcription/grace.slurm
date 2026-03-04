#!/bin/bash
#SBATCH --job-name=arandu
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Arandu SLURM Job Script for PCAD - GRACE Partition
#
# Partition: grace
# GPU: NVIDIA L40S (46 GB)
# Recommended: 6-8 workers with quantization (large VRAM)
# =============================================================================

# Use the base script with partition-specific settings
export WORKERS="${WORKERS:-6}"
export ARANDU_MODEL_ID="${ARANDU_MODEL_ID:-openai/whisper-large-v3}"

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/transcription/job_common.sh"
