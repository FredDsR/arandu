#!/bin/bash
#SBATCH --job-name=gtranscriber
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - BLAISE Partition
#
# Partition: blaise
# GPU: 4x Tesla P100 (16 GB each)
# Recommended: 2-3 workers with quantization
# =============================================================================

# Use the base script with partition-specific settings
export WORKERS="${WORKERS:-3}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/transcription/job_common.sh"
