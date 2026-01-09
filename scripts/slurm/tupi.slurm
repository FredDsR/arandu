#!/bin/bash
#SBATCH --job-name=gtranscriber
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: 4 workers with quantization
# =============================================================================

# Use the base script with partition-specific settings
export WORKERS="${WORKERS:-4}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"
