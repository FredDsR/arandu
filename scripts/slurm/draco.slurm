#!/bin/bash
#SBATCH --job-name=gtranscriber
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - DRACO Partition
#
# Partition: draco
# GPU: Tesla K20m (4 GB) - limited VRAM, consider CPU mode
# Recommended: 1-2 workers with quantization, or CPU mode
# =============================================================================

# Use the base script with partition-specific settings
export WORKERS="${WORKERS:-2}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-small}"
export USE_CPU="${USE_CPU:-true}"

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"
