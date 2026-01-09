#!/bin/bash
#SBATCH --job-name=gtranscriber
#SBATCH --partition=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - TURING Partition
#
# Partition: turing
# Accelerator: 4x NEC TSUBASA Vector Engine (48 GB each)
# Note: Vector Engine is not compatible with CUDA/PyTorch
# Recommended: CPU mode only
# =============================================================================

# NEC Vector Engines are not compatible with PyTorch/Transformers
# Force CPU mode
export USE_CPU=true
export WORKERS="${WORKERS:-8}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-distil-whisper/distil-large-v3}"

# Source the main job script
source "$(dirname "$0")/job_common.sh"
