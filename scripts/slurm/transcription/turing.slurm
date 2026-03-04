#!/bin/bash
#SBATCH --job-name=corpus_transcription
#SBATCH --partition=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/turing_%x_%j.out
#SBATCH --error=logs/turing_%x_%j.err

# =============================================================================
# Arandu SLURM Job Script for PCAD - TURING Partition
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
export ARANDU_MODEL_ID="${ARANDU_MODEL_ID:-distil-whisper/distil-large-v3}"

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/transcription/job_common.sh"
