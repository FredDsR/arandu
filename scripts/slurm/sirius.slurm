#!/bin/bash
#SBATCH --job-name=corpus_transcription
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/sirius_%x_%j.out
#SBATCH --error=logs/sirius_%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB) with ROCm support
# Recommended: 3 workers (no quantization available on AMD)
# =============================================================================

# Use ROCm for AMD GPU acceleration
export USE_ROCM=false
export USE_SCRATCH=false
export WORKERS="${WORKERS:-3}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"
export GTRANSCRIBER_RESULTS_DIR="${HOME}/etno-kgc-preprocessing/results_sirius"


# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"
