#!/bin/bash
#SBATCH --job-name=gtranscriber
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB) with ROCm support
# Recommended: 3 workers (no quantization available on AMD)
# =============================================================================

# Use ROCm for AMD GPU acceleration
export USE_ROCM=true
export WORKERS="${WORKERS:-3}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"

# Source the main job script
source "$(dirname "$0")/job_common.sh"
