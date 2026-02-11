#!/bin/bash
#SBATCH --job-name=corpus_transcription
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tupi_%x_%j.out
#SBATCH --error=logs/tupi_%x_%j.err

# =============================================================================
# G-Transcriber SLURM Job Script for PCAD - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: 4 workers without quantization (24 GB VRAM is sufficient)
# =============================================================================

# Use the base script with partition-specific settings

export WORKERS="${WORKERS:-4}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"
export GTRANSCRIBER_QUANTIZE=false  # RTX 4090 has enough VRAM, no need for quantization

# Source the main job script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/transcription/job_common.sh"
