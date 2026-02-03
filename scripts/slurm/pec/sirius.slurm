#!/bin/bash
#SBATCH --job-name=gtranscriber-pec
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/pec_sirius_%x_%j.out
#SBATCH --error=logs/pec_sirius_%x_%j.err

# =============================================================================
# G-Transcriber PEC QA Generation SLURM Job Script - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB)
# Note: Ollama does not support ROCm well, running in CPU mode
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-3}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-llama3.1:8b}"
export USE_GPU_OLLAMA=false  # CPU mode - Ollama doesn't support ROCm well
export GTRANSCRIBER_PEC_DIR="${GTRANSCRIBER_PEC_DIR:-${HOME}/etno-kgc-preprocessing/pec_dataset_sirius}"

# Source the common PEC script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/pec/pec_common.sh"
