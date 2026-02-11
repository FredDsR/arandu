#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/qa_sirius_%x_%j.out
#SBATCH --error=logs/qa_sirius_%x_%j.err

# =============================================================================
# G-Transcriber QA Generation SLURM Job Script - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB)
# Note: Ollama does not support ROCm well, running in CPU mode
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-3}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-qwen3:14b}"
export USE_GPU_OLLAMA=false  # CPU mode - Ollama doesn't support ROCm well

# Source the common QA script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/qa/qa_common.sh"
