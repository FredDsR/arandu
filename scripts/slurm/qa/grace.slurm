#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/qa_%x_%j.out
#SBATCH --error=logs/qa_%x_%j.err

# =============================================================================
# G-Transcriber QA Generation SLURM Job Script - GRACE Partition
#
# Partition: grace
# GPU: NVIDIA L40S (46 GB)
# Recommended: GPU-accelerated Ollama with larger models
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-6}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-qwen3:14b}"
export USE_GPU_OLLAMA=true

# Source the common QA script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/qa/qa_common.sh"
