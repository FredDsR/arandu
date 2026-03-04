#!/bin/bash
#SBATCH --job-name=arandu-qa
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/qa_tupi_%x_%j.out
#SBATCH --error=logs/qa_tupi_%x_%j.err

# =============================================================================
# Arandu QA Generation SLURM Job Script - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: GPU-accelerated Ollama with 14B models
# =============================================================================

# Partition-specific settings
export ARANDU_QA_WORKERS="${ARANDU_QA_WORKERS:-4}"
export ARANDU_QA_MODEL_ID="${ARANDU_QA_MODEL_ID:-qwen3:14b}"
export USE_GPU_OLLAMA=true

# Source the common QA script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/qa/qa_common.sh"
