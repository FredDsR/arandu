#!/bin/bash
#SBATCH --job-name=arandu-kg
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/kg_tupi_%x_%j.out
#SBATCH --error=logs/kg_tupi_%x_%j.err

# =============================================================================
# Arandu KG Construction SLURM Job Script - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: GPU-accelerated Ollama with 8B models
# =============================================================================

# Partition-specific settings
export ARANDU_KG_MODEL_ID="${ARANDU_KG_MODEL_ID:-llama3.1:8b}"
export USE_GPU_OLLAMA=true

# Source the common KG script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/kg/kg_common.sh"
