#!/bin/bash
#SBATCH --job-name=gtranscriber-kg
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/kg_%x_%j.out
#SBATCH --error=logs/kg_%x_%j.err

# =============================================================================
# G-Transcriber KG Construction SLURM Job Script - GRACE Partition
#
# Partition: grace
# GPU: NVIDIA L40S (46 GB)
# Recommended: GPU-accelerated Ollama with larger models
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_KG_MODEL_ID="${GTRANSCRIBER_KG_MODEL_ID:-llama3.1:8b}"
export USE_GPU_OLLAMA=true

# Source the common KG script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/kg/kg_common.sh"
