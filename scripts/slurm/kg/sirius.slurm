#!/bin/bash
#SBATCH --job-name=gtranscriber-kg
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/kg_sirius_%x_%j.out
#SBATCH --error=logs/kg_sirius_%x_%j.err

# =============================================================================
# G-Transcriber KG Construction SLURM Job Script - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB)
# Note: Ollama does not support ROCm well, running in CPU mode
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_KG_MODEL_ID="${GTRANSCRIBER_KG_MODEL_ID:-llama3.1:8b}"
export USE_GPU_OLLAMA=false  # CPU mode - Ollama doesn't support ROCm well

# Source the common KG script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/kg/kg_common.sh"
