#!/bin/bash
#SBATCH --job-name=cep_qa_dataset_gen
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Note: GPU not requested - Ollama runs in CPU mode on this partition (ROCm not well supported)
#SBATCH --output=logs/cep_sirius_%x_%j.out
#SBATCH --error=logs/cep_sirius_%x_%j.err

# =============================================================================
# Arandu CEP QA Generation SLURM Job Script - SIRIUS Partition
#
# Partition: sirius
# GPU: AMD Radeon RX 7900 XT/XTX (21 GB)
# Note: Ollama does not support ROCm well, running in CPU mode
# =============================================================================

# Partition-specific settings
export ARANDU_QA_WORKERS="${ARANDU_QA_WORKERS:-3}"
export ARANDU_QA_MODEL_ID="${ARANDU_QA_MODEL_ID:-qwen3:14b}"
export USE_GPU_OLLAMA=false  # CPU mode - Ollama doesn't support ROCm well

# Source the common CEP script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/cep/cep_common.sh"
