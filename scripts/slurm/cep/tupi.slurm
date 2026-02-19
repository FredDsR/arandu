#!/bin/bash
#SBATCH --job-name=cep_qa_dataset_gen
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cep_tupi_%x_%j.out
#SBATCH --error=logs/cep_tupi_%x_%j.err

# =============================================================================
# G-Transcriber CEP QA Generation SLURM Job Script - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: GPU-accelerated Ollama with 14B models
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-4}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-qwen3:14b}"
export USE_GPU_OLLAMA=true

# Source the common CEP script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/cep/cep_common.sh"
