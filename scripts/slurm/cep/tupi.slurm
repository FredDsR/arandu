#!/bin/bash
#SBATCH --job-name=gtranscriber-cep
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/cep_tupi_%x_%j.out
#SBATCH --error=logs/cep_tupi_%x_%j.err

# =============================================================================
# G-Transcriber CEP QA Generation SLURM Job Script - TUPI Partition
#
# Partition: tupi
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
# Recommended: GPU-accelerated Ollama with 8B models
# =============================================================================

# Partition-specific settings
export GTRANSCRIBER_QA_WORKERS="${GTRANSCRIBER_QA_WORKERS:-4}"
export GTRANSCRIBER_QA_MODEL_ID="${GTRANSCRIBER_QA_MODEL_ID:-llama3.1:8b}"
export USE_GPU_OLLAMA=true
export GTRANSCRIBER_CEP_DIR="${GTRANSCRIBER_CEP_DIR:-${HOME}/etno-kgc-preprocessing/cep_dataset_tupi}"
export GTRANSCRIBER_RESULTS_DIR="${GTRANSCRIBER_RESULTS_DIR:-${HOME}/etno-kgc-preprocessing/results_tupi}"

# Source the common CEP script
source "${SLURM_SUBMIT_DIR}/scripts/slurm/cep/cep_common.sh"
