#!/bin/bash
#SBATCH --job-name=arandu-rag-nonans
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/rag_nonans_%j.out
#SBATCH --error=logs/rag_nonans_%j.err

# Phase C: entity-perturbed non-answerable items (LLM via ollama qwen3:14b).
export RAG_CLI_ARGS="generate-non-answerable --id ${PIPELINE_ID}"
export RAG_NEEDS_OLLAMA=true
# Sequential client (one request in flight) + short entity-swap prompts:
# one slot at a modest context. Revisit with rag-batch-concurrency.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
