#!/bin/bash
#SBATCH --job-name=arandu-rag-answer
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rag_answer_%j.out
#SBATCH --error=logs/rag_answer_%j.err

# Phase C: answer every RetrievalRecord (LLM via ollama qwen3:14b), held constant
# across arms. Reasoning-model headroom (max_tokens=8192) avoids the truncation
# -> spurious-abstention failure mode (PR #126/#127).
export RAG_CLI_ARGS="answer --id ${PIPELINE_ID}"
export RAG_NEEDS_OLLAMA=true
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
