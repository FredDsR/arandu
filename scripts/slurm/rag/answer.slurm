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
# The answer batch client is SEQUENTIAL (one request in flight), so one
# server slot maximizes per-token speed: no idle KV slots, no CPU spill
# (3x32k slots = ~25GB > 24GB VRAM, measured 2026-06-11). Context sized to
# the answerer packer ceiling (max_context_tokens=16384) plus margin.
# Revisit when client-side concurrency lands (rag-batch-concurrency task).
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-24576}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
