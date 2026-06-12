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
# Client-side concurrency (ARANDU_ANSWERER_WORKERS) must match the server
# slots, and slots x context KV must fit the 24GB card (KV ~0.15GB per 1k
# ctx per slot, qwen3:14b weights ~9.3GB). 3 x 24576 ~ 11.3GB KV, ~21.6GB
# total. Context sized to the answerer packer ceiling (16384) plus margin.
export ARANDU_ANSWERER_WORKERS="${ARANDU_ANSWERER_WORKERS:-3}"
# Server slots follow the client worker count unless explicitly overridden,
# so the two can never drift apart on a plain submit.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$ARANDU_ANSWERER_WORKERS}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-24576}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
