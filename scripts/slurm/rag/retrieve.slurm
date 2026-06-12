#!/bin/bash
#SBATCH --job-name=arandu-rag-retrieve
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/rag_retrieve_%j.out
#SBATCH --error=logs/rag_retrieve_%j.err

# Phase C: run all 5 arms. atlas_rag needs the LLM (NER + edge-filter) via ollama
# and the local embedder for query encoding; bm25/khop/null need neither.
export RAG_CLI_ARGS="retrieve --id ${PIPELINE_ID}"
export RAG_NEEDS_OLLAMA=true
# Sequential client; atlas_rag NER/edge-filter calls are short: one slot
# at a modest context. Revisit with rag-batch-concurrency.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
