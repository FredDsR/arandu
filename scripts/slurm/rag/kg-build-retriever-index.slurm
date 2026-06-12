#!/bin/bash
#SBATCH --job-name=arandu-rag-kg-index
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/rag_kg_index_%j.out
#SBATCH --error=logs/rag_kg_index_%j.err

# Phase C: build the atlas-rag precompute (node/edge/text embeddings + faiss).
# Uses the local sentence-transformers embedder (ARANDU_EMBEDDER_*), so --rebuild
# is required: any prior Gemini-embedding precompute is incompatible.
export RAG_CLI_ARGS="kg-build-retriever-index --id ${PIPELINE_ID} --rebuild"
export RAG_NEEDS_OLLAMA=false
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
