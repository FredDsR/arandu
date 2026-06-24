#!/bin/bash
#SBATCH --job-name=arandu-rag-chunk
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/rag_chunk_%j.out
#SBATCH --error=logs/rag_chunk_%j.err

# Phase C: chunk transcriptions into the cep_4k view (no LLM). Prereq for cep + bm25.
# CPU-only: no GPU request, runs via arandu-rag-cpu so it doesn't wait for a GPU.
# Set CHUNK_REBUILD=1 to clear stale view outputs + checkpoint first (e.g. to
# drop chunks of transcriptions the transcription judge rejected after a prior
# run). Only the exact value "1" enables it (so CHUNK_REBUILD=0 is a real off).
REBUILD_FLAG=""
if [ "${CHUNK_REBUILD:-}" = "1" ]; then
    REBUILD_FLAG=" --rebuild"
fi
export RAG_CLI_ARGS="chunk /app/results/${PIPELINE_ID}/transcription/outputs --id ${PIPELINE_ID} --view cep_4k${REBUILD_FLAG}"
export RAG_NEEDS_OLLAMA=false
export RAG_SERVICE=arandu-rag-cpu
export RAG_PROFILE=rag-cpu
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
