#!/bin/bash
#SBATCH --job-name=arandu-rag-chunk
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/rag_chunk_%j.out
#SBATCH --error=logs/rag_chunk_%j.err

# Phase C: chunk transcriptions into the cep_4k view (no LLM). Prereq for cep + bm25.
export RAG_CLI_ARGS="chunk /app/results/${PIPELINE_ID}/transcription/outputs --id ${PIPELINE_ID} --view cep_4k"
export RAG_NEEDS_OLLAMA=false
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
