#!/bin/bash
#SBATCH --job-name=arandu-rag-kg-link
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/rag_kg_link_%j.out
#SBATCH --error=logs/rag_kg_link_%j.err

# Phase C: map atlas-rag passages -> char offsets in transcription space (no LLM).
# CPU-only: no GPU request, runs via arandu-rag-cpu.
export RAG_CLI_ARGS="kg-link-passages --id ${PIPELINE_ID}"
export RAG_NEEDS_OLLAMA=false
export RAG_SERVICE=arandu-rag-cpu
export RAG_PROFILE=rag-cpu
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
