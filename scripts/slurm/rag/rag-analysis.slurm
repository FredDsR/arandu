#!/bin/bash
#SBATCH --job-name=arandu-rag-analysis
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/rag_analysis_%j.out
#SBATCH --error=logs/rag_analysis_%j.err

# Phase C: aggregate judged answers into report.json + tables.md (no LLM).
# CPU-only: no GPU request, runs via arandu-rag-cpu.
export RAG_CLI_ARGS="rag-analysis --id ${PIPELINE_ID}"
export RAG_NEEDS_OLLAMA=false
export RAG_SERVICE=arandu-rag-cpu
export RAG_PROFILE=rag-cpu
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
