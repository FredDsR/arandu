#!/bin/bash
#SBATCH --job-name=arandu-rag-retrieve-khop
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/rag_retrieve_khop_%j.out
#SBATCH --error=logs/rag_retrieve_khop_%j.err

# One-off khop re-validation: re-run ONLY the khop arms, CPU-only. khop is pure
# lexical + graph (no LLM, no embedder), so it needs neither Ollama nor a GPU —
# runs on a CPU slot to skip GPU contention. The standard answer/judge-answers
# wrappers handle khop automatically via resume (their checkpoints were stripped
# of khop keys). No --gres=gpu:1 here.
export RAG_CLI_ARGS="retrieve --id ${PIPELINE_ID} --arm khop_passage --arm khop_triple"
export RAG_NEEDS_OLLAMA=false
export RAG_SERVICE=arandu-rag-cpu
export RAG_PROFILE=rag-cpu
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
