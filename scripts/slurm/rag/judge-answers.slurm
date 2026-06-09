#!/bin/bash
#SBATCH --job-name=arandu-rag-judge-ans
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rag_judge_ans_%j.out
#SBATCH --error=logs/rag_judge_ans_%j.err

# Phase C: gated 5-criteria answer judge (LLM via ollama qwen3:14b). The heaviest
# stage by call count (~58% of the LLM fan-out). Resumes by default; pass
# JUDGE_REJUDGE=1 semantics via --rejudge if a clean re-judge is needed.
export RAG_CLI_ARGS="judge-answers --id ${PIPELINE_ID}${JUDGE_ANSWERS_REJUDGE:+ --rejudge}"
export RAG_NEEDS_OLLAMA=true
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
