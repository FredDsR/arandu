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
# The judge batch client is SEQUENTIAL (one request in flight): one server
# slot, sized for the measured prompt ceiling - khop_passage/atlas_rag
# coverage+faithfulness prompts reach ~25k tokens (passages are document
# scale, 2026-06-11 measurement) + 8192 output budget, so 32k overflows.
# 49152 = ceiling + tokenizer-estimate margin; KV ~7.5GB, total ~18GB.
# Revisit when client-side concurrency lands (rag-batch-concurrency task).
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-49152}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
