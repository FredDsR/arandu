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
# Client-side concurrency (ARANDU_JUDGE_ANSWERS_WORKERS) must match the
# server slots, and slots x context KV must fit the 24GB card (KV ~0.15GB
# per 1k ctx per slot, weights ~9.3GB). Context must clear the measured
# prompt ceiling: khop_passage/atlas_rag coverage+faithfulness prompts
# reach ~25k tokens + 8192 output budget (32k overflows). 2 x 40960 ~
# 12.5GB KV, ~22.8GB total - the max concurrent config that still fits.
export ARANDU_JUDGE_ANSWERS_WORKERS="${ARANDU_JUDGE_ANSWERS_WORKERS:-2}"
# Server slots follow the client worker count unless explicitly overridden,
# so the two can never drift apart on a plain submit.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$ARANDU_JUDGE_ANSWERS_WORKERS}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-40960}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
