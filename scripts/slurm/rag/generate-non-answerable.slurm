#!/bin/bash
#SBATCH --job-name=arandu-rag-nonans
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/rag_nonans_%j.out
#SBATCH --error=logs/rag_nonans_%j.err
# Send SIGTERM to the batch shell 60s before the time limit so rag_common.sh's
# teardown trap can stop the docker containers before SLURM's SIGKILL (prevents
# orphaned ollama-gpu / arandu-rag-run containers on the node; see rag_common).
#SBATCH --signal=B:TERM@60

# Phase C: entity-perturbed non-answerable items (LLM via ollama qwen3:14b).
# Set NONANSWERABLE_REGENERATE=1 to discard the checkpoint + prior items and
# re-perturb every seed (failed seeds are otherwise skipped on resume).
export RAG_CLI_ARGS="generate-non-answerable --id ${PIPELINE_ID}${NONANSWERABLE_REGENERATE:+ --regenerate}"
export RAG_NEEDS_OLLAMA=true
# Concurrent client (ARANDU_NONANSWERABLE_WORKERS) with matching server
# slots; short entity-swap prompts keep the per-slot context modest
# (3 x 16384 KV ~ 7.5GB + 9.3GB weights, well inside the 24GB card).
export ARANDU_NONANSWERABLE_WORKERS="${ARANDU_NONANSWERABLE_WORKERS:-3}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$ARANDU_NONANSWERABLE_WORKERS}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
