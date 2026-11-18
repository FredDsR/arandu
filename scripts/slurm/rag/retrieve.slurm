#!/bin/bash
#SBATCH --job-name=arandu-rag-retrieve
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/rag_retrieve_%j.out
#SBATCH --error=logs/rag_retrieve_%j.err
# Send SIGTERM to the batch shell 60s before the time limit so rag_common.sh's
# teardown trap can stop the docker containers before SLURM's SIGKILL (prevents
# orphaned ollama-gpu / arandu-rag-run containers on the node; see rag_common).
#SBATCH --signal=B:TERM@60

# Phase C: run the default arm set (bm25, atlas_rag, khop_passage, khop_triple,
# null, oracle, dense_mpnet128, dense_bgem3_256). Arms already in the retrieve
# checkpoint are skipped, so resubmitting over a finished run retrieves only the
# missing arms. atlas_rag needs the LLM (NER + edge-filter) via ollama and the
# local embedder for query encoding; the dense arms load their encoder
# in-process; bm25/khop/null/oracle need neither. RETRIEVE_INCLUDE_REJECTED=1
# keeps the CEP pairs judge-qa rejected, as the arms of thesis-run-02 were run.
INCLUDE_REJECTED_ARG=""
if [ "${RETRIEVE_INCLUDE_REJECTED:-0}" = "1" ]; then
    INCLUDE_REJECTED_ARG=" --include-rejected"
fi
export RAG_CLI_ARGS="retrieve --id ${PIPELINE_ID}${INCLUDE_REJECTED_ARG}"
export RAG_NEEDS_OLLAMA=true
# Sequential client; atlas_rag NER/edge-filter calls are short: one slot
# at a modest context. Revisit with rag-batch-concurrency.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
