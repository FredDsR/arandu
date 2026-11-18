#!/bin/bash
#SBATCH --job-name=arandu-rag-retrieve-dense
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/rag_retrieve_dense_%j.out
#SBATCH --error=logs/rag_retrieve_dense_%j.err
# Send SIGTERM to the batch shell 60s before the time limit so rag_common.sh's
# teardown trap can stop the docker containers before SLURM's SIGKILL.
#SBATCH --signal=B:TERM@60

# Phase C, incremental: retrieve only the dense arms named in DENSE_ARMS
# (space-separated, e.g. "dense_mpnet128 dense_bgem3_256") over every CEP pair
# and probe, rejected pairs included, as the canonical arms were. The encoder
# runs in-process on the GPU; no LLM, so no ollama sidecar. Each arm writes only
# retrieve/outputs/<arm>/ and its own index; existing checkpoint keys are kept.
DENSE_ARMS="${DENSE_ARMS:?DENSE_ARMS must list the dense arms to retrieve}"
ARM_ARGS=""
for arm in $DENSE_ARMS; do
    ARM_ARGS="$ARM_ARGS --arm $arm"
done
export RAG_CLI_ARGS="retrieve --id ${PIPELINE_ID}${ARM_ARGS} --include-rejected"
export RAG_NEEDS_OLLAMA=false
source "${SLURM_SUBMIT_DIR}/scripts/slurm/rag/rag_common.sh"
