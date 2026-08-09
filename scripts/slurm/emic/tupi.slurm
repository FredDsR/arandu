#!/bin/bash
#SBATCH --job-name=arandu-emic-judge
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/emic_judge_tupi_%j.out
#SBATCH --error=logs/emic_judge_tupi_%j.err
# Send SIGTERM to the batch shell 60s before the time limit so the teardown
# trap can stop the docker containers before SLURM's SIGKILL (prevents orphaned
# ollama-gpu / arandu-emic-run containers; see scripts/slurm/container_teardown.sh).
#SBATCH --signal=B:TERM@60

# =============================================================================
# Arandu Emic Judge — TUPI Partition
#
# Scores the CEP pairs of a run with the ordinal `emic_validity` criterion
# (qwen3:14b on the GPU-accelerated Ollama sidecar) and writes per-source
# scores to results/$PIPELINE_ID/emic_judge/outputs/.
#
# These scores are the study's MEASUREMENT of emic validity; the human
# annotation round (spec §6) reports agreement with them. The model is
# therefore a methodological parameter, not a budget knob.
#
# Usage:
#     # Default: scope=all, resume from the checkpoint
#     PIPELINE_ID=thesis-run-01 sbatch scripts/slurm/emic/tupi.slurm
#
#     # Only canonically-approved pairs (cheaper, no cross-tabulation)
#     PIPELINE_ID=thesis-run-01 EMIC_SCOPE=approved \
#         sbatch scripts/slurm/emic/tupi.slurm
#
#     # Discard the checkpoint and re-score every source
#     PIPELINE_ID=thesis-run-01 EMIC_RERUN=1 \
#         sbatch scripts/slurm/emic/tupi.slurm
# =============================================================================

export USE_GPU_OLLAMA=true

# Client-side concurrency must match the server slots, and slots x context KV
# must fit the 24 GB card (KV ~0.15 GB per 1k ctx per slot, weights ~9.3 GB).
#
# Context sizing. The emic prompt is template + segment + question + answer:
# the template is ~7.5 KB (~2k tokens pt), the segment is one `cep_4k` chunk
# (RecursiveChunker chunk_size=4000, so <=4k tokens), Q+A add a few hundred.
# Input ceiling ~6.5k tokens; the response budget is max_tokens=8192
# (REASONING_MODEL_MAX_TOKENS — qwen3 burns thinking tokens against it). So
# ~15k is the working ceiling and 16384 leaves headroom.
#
# 4 x 16384 ~ 9.8 GB KV + ~9.3 GB weights ~ 19 GB, inside the envelope
# judge-answers already runs at (2 x 40960 ~ 22.8 GB). Emic gets twice the
# concurrency because its prompts are ~4x shorter than the RAG judge's, which
# carry retrieved passages.
export ARANDU_EMIC_JUDGE_WORKERS="${ARANDU_EMIC_JUDGE_WORKERS:-4}"
# Server slots follow the client worker count unless explicitly overridden, so
# the two can never drift apart on a plain submit.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-$ARANDU_EMIC_JUDGE_WORKERS}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"

source "${SLURM_SUBMIT_DIR}/scripts/slurm/emic/emic_common.sh"
