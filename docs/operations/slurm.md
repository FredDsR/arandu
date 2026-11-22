# Generic SLURM Deployment Guide

This guide describes how to run Arandu stages on a SLURM-based HPC cluster
using Docker/Podman or Apptainer/Singularity.

> **Cluster-specific scripts:** concrete partition scripts and submission
> wrappers used for our HPC runs are maintained separately in
> [`arandu-deploy`](https://github.com/FredDsR/arandu-deploy).

## Prerequisites

- SLURM job scheduler
- Container runtime (Docker, rootless Podman, or Apptainer/Singularity)
- GPU nodes (NVIDIA CUDA or AMD ROCm) for Whisper transcription, local Ollama
  inference, and dense embedding indexing

## Container Images

Arandu ships two main Dockerfiles at the repository root:

- `Dockerfile` (`arandu:latest`): transcription, chunking, CEP QA generation,
  judges, and lightweight retrieval/answering stages.
- `Dockerfile.kg` (`arandu-kg:latest`): knowledge graph construction
  (`build-kg`) and embedding/retrieval stages that require the `kg` extra
  (`atlas-rag`, `faiss-cpu`, `sentence-transformers`).
- `Dockerfile.rocm`: AMD ROCm variant for Whisper transcription.

See [Docker Deployment](docker.md) for Compose profiles and volume mounts.

## Minimal SLURM Template

Every stage reads its configuration from `ARANDU_*` environment variables and
checkpoints under `results/<PIPELINE_ID>/<stage>/outputs/`.

### With Docker Compose / Podman Compose

```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/arandu_%j.out
#SBATCH --error=logs/arandu_%j.err

export PIPELINE_ID="${PIPELINE_ID:-run-01}"
export ARANDU_QA_WORKERS=3
export ARANDU_QA_MODEL_ID="qwen3:14b"
export ARANDU_JUDGE_VALIDATOR_MODEL="qwen3:14b"

docker compose --profile cep-gpu up --abort-on-container-exit
```

### With Apptainer / Singularity

```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Build once from a locally built image
apptainer build arandu.sif docker-daemon://arandu:latest

# Execute a stage against an external or node-local OpenAI-compatible/Ollama server
apptainer exec --nv \
  --bind "$PWD/results:/app/results" \
  arandu.sif \
  arandu generate-cep-qa /app/results --id "${PIPELINE_ID:-run-01}"
```

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `PIPELINE_ID` | Groups all stages of a run under `results/<id>/` |
| `ARANDU_QA_WORKERS` | Concurrent workers for CEP QA generation |
| `ARANDU_QA_MODEL_ID` / `ARANDU_KG_MODEL_ID` | Generation / KG extraction model ID |
| `ARANDU_JUDGE_VALIDATOR_MODEL` | Validator model for `judge-qa`, `judge-transcription`, and the CEP scaffolding gate |
| `ARANDU_CEP_ENABLE_SCAFFOLDING_GATE` | `true` (default) judges each pair during generation; `false` defers judging to `judge-qa` |
| `ARANDU_CEP_LANGUAGE` | Prompt language (`pt` or `en`) |
| `OLLAMA_NUM_PARALLEL` / `OLLAMA_CONTEXT_LENGTH` | Ollama sidecar concurrency slots and context window |

See [Configuration](../guide/configuration.md) for the complete list of
stage-specific variables.

## Checkpoint and Resume

All pipeline stages checkpoint their progress under `results/<id>/<stage>/`. If
a SLURM job hits its wall-time limit (`TIMEOUT`) or is preempted, resubmitting
the same job with the same `PIPELINE_ID` automatically resumes from the last
completed item.

## Resource Guidelines

| Stage | Typical CPUs | Memory | GPU |
|-------|--------------|--------|-----|
| Transcription (Whisper) | 4–8 | 16 GB | 1× GPU recommended |
| CEP QA generation + gate | 8 | 16–32 GB | 1× GPU (for local Ollama) |
| KG construction (`build-kg`) | 8–16 | 32–64 GB | 1× GPU (for local Ollama) |
| Retrieval / Answering / Judging | 4–8 | 16–32 GB | 1× GPU for dense indexing / local LLM |

---

**See also**: [Docker Deployment](docker.md) | [Configuration](../guide/configuration.md) | [`arandu-deploy`](https://github.com/FredDsR/arandu-deploy)
