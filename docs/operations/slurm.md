# Generic SLURM Deployment Guide

This guide covers running Arandu on any SLURM-based HPC cluster.

## Prerequisites

- SLURM job scheduler
- Docker or Singularity/Apptainer (recommended for HPC)
- GPU nodes (NVIDIA CUDA or AMD ROCm)

## SLURM Scripts

Scripts live under `scripts/slurm/<step>/` as thin **per-partition** files
(`<partition>.slurm`) that source a shared `<step>_common.sh`. There is no
single `run_<x>.slurm` entry point; you submit the partition script for the
node type you want.

| Step | Scripts | Description |
|------|---------|-------------|
| Transcription | `transcription/<partition>.slurm` + `job_common.sh` | Batch Whisper transcription |
| CEP QA | `cep/{grace,tupi,sirius}.slurm` + `cep_common.sh` | Bloom-scaffolded QA generation (`generate-cep-qa`) |
| Judge | `judge/transcription/tupi.slurm`, `judge/qa/tupi.slurm` + `judge_common.sh` | LLM-as-a-Judge (`judge-transcription` / `judge-qa`) |
| KG | `kg/{grace,tupi,sirius}.slurm` + `kg_common.sh` | atlas-rag KG construction (`build-kg`) |
| RAG (Phase C) | `rag/<stage>.slurm` + `rag_common.sh` | chunk / retrieve / answer / judge-answers / rag-analysis / generate-non-answerable |
| Cleanup | `general/cleanup.slurm`, `kg/pipeline-cleanup.slurm` | Docker/disk cleanup |

For the authoritative cluster reference (partition access + GPU rules, the
code→config→compose→image→SLURM wiring, the rsync deploy flow), see
[`scripts/slurm/AGENTS.md`](../../scripts/slurm/AGENTS.md) and the
[PCAD Guide](pcad.md).

## Basic Usage

### Submit a Job

Every job takes a `PIPELINE_ID` (groups all of a run's stages under
`results/<id>/`). Submit the partition script for the node you want:

```bash
PIPELINE_ID=run-01 sbatch scripts/slurm/cep/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/kg/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/rag/retrieve.slurm
```

### Override Settings

Overrides are environment variables passed at submit time (the scripts read the
same `ARANDU_*` config the CLI does):

```bash
# Different model + worker count for a CEP run
PIPELINE_ID=run-01 ARANDU_QA_MODEL_ID=qwen3:14b ARANDU_QA_WORKERS=3 \
  sbatch scripts/slurm/cep/tupi.slurm

# Avoid a known-bad node instead of pinning one
PIPELINE_ID=run-01 sbatch --exclude=tupi5 scripts/slurm/kg/tupi.slurm
```

### Monitor Jobs

```bash
# View your jobs
squeue -u $USER

# View job details
scontrol show job <job_id>

# View logs
tail -f logs/arandu_<job_id>.out
```

### Cancel a Job

```bash
scancel <job_id>
```

## Script Structure

A partition script is thin: `#SBATCH` directives, then it sources the step's
`<step>_common.sh`, which builds the image, starts the Ollama sidecar (for LLM
stages), and runs the pipeline container via `docker compose`. For example
`scripts/slurm/cep/tupi.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=tupi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cep_tupi_%j.out
#SBATCH --error=logs/cep_tupi_%j.err

export USE_GPU_OLLAMA=true          # selects the cep-gpu compose profile
export ARANDU_QA_WORKERS=3
source "$(dirname "$0")/cep_common.sh"   # builds image, runs `arandu generate-cep-qa` via compose
```

The container runs `arandu <command>` off the image `ENTRYPOINT`; the
`<step>_common.sh` helper handles image build, the Ollama sidecar, model pull,
and cleanup. See [`scripts/slurm/AGENTS.md`](../../scripts/slurm/AGENTS.md) for
the full step→service→profile→image mapping.

## Environment Variables

Set these at submit time; the scripts forward them to the container, where the
matching `*Config` reads them (same contract as the CLI):

| Variable | Description |
|----------|-------------|
| `PIPELINE_ID` | **Required.** Groups a run's stages under `results/<id>/` |
| `USE_GPU_OLLAMA` | `true` selects the `-gpu` compose profile (set by GPU partition scripts) |
| `ARANDU_QA_WORKERS` | Parallel workers for CEP/QA generation |
| `ARANDU_QA_MODEL_ID` / `ARANDU_KG_MODEL_ID` | Generation / KG model override |
| `ARANDU_JUDGE_VALIDATOR_MODEL` | Validator model for `judge-qa` / `judge-transcription` |
| `JUDGE_REJUDGE` | `1` forces a fresh judge pass instead of resume |
| `ARANDU_CEP_LANGUAGE` | Prompt language (`pt` or `en`) |

## Using Docker on SLURM

The repo scripts already drive Docker Compose through `<step>_common.sh`. To run
a compose profile directly (the `<step>_common.sh` scripts do this for you):

```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=tupi
#SBATCH --gres=gpu:1

# cep / judge / kg / rag (+ -gpu variants); each brings up its ollama sidecar
docker compose --profile cep up --abort-on-container-exit
```

## Using Singularity/Apptainer

For HPC clusters without Docker, build a `.sif` from the locally built image
(`docker build -f Dockerfile -t arandu:latest .`; KG/RAG use `Dockerfile.kg`):

```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=tupi
#SBATCH --gres=gpu:1

# Build the .sif once from the local Docker image
singularity build arandu.sif docker-daemon://arandu:latest

# Run a pipeline command (ENTRYPOINT is `arandu`)
singularity exec --nv arandu.sif arandu generate-cep-qa results/ -o qa_dataset/
```

## Resource Recommendations

### QA Generation

| Resource | Recommendation |
|----------|----------------|
| CPUs | 8-16 per worker |
| Memory | 16-32 GB |
| Time | 1-4 hours (depends on corpus size) |

### KG Construction

| Resource | Recommendation |
|----------|----------------|
| CPUs | 16-32 per worker |
| Memory | 32-64 GB |
| Time | 2-8 hours (depends on corpus size) |

### GPU vs CPU

- GPU recommended for Ollama LLM inference
- CPU-only possible but slower
- Set `--gres=gpu:0` for CPU-only

## CEP QA Generation Scripts

The CEP (Cognitive Elicitation Pipeline) pipeline has dedicated SLURM scripts organized by cluster partition:

```
scripts/slurm/cep/
├── cep_common.sh      # Shared logic for all CEP jobs
├── grace.slurm        # Grace partition (NVIDIA L40S)
├── tupi.slurm         # Tupi partition (NVIDIA RTX 4090)
└── sirius.slurm       # Sirius partition (AMD, CPU mode)
```

### Submit CEP Jobs

```bash
# Tupi partition (the GPU partition to use on PCAD)
PIPELINE_ID=run-01 sbatch scripts/slurm/cep/tupi.slurm

# Sirius partition (CPU-only, for AMD nodes)
PIPELINE_ID=run-01 sbatch scripts/slurm/cep/sirius.slurm
```

> **Partition access (PCAD)**: `tupi` is the only usable GPU partition;
> `grace` is **not accessible** (jobs PEND forever with
> `uid_..._not_in_group_permitted`), so `cep/grace.slurm` exists but won't run
> as submitted. See [`scripts/slurm/AGENTS.md`](../../scripts/slurm/AGENTS.md)
> for the authoritative partition rules.

### CEP Script Architecture

Each partition script sources `cep_common.sh` which handles:

1. **Ollama Sidecar Management**: Starts Ollama container, pulls required model
2. **Container Lifecycle**: Unique container names per job to avoid conflicts
3. **CEP Configuration**: Bloom taxonomy distribution, validation settings
4. **Cleanup**: Automatic container removal on job completion

### CEP-Specific Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARANDU_QA_WORKERS` | Parallel workers for CEP generation | Partition-dependent |
| `USE_GPU_OLLAMA` | Enable GPU acceleration for Ollama | `true` (GPU partitions) |
| `ARANDU_CEP_LANGUAGE` | Prompt language (`pt` or `en`) | `pt` |

> **Note**: `generate-cep-qa` only generates pairs; it has no validation toggle
> (CEPConfig has no `enable_validation`). CEP-pair validation runs as the separate
> `judge-qa` command, configured via `ARANDU_JUDGE_VALIDATOR_*`. Run it after
> generation (e.g. a chained job with `--dependency=afterok:`).

### Override CEP Settings

```bash
# Use English prompts
ARANDU_CEP_LANGUAGE=en sbatch scripts/slurm/cep/tupi.slurm
```

### Validating CEP Pairs (separate job)

Validation runs as the dedicated `judge-qa` job, configured with `ARANDU_JUDGE_*`:

```bash
# Judge a populated run's CEP dataset
PIPELINE_ID=test-cep-01 \
ARANDU_JUDGE_VALIDATOR_MODEL=llama3.3:70b \
sbatch scripts/slurm/judge/qa/tupi.slurm

# Force a fresh pass over already-judged pairs
PIPELINE_ID=test-cep-01 JUDGE_REJUDGE=1 \
sbatch scripts/slurm/judge/qa/tupi.slurm
```

### CEP Resource Recommendations

| Partition | GPUs | Workers | Best For |
|-----------|------|---------|----------|
| Tupi (RTX 4090) | 1 | 3 | Standard generation (14B models); the GPU partition to use |
| Sirius (AMD) | 0 | 2 | CPU-only fallback |
| Grace (L40S) | 1 | 4 | Large models (70B) **but not accessible** on PCAD (jobs PEND); script exists, do not submit |

### Monitor CEP Jobs

```bash
# View job output
tail -f logs/cep_tupi_<jobid>.out

# Check container status
docker ps --filter name=ollama-cep
docker ps --filter name=arandu-cep
```

## Checkpoint and Resume

All pipelines checkpoint under `results/<id>/`. If a job fails or hits the wall
limit, resubmit the exact same command; it resumes automatically (KG extraction
and concept generation both checkpoint, so a `TIMEOUT` just needs a resubmit):

```bash
PIPELINE_ID=run-01 sbatch scripts/slurm/cep/tupi.slurm
```

To start fresh, remove that run's checkpoint under
`results/<id>/<stage>/` before resubmitting.

## Troubleshooting

### Job Fails Immediately

Check error log:
```bash
cat logs/arandu_<job_id>.err
```

### Out of Memory

Reduce workers:
```bash
PIPELINE_ID=run-01 ARANDU_QA_WORKERS=2 sbatch scripts/slurm/cep/tupi.slurm
```

### A "silent" log is not a hang

Long LLM loops (KG concept generation especially) buffer stdout and write
progress to result files instead. Check artifact mtimes under
`results/<id>/...` before assuming a job is stuck. See
[`scripts/slurm/AGENTS.md`](../../scripts/slurm/AGENTS.md) for how to attach to a
running allocation.

---

**See also**: [PCAD Guide](pcad.md) | [Docker Deployment](docker.md)
