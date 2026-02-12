# Generic SLURM Deployment Guide

This guide covers running G-Transcriber on any SLURM-based HPC cluster.

## Prerequisites

- SLURM job scheduler
- Docker or Singularity/Apptainer (recommended for HPC)
- GPU nodes (NVIDIA CUDA or AMD ROCm)

## SLURM Scripts

G-Transcriber provides SLURM scripts for each pipeline in `scripts/slurm/`:

| Script | Pipeline | Description |
|--------|----------|-------------|
| `run_transcription.slurm` | Transcription | Batch transcription with Whisper |
| `run_qa_generation.slurm` | QA | Generate QA pairs from transcriptions |
| `cep/grace.slurm` | CEP QA | Cognitive QA generation (Grace/L40S) |
| `cep/tupi.slurm` | CEP QA | Cognitive QA generation (Tupi/RTX 4090) |
| `cep/sirius.slurm` | CEP QA | Cognitive QA generation (Sirius/AMD CPU) |
| `run_kg_construction.slurm` | KG | Build knowledge graphs |
| `run_evaluation.slurm` | Evaluation | Compute quality metrics |

## Basic Usage

### Submit a Job

```bash
sbatch scripts/slurm/run_qa_generation.slurm
```

### Override Settings

```bash
# Override workers and model
WORKERS=8 QA_MODEL=qwen3:14b sbatch scripts/slurm/run_qa_generation.slurm

# Override partition
sbatch --partition=gpu scripts/slurm/run_kg_construction.slurm
```

### Monitor Jobs

```bash
# View your jobs
squeue -u $USER

# View job details
scontrol show job <job_id>

# View logs
tail -f logs/gtranscriber_<job_id>.out
```

### Cancel a Job

```bash
scancel <job_id>
```

## Script Structure

A typical SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/gtranscriber_%j.out
#SBATCH --error=logs/gtranscriber_%j.err

# Load modules (cluster-specific)
module load python/3.13
module load cuda/12.4

# Activate environment
source .venv/bin/activate

# Set environment variables
export GTRANSCRIBER_QA_PROVIDER=ollama
export GTRANSCRIBER_QA_MODEL_ID=${QA_MODEL:-qwen3:14b}
export GTRANSCRIBER_WORKERS=${WORKERS:-4}

# Run pipeline
gtranscriber generate-cep-qa results/ -o qa_dataset/
```

## Environment Variables

Override via command line or in script:

| Variable | Description |
|----------|-------------|
| `WORKERS` | Number of parallel workers |
| `QA_MODEL` | Model for QA generation |
| `KG_MODEL` | Model for KG construction |
| `QA_PROVIDER` | LLM provider (ollama, openai) |
| `QUESTIONS_PER_DOCUMENT` | QA pairs per document |
| `KG_LANGUAGE` | Language for KG extraction |

## Using Docker on SLURM

If Docker is available:

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Run via Docker Compose
docker compose --profile qa up --abort-on-container-exit
```

## Using Singularity/Apptainer

For HPC clusters without Docker:

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Build container (once)
singularity build gtranscriber.sif docker://ghcr.io/fredDsR/gtranscriber:latest

# Run
singularity exec --nv gtranscriber.sif \
    gtranscriber generate-cep-qa results/ -o qa_dataset/
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
# Grace partition (best for large models)
sbatch scripts/slurm/cep/grace.slurm

# Tupi partition (good balance of speed/availability)
sbatch scripts/slurm/cep/tupi.slurm

# Sirius partition (CPU-only, for AMD nodes)
sbatch scripts/slurm/cep/sirius.slurm
```

### CEP Script Architecture

Each partition script sources `cep_common.sh` which handles:

1. **Ollama Sidecar Management**: Starts Ollama container, pulls required model
2. **Container Lifecycle**: Unique container names per job to avoid conflicts
3. **CEP Configuration**: Bloom taxonomy distribution, validation settings
4. **Cleanup**: Automatic container removal on job completion

### CEP-Specific Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GTRANSCRIBER_QA_WORKERS` | Parallel workers for CEP generation | Partition-dependent |
| `USE_GPU_OLLAMA` | Enable GPU acceleration for Ollama | `true` (GPU partitions) |
| `GTRANSCRIBER_CEP_ENABLE_VALIDATION` | Enable LLM-as-a-Judge validation | `true` |
| `GTRANSCRIBER_CEP_LANGUAGE` | Prompt language (`pt` or `en`) | `pt` |

### Override CEP Settings

```bash
# Disable validation for faster processing
GTRANSCRIBER_CEP_ENABLE_VALIDATION=false \
sbatch scripts/slurm/cep/tupi.slurm

# Use custom validator model
GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=llama3.3:70b \
sbatch scripts/slurm/cep/grace.slurm

# Use English prompts
GTRANSCRIBER_CEP_LANGUAGE=en sbatch scripts/slurm/cep/tupi.slurm
```

### CEP Resource Recommendations

| Partition | GPUs | Workers | Best For |
|-----------|------|---------|----------|
| Grace (L40S) | 1 | 4 | Large models (70B), validation |
| Tupi (RTX 4090) | 1 | 3 | Standard generation (14B models) |
| Sirius (AMD) | 0 | 2 | CPU-only fallback |

### Monitor CEP Jobs

```bash
# View job output
tail -f logs/cep_grace_<jobid>.out

# Check container status
docker ps --filter name=ollama-cep
docker ps --filter name=gtranscriber-cep
```

## Checkpoint and Resume

All pipelines support checkpointing. If a job fails:

```bash
# Simply resubmit - checkpoint resumes automatically
sbatch scripts/slurm/run_qa_generation.slurm
```

To start fresh:

```bash
rm qa_dataset/qa_checkpoint.json
sbatch scripts/slurm/run_qa_generation.slurm
```

## Troubleshooting

### Job Fails Immediately

Check error log:
```bash
cat logs/gtranscriber_<job_id>.err
```

### Out of Memory

Reduce workers:
```bash
WORKERS=2 sbatch scripts/slurm/run_qa_generation.slurm
```

### Module Not Found

Ensure Python environment is activated in script.

---

**See also**: [PCAD Guide](pcad.md) | [Docker Deployment](docker.md)
