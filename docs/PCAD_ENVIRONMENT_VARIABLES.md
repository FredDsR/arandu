# PCAD Environment Variables Report

This document provides a comprehensive overview of where and how environment variables are set when running G-Transcriber jobs on PCAD (Parque Computacional de Alto Desempenho).

## Table of Contents

1. [Overview](#overview)
2. [Configuration Hierarchy](#configuration-hierarchy)
3. [Environment Variable Sources](#environment-variable-sources)
4. [Complete Variable Reference](#complete-variable-reference)
5. [Variable Flow Diagram](#variable-flow-diagram)

---

## Overview

The G-Transcriber system uses a multi-layered configuration approach when running on PCAD, with environment variables set at different levels:

1. **User Level**: `.env` file and command-line overrides
2. **Partition Level**: Partition-specific SLURM scripts
3. **Job Level**: Common job script (`job_common.sh`)
4. **Docker Level**: `docker-compose.yml` and container environment
5. **Application Level**: Python Pydantic settings in `config.py`

---

## Configuration Hierarchy

The configuration follows this precedence order (highest to lowest):

```
1. Command-line overrides (sbatch submission)
   ↓
2. Partition-specific SLURM script exports
   ↓
3. job_common.sh computations and exports
   ↓
4. .env file (if present)
   ↓
5. Application defaults (config.py)
```

---

## Environment Variable Sources

### 1. User-Provided Configuration

#### Location: `.env` file (project root)
**File**: [.env.example](.env.example)

Variables that can be set by users before syncing to PCAD:

```bash
# Model and Processing
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3-turbo
GTRANSCRIBER_WORKERS=4
GTRANSCRIBER_QUANTIZE=true
GTRANSCRIBER_FORCE_CPU=false

# Paths
GTRANSCRIBER_CATALOG_FILE=catalog.csv
GTRANSCRIBER_INPUT_DIR=./input
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_CREDENTIALS_DIR=./
GTRANSCRIBER_HF_CACHE_DIR=./cache/huggingface

# GPU Settings
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=

# AMD ROCm (for sirius partition)
USE_ROCM=false
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

**When loaded**:
- By Docker containers (via docker-compose.yml)
- By Python application (via Pydantic Settings)
- Copied to `$SCRATCH` if scratch optimization is enabled

---

### 2. Command-Line Overrides (at Job Submission)

#### Location: Terminal when running `sbatch`
**Documentation**: [PCAD_GUIDE.md](PCAD_GUIDE.md)

Users can override settings when submitting jobs:

```bash
# Override workers
WORKERS=6 sbatch scripts/slurm/tupi.slurm

# Override model
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3 sbatch scripts/slurm/grace.slurm

# Override catalog file
CATALOG_FILE=my_subset.csv sbatch scripts/slurm/tupi.slurm

# Force CPU mode
USE_CPU=true sbatch scripts/slurm/draco.slurm

# Disable SCRATCH optimization
USE_SCRATCH=false sbatch scripts/slurm/tupi.slurm
```

**Precedence**: These override all other sources

---

### 3. Partition-Specific SLURM Scripts

Each PCAD partition has a dedicated script that sets partition-specific defaults.

#### tupi.slurm (RTX 4090)
**File**: [scripts/slurm/tupi.slurm](../scripts/slurm/tupi.slurm:21-22)

```bash
export WORKERS="${WORKERS:-4}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"
```

#### grace.slurm (L40S - 46GB VRAM)
**File**: [scripts/slurm/grace.slurm](../scripts/slurm/grace.slurm:21-22)

```bash
export WORKERS="${WORKERS:-6}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3}"
```

#### sirius.slurm (AMD RX 7900 - ROCm)
**File**: [scripts/slurm/sirius.slurm](../scripts/slurm/sirius.slurm:21-23)

```bash
export USE_ROCM=true
export WORKERS="${WORKERS:-3}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}"
```

#### turing.slurm (CPU-only)
**File**: [scripts/slurm/turing.slurm](../scripts/slurm/turing.slurm:22-24)

```bash
export USE_CPU=true
export WORKERS="${WORKERS:-8}"
export GTRANSCRIBER_MODEL_ID="${GTRANSCRIBER_MODEL_ID:-distil-whisper/distil-large-v3}"
```

**Additional partitions**: `blaise.slurm` and `draco.slurm` follow similar patterns

---

### 4. Common Job Script (job_common.sh)

#### Location: `scripts/slurm/job_common.sh`
**File**: [scripts/slurm/job_common.sh](../scripts/slurm/job_common.sh)

This is the core script that processes and exports all variables. It's sourced by all partition scripts.

#### 4.1. Initial Configuration
**Lines**: [23-39](../scripts/slurm/job_common.sh:23-39)

```bash
# Project directory
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Support both GTRANSCRIBER_WORKERS and WORKERS
if [ -n "${GTRANSCRIBER_WORKERS:-}" ]; then
    WORKERS="${GTRANSCRIBER_WORKERS}"
fi
WORKERS="${WORKERS:-1}"

# Support both GTRANSCRIBER_CATALOG_FILE and CATALOG_FILE
if [ -n "${GTRANSCRIBER_CATALOG_FILE:-}" ]; then
    CATALOG_FILE="${GTRANSCRIBER_CATALOG_FILE}"
fi
CATALOG_FILE="${CATALOG_FILE:-catalog.csv}"

USE_CPU="${USE_CPU:-false}"
USE_ROCM="${USE_ROCM:-false}"
USE_SCRATCH="${USE_SCRATCH:-true}"
```

**Purpose**: Provides defaults and backward compatibility for old/new variable names

---

#### 4.2. SCRATCH Directory Setup
**Lines**: [51-109](../scripts/slurm/job_common.sh:51-109)

If `USE_SCRATCH=true` and `$SCRATCH` is available:

1. Creates working directory: `$SCRATCH/gtranscriber_${SLURM_JOB_ID}`
2. Copies files from `$HOME` to `$SCRATCH`:
   - Input catalog
   - `.env` file
   - `credentials.json`
   - `token.json`
   - Existing checkpoint (if resuming)
   - Hugging Face cache (if exists)

**Result**: Sets these variables:
```bash
SCRATCH_WORK_DIR="$SCRATCH/gtranscriber_${SLURM_JOB_ID}"
USING_SCRATCH=true
```

---

#### 4.3. Working Directory Selection
**Lines**: [211-230](../scripts/slurm/job_common.sh:211-230)

Based on whether SCRATCH is enabled:

```bash
if setup_scratch; then
    # Use SCRATCH directories
    WORK_INPUT_DIR="$SCRATCH_WORK_DIR/input"
    WORK_RESULTS_DIR="$SCRATCH_WORK_DIR/results"
    WORK_CREDENTIALS_DIR="$SCRATCH_WORK_DIR/credentials"
    WORK_HF_CACHE_DIR="$SCRATCH_WORK_DIR/cache/huggingface"
else
    # Fallback to $HOME directories
    WORK_INPUT_DIR="$PROJECT_DIR/input"
    WORK_RESULTS_DIR="$PROJECT_DIR/results"
    WORK_CREDENTIALS_DIR="$PROJECT_DIR"
    WORK_HF_CACHE_DIR="$PROJECT_DIR/cache/huggingface"
fi
```

---

#### 4.4. Environment Variable Export for Docker
**Lines**: [233-272](../scripts/slurm/job_common.sh:233-272)

Exports all variables needed by Docker Compose:

```bash
# SLURM job information
export SLURM_JOB_ID

# Workers (both old and new names)
export WORKERS
export GTRANSCRIBER_WORKERS="$WORKERS"

# Model
export GTRANSCRIBER_MODEL_ID

# Catalog file (both old and new names)
export CATALOG_FILE
export GTRANSCRIBER_CATALOG_FILE="$CATALOG_FILE"

# Paths (both old and new names for backward compatibility)
export INPUT_DIR="$WORK_INPUT_DIR"
export GTRANSCRIBER_INPUT_DIR="$WORK_INPUT_DIR"

export RESULTS_DIR="$WORK_RESULTS_DIR"
export GTRANSCRIBER_RESULTS_DIR="$WORK_RESULTS_DIR"

export CREDENTIALS_DIR="$WORK_CREDENTIALS_DIR"
export GTRANSCRIBER_CREDENTIALS_DIR="$WORK_CREDENTIALS_DIR"

export HF_CACHE_DIR="$WORK_HF_CACHE_DIR"
export GTRANSCRIBER_HF_CACHE_DIR="$WORK_HF_CACHE_DIR"

# Quantization and CPU flags
if [ "$USE_CPU" = "true" ]; then
    export GTRANSCRIBER_QUANTIZE=false
    export GTRANSCRIBER_FORCE_CPU=true
    export QUANTIZE_FLAG=""
    export CPU_FLAG="--cpu"
else
    export GTRANSCRIBER_QUANTIZE=true
    export GTRANSCRIBER_FORCE_CPU=false
    export QUANTIZE_FLAG="--quantize"
    export CPU_FLAG=""
fi
```

**Note**: Dual variable names (`WORKERS`/`GTRANSCRIBER_WORKERS`) maintain backward compatibility

---

### 5. Docker Compose Configuration

#### Location: `docker-compose.yml`
**File**: [docker-compose.yml](../docker-compose.yml)

Docker Compose receives exported variables and passes them to containers.

#### 5.1. NVIDIA GPU Service (gtranscriber)
**Lines**: [18-53](../docker-compose.yml:18-53)

```yaml
environment:
  - GTRANSCRIBER_MODEL_ID=${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}
  - GTRANSCRIBER_QUANTIZE=${GTRANSCRIBER_QUANTIZE:-true}
  - GTRANSCRIBER_FORCE_CPU=${GTRANSCRIBER_FORCE_CPU:-false}
  - GTRANSCRIBER_CREDENTIALS=/app/credentials/credentials.json
  - GTRANSCRIBER_TOKEN=/app/credentials/token.json
  - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
  - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
  - HF_HOME=/app/.cache/huggingface
  - TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

command: >
  batch-transcribe
  /app/input/${GTRANSCRIBER_CATALOG_FILE:-${CATALOG_FILE:-catalog.csv}}
  --credentials /app/credentials/credentials.json
  --token /app/credentials/token.json
  --output-dir /app/results
  --workers ${GTRANSCRIBER_WORKERS:-${WORKERS:-4}}
  ${QUANTIZE_FLAG:---quantize}
  ${CPU_FLAG:-}
```

#### 5.2. CPU Service (gtranscriber-cpu)
**Lines**: [78-94](../docker-compose.yml:78-94)

```yaml
environment:
  - GTRANSCRIBER_MODEL_ID=${GTRANSCRIBER_MODEL_ID:-distil-whisper/distil-large-v3}
  - GTRANSCRIBER_QUANTIZE=false
  - GTRANSCRIBER_FORCE_CPU=true
  - GTRANSCRIBER_CREDENTIALS=/app/credentials/credentials.json
  - GTRANSCRIBER_TOKEN=/app/credentials/token.json
  - HF_HOME=/app/.cache/huggingface
  - TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

command: >
  batch-transcribe
  /app/input/${GTRANSCRIBER_CATALOG_FILE:-${CATALOG_FILE:-catalog.csv}}
  --credentials /app/credentials/credentials.json
  --token /app/credentials/token.json
  --output-dir /app/results
  --workers ${GTRANSCRIBER_WORKERS:-${WORKERS:-2}}
  --cpu
```

#### 5.3. AMD ROCm Service (gtranscriber-rocm)
**Lines**: [116-143](../docker-compose.yml:116-143)

```yaml
environment:
  - HSA_OVERRIDE_GFX_VERSION=11.0.0
  - GTRANSCRIBER_MODEL_ID=${GTRANSCRIBER_MODEL_ID:-openai/whisper-large-v3-turbo}
  - GTRANSCRIBER_QUANTIZE=false
  - GTRANSCRIBER_FORCE_CPU=false
  - GTRANSCRIBER_CREDENTIALS=/app/credentials/credentials.json
  - GTRANSCRIBER_TOKEN=/app/credentials/token.json
  - HF_HOME=/app/.cache/huggingface
  - TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

command: >
  batch-transcribe
  /app/input/${GTRANSCRIBER_CATALOG_FILE:-${CATALOG_FILE:-catalog.csv}}
  --credentials /app/credentials/credentials.json
  --token /app/credentials/token.json
  --output-dir /app/results
  --workers ${GTRANSCRIBER_WORKERS:-${WORKERS:-3}}
```

**Note**: ROCm has `GTRANSCRIBER_QUANTIZE=false` because bitsandbytes doesn't support ROCm

---

### 6. Python Application Configuration

#### Location: `src/gtranscriber/config.py`
**File**: [src/gtranscriber/config.py](../src/gtranscriber/config.py)

Uses Pydantic Settings to load environment variables.

#### Configuration Class
**Lines**: [17-29](../src/gtranscriber/config.py:17-29)

```python
class TranscriberConfig(BaseSettings):
    """Configuration settings for the transcriber.

    Settings are loaded from environment variables with the GTRANSCRIBER_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
```

**Loading priority**:
1. Environment variables with `GTRANSCRIBER_` prefix
2. `.env` file (if present)
3. Default values defined in the class

#### Application Settings
**Lines**: [31-118](../src/gtranscriber/config.py:31-118)

```python
# Model settings
model_id: str = "openai/whisper-large-v3"
return_timestamps: bool = True
chunk_length_s: int = 30
stride_length_s: int = 5

# Hardware settings
force_cpu: bool = False
quantize: bool = False
quantize_bits: int = 8

# Google Drive settings
credentials: str = "credentials.json"
token: str = "token.json"

# Batch processing settings
workers: int = 1
catalog_file: str = "catalog.csv"

# Path settings
input_dir: str = "./input"
results_dir: str = "./results"
credentials_dir: str = "./"
hf_cache_dir: str = "./cache/huggingface"

# Processing settings
temp_dir: str = (platform-specific temp dir)
max_retries: int = 3
retry_delay: float = 1.0
```

---

## Complete Variable Reference

### Core Configuration Variables

| Variable | Set By | Purpose | Example Values |
|----------|--------|---------|----------------|
| `GTRANSCRIBER_MODEL_ID` | Partition scripts / .env | Whisper model to use | `openai/whisper-large-v3-turbo` |
| `WORKERS` / `GTRANSCRIBER_WORKERS` | Partition scripts / .env | Number of parallel workers | `4`, `6`, `8` |
| `CATALOG_FILE` / `GTRANSCRIBER_CATALOG_FILE` | job_common.sh / .env | Input catalog filename | `catalog.csv` |
| `USE_CPU` | Partition scripts / sbatch | Force CPU mode | `true`, `false` |
| `USE_ROCM` | Partition scripts | Enable AMD ROCm | `true`, `false` |
| `USE_SCRATCH` | sbatch / job_common.sh | Enable SCRATCH optimization | `true`, `false` |

### Path Variables

| Variable | Set By | Purpose | Value Pattern |
|----------|--------|---------|---------------|
| `PROJECT_DIR` | job_common.sh | Project root on PCAD | `$HOME/etno-kgc-preprocessing` |
| `INPUT_DIR` / `GTRANSCRIBER_INPUT_DIR` | job_common.sh | Input directory path | `$SCRATCH/.../input` or `$PROJECT_DIR/input` |
| `RESULTS_DIR` / `GTRANSCRIBER_RESULTS_DIR` | job_common.sh | Results directory path | `$SCRATCH/.../results` or `$PROJECT_DIR/results` |
| `CREDENTIALS_DIR` / `GTRANSCRIBER_CREDENTIALS_DIR` | job_common.sh | Credentials directory | `$SCRATCH/.../credentials` or `$PROJECT_DIR` |
| `HF_CACHE_DIR` / `GTRANSCRIBER_HF_CACHE_DIR` | job_common.sh | Hugging Face cache | `$SCRATCH/.../cache/huggingface` or `$PROJECT_DIR/cache/...` |

### Hardware Configuration

| Variable | Set By | Purpose | Example Values |
|----------|--------|---------|----------------|
| `GTRANSCRIBER_QUANTIZE` | job_common.sh / .env | Enable 8-bit quantization | `true`, `false` |
| `GTRANSCRIBER_FORCE_CPU` | job_common.sh / .env | Force CPU execution | `true`, `false` |
| `NVIDIA_VISIBLE_DEVICES` | Docker Compose / .env | NVIDIA GPU selection | `all`, `0`, `0,1` |
| `CUDA_VISIBLE_DEVICES` | Docker Compose / .env | CUDA device selection | empty, `0`, `0,1` |
| `HSA_OVERRIDE_GFX_VERSION` | Docker Compose / .env | AMD ROCm GFX override | `11.0.0` |

### SLURM Variables (Automatic)

| Variable | Set By | Purpose |
|----------|--------|---------|
| `SLURM_JOB_ID` | SLURM | Unique job identifier |
| `SLURM_JOB_NAME` | SLURM | Job name |
| `SLURM_JOB_PARTITION` | SLURM | Partition name |
| `SLURM_CPUS_PER_TASK` | SLURM | Number of CPUs allocated |
| `SLURM_SUBMIT_DIR` | SLURM | Directory where job was submitted |

### Internal Working Variables

| Variable | Set By | Purpose | Visibility |
|----------|--------|---------|------------|
| `SCRATCH_WORK_DIR` | job_common.sh | SCRATCH working directory | Internal only |
| `USING_SCRATCH` | job_common.sh | SCRATCH status flag | Internal only |
| `QUANTIZE_FLAG` | job_common.sh | CLI flag for quantization | Docker command |
| `CPU_FLAG` | job_common.sh | CLI flag for CPU mode | Docker command |

### Docker Container Variables

| Variable | Set By | Purpose | Container Path |
|----------|--------|---------|----------------|
| `GTRANSCRIBER_CREDENTIALS` | Docker Compose | Credentials file path | `/app/credentials/credentials.json` |
| `GTRANSCRIBER_TOKEN` | Docker Compose | Token file path | `/app/credentials/token.json` |
| `HF_HOME` | Docker Compose | Hugging Face home | `/app/.cache/huggingface` |
| `TRANSFORMERS_CACHE` | Docker Compose | Transformers cache | `/app/.cache/huggingface/hub` |

---

## Variable Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User Configuration                                           │
│    • .env file                                                  │
│    • Command-line: WORKERS=6 sbatch scripts/slurm/tupi.slurm   │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. SLURM Partition Script (e.g., tupi.slurm)                   │
│    • Sets partition defaults: WORKERS, GTRANSCRIBER_MODEL_ID   │
│    • Sources job_common.sh                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. job_common.sh Processing                                     │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 3a. Variable Normalization                              │ │
│    │     • Merge GTRANSCRIBER_* and legacy names             │ │
│    │     • Apply defaults                                    │ │
│    └─────────────────────────────────────────────────────────┘ │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 3b. SCRATCH Setup (if USE_SCRATCH=true)                │ │
│    │     • Create $SCRATCH/gtranscriber_${SLURM_JOB_ID}     │ │
│    │     • Copy .env, credentials, catalog, cache           │ │
│    │     • Set WORK_*_DIR to SCRATCH paths                  │ │
│    └─────────────────────────────────────────────────────────┘ │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ 3c. Export for Docker                                   │ │
│    │     • Export all GTRANSCRIBER_* variables              │ │
│    │     • Export path variables (INPUT_DIR, etc.)          │ │
│    │     • Set QUANTIZE_FLAG and CPU_FLAG                   │ │
│    └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Docker Compose (docker-compose.yml)                         │
│    • Receives exported shell variables                          │
│    • Maps to container environment variables                    │
│    • Selects service: gtranscriber / gtranscriber-cpu / -rocm  │
│    • Builds Docker command with flags                           │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Docker Container                                             │
│    • Mounts volumes: input, results, credentials, cache         │
│    • Passes environment variables to Python app                 │
│    • Executes: batch-transcribe with CLI args                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Python Application (config.py)                              │
│    • Pydantic Settings loads GTRANSCRIBER_* variables          │
│    • Falls back to .env file if present                        │
│    • Uses class defaults as final fallback                     │
│    • Creates TranscriberConfig instance                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example: Tracing a Variable Through the System

Let's trace `WORKERS=6` from submission to application:

### Step 1: Job Submission
```bash
WORKERS=6 sbatch scripts/slurm/tupi.slurm
```
**Result**: `WORKERS=6` set in shell environment

---

### Step 2: tupi.slurm
```bash
export WORKERS="${WORKERS:-4}"
```
**Result**: `WORKERS=6` (uses existing value, not default)

---

### Step 3: job_common.sh - Normalization
```bash
if [ -n "${GTRANSCRIBER_WORKERS:-}" ]; then
    WORKERS="${GTRANSCRIBER_WORKERS}"
fi
WORKERS="${WORKERS:-1}"
```
**Result**: `WORKERS=6` (no GTRANSCRIBER_WORKERS set, uses existing WORKERS)

---

### Step 4: job_common.sh - Export
```bash
export WORKERS
export GTRANSCRIBER_WORKERS="$WORKERS"
```
**Result**: Both `WORKERS=6` and `GTRANSCRIBER_WORKERS=6` exported

---

### Step 5: docker-compose.yml
```yaml
command: >
  batch-transcribe
  ...
  --workers ${GTRANSCRIBER_WORKERS:-${WORKERS:-4}}
```
**Result**: `--workers 6` passed to command (uses GTRANSCRIBER_WORKERS)

---

### Step 6: Python Application
```python
workers: int = Field(default=1, ...)
```
**Result**: Pydantic loads from `GTRANSCRIBER_WORKERS=6` environment variable

**Final value**: `config.workers = 6`

---

## Summary

Environment variables in the PCAD G-Transcriber system flow through six distinct layers:

1. **User Configuration** (.env file, command-line)
2. **Partition Defaults** (SLURM partition scripts)
3. **Processing & Export** (job_common.sh)
4. **Container Mapping** (docker-compose.yml)
5. **Container Runtime** (Docker environment)
6. **Application Loading** (Pydantic Settings)

The system maintains backward compatibility by supporting both old (`WORKERS`, `CATALOG_FILE`) and new (`GTRANSCRIBER_*`) variable names throughout the stack. The SCRATCH optimization transparently redirects all paths to local fast storage when available, with automatic cleanup and result copying.

All variables follow a clear precedence: command-line > partition defaults > .env > application defaults, ensuring predictable behavior while maintaining flexibility for users to override settings at any level.
