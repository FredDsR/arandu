# Running G-Transcriber Locally with Docker Compose

This guide covers how to run the transcription process on your local machine using Docker Compose.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Running Transcription](#running-transcription)
4. [Configuration Options](#configuration-options)
5. [Managing Containers](#managing-containers)
6. [Viewing Logs and Progress](#viewing-logs-and-progress)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+ or Docker Desktop)
- **NVIDIA Container Toolkit** (for GPU support)

### Install Docker

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
```

**macOS/Windows:**
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### Install NVIDIA Container Toolkit (GPU Support)

For NVIDIA GPU support on Linux:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Google OAuth Credentials

Ensure you have valid credentials:

```bash
# Verify credentials exist
ls -la credentials.json token.json

# Refresh token if needed
gtranscriber info
```

---

## Initial Setup

### 1. Clone/Navigate to Project

```bash
cd /path/to/etno-kgc-preprocessing
```

### 2. Create Environment File

```bash
cp .env.example .env
```

### 3. Configure Settings

Edit `.env` to customize your setup:

```bash
# Whisper model (adjust based on your GPU VRAM)
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3-turbo

# Number of workers (adjust based on GPU VRAM)
# 24GB VRAM: 4 workers
# 16GB VRAM: 2-3 workers
# 8GB VRAM: 1-2 workers
WORKERS=4

# Enable quantization (reduces VRAM usage by ~50%)
GTRANSCRIBER_QUANTIZE=true

# Input catalog file
CATALOG_FILE=catalog.csv
```

### 4. Verify Input Catalog

```bash
ls -la input/catalog.csv
```

### 5. Build Docker Image

```bash
docker compose --profile gpu build gtranscriber
```

---

## Running Transcription

### GPU Mode (Recommended)

Run with NVIDIA GPU acceleration:

```bash
docker compose --profile gpu up gtranscriber
```

### CPU Mode

Run on CPU only (slower but works without GPU):

```bash
docker compose --profile cpu up gtranscriber-cpu
```

### Run in Background (Detached)

```bash
# GPU mode
docker compose --profile gpu up -d gtranscriber

# CPU mode
docker compose --profile cpu up -d gtranscriber-cpu
```

### Run with Custom Settings

Override settings without editing `.env`:

```bash
# Use more workers
WORKERS=6 docker compose --profile gpu up gtranscriber

# Use a different model
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3 docker compose --profile gpu up gtranscriber

# Use a different catalog
CATALOG_FILE=my_subset.csv docker compose --profile gpu up gtranscriber

# Combine multiple overrides
WORKERS=2 GTRANSCRIBER_MODEL_ID=distil-whisper/distil-large-v3 docker compose --profile gpu up gtranscriber
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_MODEL_ID` | `openai/whisper-large-v3-turbo` | Whisper model from Hugging Face |
| `WORKERS` | `4` | Number of parallel transcription workers |
| `GTRANSCRIBER_QUANTIZE` | `true` | Enable 8-bit quantization (reduces VRAM) |
| `GTRANSCRIBER_FORCE_CPU` | `false` | Force CPU execution |
| `CATALOG_FILE` | `catalog.csv` | Input catalog filename |
| `INPUT_DIR` | `./input` | Directory containing catalog |
| `RESULTS_DIR` | `./results` | Output directory for results |
| `CREDENTIALS_DIR` | `./` | Directory containing credentials |
| `HF_CACHE_DIR` | `./cache/huggingface` | Hugging Face model cache |

### Model Selection Guide

| Model | VRAM Required | Speed | Accuracy | Best For |
|-------|---------------|-------|----------|----------|
| `openai/whisper-large-v3` | ~10GB | Slow | Highest | Final production runs |
| `openai/whisper-large-v3-turbo` | ~6GB | Medium | High | Good balance |
| `distil-whisper/distil-large-v3` | ~3GB | Fast | Good | Quick processing, limited VRAM |

### Worker Configuration Guide

| GPU VRAM | Recommended Workers | With Quantization |
|----------|---------------------|-------------------|
| 24GB (RTX 4090) | 3-4 | 5-6 |
| 16GB (RTX 4080) | 2-3 | 3-4 |
| 12GB (RTX 4070) | 1-2 | 2-3 |
| 8GB (RTX 3070) | 1 | 1-2 |

---

## Managing Containers

### View Running Containers

```bash
docker compose ps
```

### Stop Transcription

```bash
# Graceful stop (allows current file to complete)
docker compose stop

# Force stop
docker compose kill
```

### Remove Containers

```bash
docker compose down
```

### Rebuild After Code Changes

```bash
docker compose build --no-cache
```

### Clean Up Docker Resources

```bash
# Remove stopped containers and unused images
docker system prune

# Remove everything including volumes (careful!)
docker system prune -a --volumes
```

---

## Viewing Logs and Progress

### View Live Logs

```bash
# Follow logs in real-time
docker compose logs -f gtranscriber

# View last 100 lines
docker compose logs --tail 100 gtranscriber
```

### Check Progress

```bash
# Count completed transcriptions
ls -1 results/*_transcription.json 2>/dev/null | wc -l

# View checkpoint status
cat results/checkpoint.json | python -m json.tool
```

### Detailed Progress Script

```bash
python -c "
import json
from pathlib import Path

checkpoint = Path('results/checkpoint.json')
if checkpoint.exists():
    with open(checkpoint) as f:
        cp = json.load(f)
    completed = len(cp.get('completed_files', []))
    failed = len(cp.get('failed_files', {}))
    total = cp.get('total_files', 'unknown')
    print(f'Progress: {completed}/{total} completed')
    print(f'Failed: {failed}')
    if cp.get('failed_files'):
        print('Failed files:')
        for fid, err in cp['failed_files'].items():
            print(f'  - {fid}: {err[:50]}...')
else:
    print('No checkpoint found - transcription not started')
"
```

---

## Troubleshooting

### Docker Build Fails

**Python package installation errors:**
```bash
# Clean build cache and retry
docker compose build --no-cache
```

**Disk space issues:**
```bash
# Check available space
df -h

# Clean Docker resources
docker system prune -a
```

### GPU Not Detected

**Verify NVIDIA runtime:**
```bash
# Check if GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Check Docker Compose GPU config:**
```bash
# Verify GPU reservation in docker-compose.yml
docker compose config | grep -A 10 "deploy:"
```

**Fall back to CPU mode:**
```bash
docker compose --profile cpu up gtranscriber-cpu
```

### Out of Memory (OOM) Errors

**Reduce workers:**
```bash
WORKERS=1 docker compose --profile gpu up gtranscriber
```

**Enable quantization:**
```bash
GTRANSCRIBER_QUANTIZE=true docker compose --profile gpu up gtranscriber
```

**Use smaller model:**
```bash
GTRANSCRIBER_MODEL_ID=distil-whisper/distil-large-v3 docker compose --profile gpu up gtranscriber
```

### OAuth Token Expired

**Error message:** `RefreshError` or authentication failure

**Solution:**
```bash
# Stop the container
docker compose stop

# Refresh token locally (outside Docker)
gtranscriber info

# Restart transcription
docker compose --profile gpu up gtranscriber
```

### Shared Memory Issues

**Error:** `RuntimeError: unable to open shared memory object`

**Solution:** Increase shared memory size in `docker-compose.yml`:
```yaml
shm_size: '32gb'  # Increase from default 16gb
```

### Network/Download Timeout

**Pre-download models:**
```bash
# Download model before running transcription
docker compose --profile gpu run --rm gtranscriber python -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
model_id = 'openai/whisper-large-v3-turbo'
AutoProcessor.from_pretrained(model_id)
AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
print('Model downloaded successfully')
"
```

### Resume After Interruption

The checkpoint system automatically handles resume. Simply restart:

```bash
docker compose --profile gpu up gtranscriber
```

To start fresh:
```bash
rm results/checkpoint.json
docker compose --profile gpu up gtranscriber
```

---

## Quick Reference

### Common Commands

```bash
# Build image
docker compose --profile gpu build gtranscriber

# Run with GPU
docker compose --profile gpu up gtranscriber

# Run with CPU
docker compose --profile cpu up gtranscriber-cpu

# Run in background
docker compose --profile gpu up -d gtranscriber

# View logs
docker compose logs -f gtranscriber

# Stop
docker compose stop

# Clean up
docker compose down
```

### Example: Full Local Workflow

```bash
# 1. Setup
cp .env.example .env
# Edit .env as needed

# 2. Build
docker compose --profile gpu build gtranscriber

# 3. Run transcription
docker compose --profile gpu up gtranscriber

# 4. Check results
ls results/*_transcription.json | wc -l

# 5. Clean up
docker compose down
```

### Example: Quick Test Run

Test with a small subset of files:

```bash
# Create a test catalog with 5 files
head -6 input/catalog.csv > input/test_catalog.csv

# Run test
CATALOG_FILE=test_catalog.csv WORKERS=1 docker compose --profile gpu up gtranscriber

# Check results
ls results/
```
