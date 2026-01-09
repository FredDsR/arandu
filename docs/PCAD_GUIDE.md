# Running G-Transcriber on PCAD

This guide covers the complete workflow for running the transcription process on the PCAD (Parque Computacional de Alto Desempenho) at UFRGS.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Syncing to PCAD](#syncing-to-pcad)
4. [Submitting Jobs](#submitting-jobs)
5. [Monitoring Jobs](#monitoring-jobs)
6. [Gathering Results](#gathering-results)
7. [Storage: $HOME vs $SCRATCH](#storage-home-vs-scratch)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Machine

- Valid Google OAuth2 credentials (`credentials.json` and `token.json`)
- SSH access to PCAD (`gppd-hpc.inf.ufrgs.br`)
- `rsync` and `scp` installed

### PCAD Account

- Active PCAD user account
- Familiarity with SLURM job scheduler

### Refresh OAuth Token

Before syncing to PCAD, ensure your Google OAuth token is fresh. Run locally:

```bash
gtranscriber info
```

This will refresh the token if needed. If the token is expired, you'll be prompted to re-authenticate via browser.

---

## Project Setup

### 1. Configure Environment (Optional)

Copy the example environment file and customize:

```bash
cp .env.example .env
```

Edit `.env` to adjust settings:

```bash
# Model selection
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3-turbo

# Number of parallel workers
WORKERS=4

# Input catalog file
CATALOG_FILE=catalog.csv
```

### 2. Verify Input Catalog

Ensure your catalog file exists in the `input/` directory:

```bash
ls -la input/catalog.csv
```

The catalog should contain columns: `gdrive_id`, `name`, `mime_type`, `size_bytes`, `parents`, `web_content_link`.

---

## Syncing to PCAD

### Initial Sync

Sync the entire project to your PCAD home directory:

```bash
rsync -avz --progress \
    --exclude '.venv' \
    --exclude 'results/*' \
    --exclude 'cache/*' \
    --exclude 'logs/*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    . user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/
```

### Sync Credentials Separately

For security, sync credentials files separately:

```bash
scp credentials.json token.json user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/
```

### Update Sync (After Code Changes)

To sync only changed files:

```bash
rsync -avz --progress \
    --exclude '.venv' \
    --exclude 'results/*' \
    --exclude 'cache/*' \
    --exclude 'logs/*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'credentials.json' \
    --exclude 'token.json' \
    . user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/
```

---

## Submitting Jobs

### Available Partitions

#### NVIDIA GPU Partitions (Recommended)

| Partition | GPU | VRAM | Recommended Workers | Best For |
|-----------|-----|------|---------------------|----------|
| `grace` | NVIDIA L40S | 46 GB | 6-8 | Large batches, full model |
| `tupi` | RTX 4090 | 24 GB | 4 | Good balance |
| `blaise` | Tesla P100 | 16 GB | 2-3 | Medium batches |
| `draco` | Tesla K20m | 4 GB | 1-2 | Small batches, distil model |

#### AMD GPU Partition (ROCm)

| Partition | GPU | VRAM | Recommended Workers | Notes |
|-----------|-----|------|---------------------|-------|
| `sirius` | AMD Radeon RX 7900 XT/XTX | 21 GB | 3 | Uses ROCm, no quantization |

> **Note:** The `sirius` partition uses AMD ROCm for GPU acceleration. Quantization is not available on AMD GPUs (bitsandbytes doesn't support ROCm), so fewer workers are recommended compared to NVIDIA with quantization.

#### CPU-Only Partition

| Partition | Accelerator | Memory | Mode | Notes |
|-----------|-------------|--------|------|-------|
| `turing` | NEC TSUBASA Vector Engine | 48 GB | CPU | Vector Engine not compatible with PyTorch |

> **Note:** The `turing` partition uses NEC Vector Engines which lack PyTorch/Transformers support. The script automatically runs in CPU mode.

### Submit a Job

Connect to PCAD and submit:

```bash
ssh user@gppd-hpc.inf.ufrgs.br
cd ~/etno-kgc-preprocessing

# Submit to tupi partition (RTX 4090) - recommended
sbatch scripts/slurm/tupi.slurm

# Submit to grace partition (L40S - largest VRAM)
sbatch scripts/slurm/grace.slurm

# Submit to blaise partition (Tesla P100)
sbatch scripts/slurm/blaise.slurm

# Submit to draco partition (Tesla K20m - limited VRAM)
sbatch scripts/slurm/draco.slurm

# Submit to sirius partition (AMD GPU with ROCm)
sbatch scripts/slurm/sirius.slurm

# Submit to turing partition (CPU mode)
sbatch scripts/slurm/turing.slurm
```

### Override Default Settings

You can override settings when submitting:

```bash
# Use more workers
WORKERS=6 sbatch scripts/slurm/tupi.slurm

# Use a different model
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3 sbatch scripts/slurm/grace.slurm

# Use a different catalog
CATALOG_FILE=my_subset.csv sbatch scripts/slurm/tupi.slurm

# Force CPU mode
USE_CPU=true sbatch scripts/slurm/draco.slurm
```

### Submit to Custom Partition

For partitions without a dedicated script:

```bash
sbatch --partition=beagle scripts/slurm/tupi.slurm
```

---

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View all jobs on a partition
squeue -p tupi

# Detailed job info
scontrol show job <job_id>
```

### View Job Output

```bash
# Real-time output
tail -f logs/gtranscriber_<job_id>.out

# View errors
tail -f logs/gtranscriber_<job_id>.err

# View last 100 lines
tail -100 logs/gtranscriber_<job_id>.out
```

### Check Progress

The transcription process saves checkpoints. To see progress:

```bash
# View checkpoint file
cat results/checkpoint.json | python -m json.tool

# Count completed transcriptions
ls -1 results/*_transcription.json | wc -l
```

### Cancel a Job

```bash
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Gathering Results

### Download Results to Local Machine

From your local machine:

```bash
# Download all results
rsync -avz --progress \
    user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/results/ \
    ./results/

# Download only transcription JSON files
rsync -avz --progress \
    --include='*_transcription.json' \
    --exclude='*' \
    user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/results/ \
    ./results/
```

### Download Logs

```bash
rsync -avz --progress \
    user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/logs/ \
    ./logs/
```

### Verify Results

Check the number of completed transcriptions:

```bash
# Count result files
ls -1 results/*_transcription.json | wc -l

# Check for any failed files in checkpoint
python -c "
import json
with open('results/checkpoint.json') as f:
    cp = json.load(f)
print(f'Completed: {len(cp.get(\"completed_files\", []))}')
print(f'Failed: {len(cp.get(\"failed_files\", {}))}')
"
```

---

## Storage: $HOME vs $SCRATCH

PCAD provides two storage locations with different characteristics:

| Directory | Location | Performance | Persistence |
|-----------|----------|-------------|-------------|
| `$HOME` | NFS (network) | Slower | Permanent (no backup) |
| `$SCRATCH` | Local disk | **Faster** | **Temporary** - can be deleted anytime |

### Automatic $SCRATCH Optimization

The SLURM scripts **automatically use `$SCRATCH`** when available for better I/O performance:

1. **At job start**: Input files, credentials, and HF cache are copied to `$SCRATCH`
2. **During execution**: Results are written to `$SCRATCH` (fast local disk)
3. **At job end**: Results are automatically copied back to `$HOME`

This happens transparently - your final results will always be in `$HOME/etno-kgc-preprocessing/results/`.

### Disabling $SCRATCH Optimization

If you encounter issues with `$SCRATCH`, you can disable it:

```bash
USE_SCRATCH=false sbatch scripts/slurm/tupi.slurm
```

### Important Notes

- **Never store important data only in `$SCRATCH`** - it can be deleted without notice
- Results are automatically copied to `$HOME` even if the job fails (via cleanup trap)
- The HF model cache is synced back to `$HOME` so models don't need re-downloading
- If `$SCRATCH` is unavailable, the script automatically falls back to `$HOME`

---

## Troubleshooting

### Job Fails Immediately

**Check the error log:**
```bash
cat logs/gtranscriber_<job_id>.err
```

**Common causes:**
- Missing credentials files
- Docker not available on partition
- Insufficient disk quota

### Out of Memory Errors

**Reduce workers or enable quantization:**
```bash
WORKERS=2 sbatch scripts/slurm/tupi.slurm
```

**Use a smaller model:**
```bash
GTRANSCRIBER_MODEL_ID=distil-whisper/distil-large-v3 sbatch scripts/slurm/draco.slurm
```

### OAuth Token Expired

If you see authentication errors:

1. Return to your local machine
2. Run `gtranscriber info` to refresh the token
3. Re-sync the token file:
   ```bash
   scp token.json user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/
   ```

### Docker Build Fails

**Check if Docker is available:**
```bash
ssh user@gppd-hpc.inf.ufrgs.br
docker --version
docker compose version
```

**Clear Docker cache and rebuild:**
```bash
docker system prune -f
docker compose build --no-cache gtranscriber
```

### Network/Download Issues

PCAD may have network restrictions. If downloads fail:

1. Pre-download Hugging Face models locally
2. Sync the cache to PCAD:
   ```bash
   rsync -avz --progress cache/huggingface/ \
       user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/cache/huggingface/
   ```

### Resume After Interruption

The checkpoint system automatically resumes from the last completed file. Simply resubmit the job:

```bash
sbatch scripts/slurm/tupi.slurm
```

To start fresh, remove the checkpoint:
```bash
rm results/checkpoint.json
```

### ROCm Issues (Sirius Partition)

**AMD GPU not detected:**
```bash
# Check if ROCm device is available
ls -la /dev/kfd /dev/dri
```

**ROCm build fails:**
The ROCm Docker image is larger than the NVIDIA image. Ensure sufficient disk space:
```bash
df -h
docker system prune -f
```

**Performance slower than expected:**
- AMD GPUs are typically ~2x slower than equivalent NVIDIA GPUs for inference
- Quantization is not available on ROCm (no bitsandbytes support)
- Consider using fewer workers: `WORKERS=2 sbatch scripts/slurm/sirius.slurm`

---

## Quick Reference

### One-liner: Full Workflow

```bash
# 1. Sync project
rsync -avz --exclude '.venv' --exclude 'results/*' --exclude 'cache/*' \
    . user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/

# 2. Sync credentials
scp credentials.json token.json user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/

# 3. Submit job
ssh user@gppd-hpc.inf.ufrgs.br "cd ~/etno-kgc-preprocessing && sbatch scripts/slurm/tupi.slurm"

# 4. Monitor (in another terminal)
ssh user@gppd-hpc.inf.ufrgs.br "tail -f ~/etno-kgc-preprocessing/logs/gtranscriber_*.out"

# 5. Download results (after completion)
rsync -avz user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/results/ ./results/
```

### Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias pcad="ssh user@gppd-hpc.inf.ufrgs.br"
alias pcad-sync="rsync -avz --exclude '.venv' --exclude 'results/*' --exclude 'cache/*' . user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/"
alias pcad-results="rsync -avz user@gppd-hpc.inf.ufrgs.br:~/etno-kgc-preprocessing/results/ ./results/"
```

---

## Acknowledgment

Research using PCAD resources should include:

> Some experiments in this work used the PCAD infrastructure, http://gppd-hpc.inf.ufrgs.br, at INF/UFRGS.
