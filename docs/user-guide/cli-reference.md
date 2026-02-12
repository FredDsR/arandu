# CLI Commands Reference

Complete reference for all command-line interface commands in G-Transcriber.

## Table of Contents

1. [Command Overview](#command-overview)
2. [Transcription Commands](#transcription-commands)
3. [QA Generation Commands](#qa-generation-commands)
4. [Utilities Commands](#utilities-commands)
5. [Usage Examples](#usage-examples)
6. [Common Patterns](#common-patterns)

---

## Command Overview

The G-Transcriber CLI is built with [Typer](https://typer.tiangolo.com/) and provides rich terminal output using [Rich](https://rich.readthedocs.io/).

**Base Command**: `gtranscriber`

**Command Categories**:
- **Transcription**: `transcribe`, `drive-transcribe`, `batch-transcribe`
- **QA Generation**: `generate-cep-qa`
- **Utilities**: `refresh-auth`, `info`, `list-runs`, `run-info`, `validate-transcriptions`, `rebuild-index`

**Global Options**:
- `--help` - Show command help
- `--version` - Show application version

---

## Transcription Commands

### `transcribe`

Transcribe a local audio or video file.

**Usage**:
```bash
gtranscriber transcribe FILE_PATH [OPTIONS]
```

**Arguments**:
- `FILE_PATH` - Path to the audio/video file to transcribe

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model-id` | `-m` | str | `openai/whisper-large-v3` | Hugging Face model ID for transcription |
| `--output` | `-o` | Path | Auto-generated | Output file path for transcription JSON |
| `--quantize` | `-q` | flag | `False` | Enable 8-bit quantization to reduce VRAM usage |
| `--cpu` | | flag | `False` | Force CPU execution (disables CUDA/MPS) |
| `--language` | `-l` | str | Auto-detect | Language code (e.g., 'pt' for Portuguese) |

**Examples**:
```bash
# Basic transcription
gtranscriber transcribe audio.mp3

# With custom model
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3

# With quantization (reduced VRAM)
gtranscriber transcribe audio.mp3 --quantize

# Force CPU execution
gtranscriber transcribe audio.mp3 --cpu

# Specify language
gtranscriber transcribe audio.mp3 --language pt

# Custom output location
gtranscriber transcribe audio.mp3 -o results/transcription.json
```

---

### `drive-transcribe`

Transcribe a file from Google Drive. Downloads the file, transcribes it, and uploads the result to the same Drive folder.

**Usage**:
```bash
gtranscriber drive-transcribe FILE_ID [OPTIONS]
```

**Arguments**:
- `FILE_ID` - Google Drive file ID to transcribe

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model-id` | `-m` | str | `openai/whisper-large-v3` | Hugging Face model ID |
| `--credentials` | `-c` | Path | `credentials.json` | Path to Google OAuth2 credentials file |
| `--token` | `-t` | Path | `token.json` | Path to Google OAuth2 token file |
| `--quantize` | `-q` | flag | `False` | Enable 8-bit quantization |
| `--cpu` | | flag | `False` | Force CPU execution |
| `--language` | `-l` | str | Auto-detect | Language code |

**Examples**:
```bash
# Basic usage
gtranscriber drive-transcribe 1abc123xyz --credentials credentials.json

# With custom model and quantization
gtranscriber drive-transcribe 1abc123xyz --model-id openai/whisper-large-v3 --quantize
```

---

### `batch-transcribe`

Batch transcribe audio/video files from a catalog CSV with parallel processing and automatic checkpoint/resume capability.

**Usage**:
```bash
gtranscriber batch-transcribe CATALOG_FILE [OPTIONS]
```

**Arguments**:
- `CATALOG_FILE` - Path to catalog CSV file with Google Drive file metadata

**Required CSV Columns**:
- `gdrive_id` - Google Drive file ID
- `name` - File name
- `mime_type` - MIME type
- `size_bytes` - File size in bytes
- `parents` - Parent folder IDs
- `web_content_link` - Download link
- `duration_milliseconds` (optional) - Media duration

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output-dir` | `-o` | Path | `./results` | Output directory for transcription JSON files |
| `--model-id` | `-m` | str | `openai/whisper-large-v3` | Hugging Face model ID |
| `--credentials` | `-c` | Path | `credentials.json` | Path to Google OAuth2 credentials file |
| `--token` | `-t` | Path | `token.json` | Path to Google OAuth2 token file |
| `--workers` | `-w` | int | `1` | Number of parallel workers |
| `--checkpoint` | | Path | `results/checkpoint.json` | Path to checkpoint file |
| `--quantize` | `-q` | flag | `False` | Enable 8-bit quantization |
| `--cpu` | | flag | `False` | Force CPU execution |
| `--language` | `-l` | str | Auto-detect | Language code |
| `--id` | | str | Auto-generated | Pipeline ID for grouping related steps |

**Examples**:
```bash
# Basic batch transcription
gtranscriber batch-transcribe input/catalog.csv --workers 4

# With custom output directory
gtranscriber batch-transcribe input/catalog.csv -o transcriptions/ --workers 2

# With quantization and custom model
gtranscriber batch-transcribe input/catalog.csv \
    --model-id openai/whisper-large-v3 \
    --quantize \
    --workers 4

# Resume interrupted job (uses checkpoint automatically)
gtranscriber batch-transcribe input/catalog.csv --workers 4

# With custom pipeline ID
gtranscriber batch-transcribe input/catalog.csv --id my-project-001
```

---

## QA Generation Commands

### `generate-cep-qa`

Generate CEP (Cognitive Elicitation Pipeline) QA pairs from transcriptions with Bloom-level scaffolding and LLM-as-a-Judge validation.

**Usage**:
```bash
gtranscriber generate-cep-qa INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing transcription JSON files

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output-dir` | `-o` | Path | `qa_dataset` | Output directory for CEP QA dataset JSON files |
| `--provider` | | str | `ollama` | LLM provider: openai, ollama, custom |
| `--model-id` | `-m` | str | `qwen3:14b` | Model ID for QA generation |
| `--workers` | `-w` | int | `2` | Number of parallel workers |
| `--questions` | | int | `10` | Number of QA pairs per document (1-50) |
| `--temperature` | | float | `0.7` | LLM temperature for generation (0.0-2.0) |
| `--ollama-url` | | str | `http://localhost:11434/v1` | Ollama API base URL |
| `--base-url` | | str | `None` | Custom base URL for OpenAI-compatible endpoints |
| `--language` | `-l` | str | `pt` | Language for prompts: 'pt' or 'en' |
| `--validate/--no-validate` | | flag | `True` | Enable LLM-as-a-Judge validation |
| `--validator-model` | | str | `qwen3:14b` | Model ID for validation |
| `--bloom-dist` | | str | `None` | Bloom level distribution (e.g., 'remember:0.2,understand:0.3') |
| `--jsonl/--no-jsonl` | | flag | `False` | Export QA pairs to JSONL format for training |
| `--id` | | str | Auto-resolved | Pipeline ID (auto-resolves transcription outputs) |

**Examples**:
```bash
# Basic usage with Ollama
gtranscriber generate-cep-qa results/ -o qa_dataset/ --workers 4

# With custom Bloom distribution
gtranscriber generate-cep-qa results/ \
    --bloom-dist "remember:0.2,understand:0.3,analyze:0.3,evaluate:0.2" \
    --questions 15

# With OpenAI
gtranscriber generate-cep-qa results/ \
    --provider openai \
    --model-id gpt-4o-mini \
    --workers 2

# Without validation (faster)
gtranscriber generate-cep-qa results/ \
    --no-validate \
    --workers 4

# With custom validator model
gtranscriber generate-cep-qa results/ \
    --validator-model qwen3:14b \
    --questions 12

# Export to JSONL for KGQA training
gtranscriber generate-cep-qa results/ \
    --jsonl \
    --questions 20

# English prompts
gtranscriber generate-cep-qa results/ \
    --language en \
    --questions 10

# With pipeline ID
gtranscriber generate-cep-qa results/ --id my-project-001
```

**Output Structure**:
```
qa_dataset/
├── cep_qa_1abc123xyz.json
├── cep_qa_2def456uvw.json
└── cep_qa_checkpoint.json
```

---

## Utilities Commands

### `refresh-auth`

Fully refresh Google OAuth2 authentication token. Deletes existing token and initiates fresh OAuth2 authorization flow.

**Usage**:
```bash
gtranscriber refresh-auth [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--credentials` | `-c` | Path | `credentials.json` | Path to Google OAuth2 credentials file |
| `--token` | `-t` | Path | `token.json` | Path to token file to refresh |

**Example**:
```bash
gtranscriber refresh-auth --credentials credentials.json --token token.json
```

---

### `info`

Display system information and hardware capabilities.

**Usage**:
```bash
gtranscriber info
```

**Output**:
- Application version
- Device type (CPU/CUDA/MPS)
- CUDA/MPS availability
- PyTorch version and configuration
- GPU memory information (if available)

**Example**:
```bash
gtranscriber info
```

---

### `list-runs`

List all pipeline runs with status and metadata.

**Usage**:
```bash
gtranscriber list-runs [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--pipeline` | `-p` | str | All pipelines | Filter by pipeline type: transcription, qa, cep, kg, evaluation |
| `--results-dir` | `-r` | Path | `./results` | Base results directory |

**Examples**:
```bash
# List all runs
gtranscriber list-runs

# Filter by pipeline type
gtranscriber list-runs --pipeline transcription

# Custom results directory
gtranscriber list-runs --results-dir /path/to/results
```

---

### `run-info`

Display detailed information about a specific run including execution environment, hardware info, configuration, and processing statistics.

**Usage**:
```bash
gtranscriber run-info RUN_ID [OPTIONS]
```

**Arguments**:
- `RUN_ID` - Run ID to display, or "latest" for the most recent run

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--pipeline` | `-p` | str | `transcription` | Pipeline type (required when using "latest") |
| `--results-dir` | `-r` | Path | `./results` | Base results directory |

**Examples**:
```bash
# Display specific run
gtranscriber run-info transcription_20260211_143022

# Display latest transcription run
gtranscriber run-info latest --pipeline transcription

# Display latest CEP run
gtranscriber run-info latest --pipeline cep
```

---

### `validate-transcriptions`

Validate existing transcriptions for quality issues (wrong language/script, repeated words, suspicious patterns, empty content).

**Usage**:
```bash
gtranscriber validate-transcriptions INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing transcription JSON files to validate

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output-dir` | `-o` | Path | In-place update | Directory to save validated results |
| `--threshold` | `-t` | float | `0.5` | Quality threshold (0.0-1.0) for marking as valid |
| `--language` | `-l` | str | `pt` | Expected language code (e.g., 'pt', 'en') |
| `--report-only` | | flag | `False` | Only display report without updating files |

**Examples**:
```bash
# Validate and update in-place
gtranscriber validate-transcriptions results/

# Validate with custom threshold
gtranscriber validate-transcriptions results/ --threshold 0.6

# Validate English transcriptions
gtranscriber validate-transcriptions results/ --language en

# Report only (no file updates)
gtranscriber validate-transcriptions results/ --report-only

# Save validated files to new directory
gtranscriber validate-transcriptions results/ --output-dir validated/
```

**Quality Checks**:
- Script match (Latin characters for pt/en)
- Repetition detection (words and phrases)
- Segment quality (natural timestamps)
- Content density (words per minute)

---

### `rebuild-index`

Rebuild index.json from existing run directories by scanning all pipeline ID directories for run_metadata.json files.

**Usage**:
```bash
gtranscriber rebuild-index [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--results-dir` | `-r` | Path | `./results` | Base results directory |

**Example**:
```bash
gtranscriber rebuild-index --results-dir /path/to/results
```

---

## Usage Examples

### End-to-End Pipeline

Complete pipeline from transcription to CEP QA generation:

```bash
# Step 1: Batch transcribe files
gtranscriber batch-transcribe input/catalog.csv \
    --workers 4 \
    --quantize \
    --id etno-001

# Step 2: Validate transcription quality
gtranscriber validate-transcriptions results/ \
    --threshold 0.6

# Step 3: Generate CEP QA pairs
gtranscriber generate-cep-qa results/ \
    --workers 4 \
    --questions 12 \
    --language pt \
    --id etno-001

# Step 4: List all runs
gtranscriber list-runs

# Step 5: View run details
gtranscriber run-info latest --pipeline cep
```

### Resume Interrupted Job

All batch commands support automatic checkpointing:

```bash
# Start batch transcription
gtranscriber batch-transcribe input/catalog.csv --workers 4

# If interrupted, resume automatically
gtranscriber batch-transcribe input/catalog.csv --workers 4
# Will skip already processed files

# CEP generation also supports resume
gtranscriber generate-cep-qa results/ --workers 4
```

### Custom LLM Configuration

Use different LLM providers and models:

```bash
# With Ollama (default)
gtranscriber generate-cep-qa results/ \
    --provider ollama \
    --model-id qwen3:14b \
    --workers 4

# With OpenAI
export OPENAI_API_KEY=sk-...
gtranscriber generate-cep-qa results/ \
    --provider openai \
    --model-id gpt-4o-mini \
    --workers 2

# With custom OpenAI-compatible endpoint
gtranscriber generate-cep-qa results/ \
    --provider custom \
    --base-url https://my-vllm-server/v1 \
    --model-id llama3.1:70b
```

---

## Common Patterns

### Configuration Override

Command-line arguments override environment variables:

```bash
# Config says ollama, but we override to openai
export GTRANSCRIBER_QA_PROVIDER=ollama
gtranscriber generate-cep-qa results/ --provider openai
```

### Environment Variables

Set defaults via environment instead of CLI:

```bash
# Transcription settings
export GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3
export GTRANSCRIBER_WORKERS=4
export GTRANSCRIBER_QUANTIZE=true

# QA settings
export GTRANSCRIBER_QA_PROVIDER=ollama
export GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=12

# CEP settings
export GTRANSCRIBER_CEP_ENABLE_VALIDATION=true
export GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=qwen3:14b

# Now run with defaults
gtranscriber batch-transcribe input/catalog.csv
gtranscriber generate-cep-qa results/
```

### Pipeline ID Tracking

Use consistent pipeline IDs across related steps:

```bash
# All steps use same pipeline ID
PIPELINE_ID="etno-project-001"

gtranscriber batch-transcribe input/catalog.csv --id $PIPELINE_ID
gtranscriber validate-transcriptions results/ 
gtranscriber generate-cep-qa results/ --id $PIPELINE_ID

# View all runs for this pipeline
gtranscriber list-runs
gtranscriber run-info $PIPELINE_ID
```

---

## Error Handling

### Common Errors

**LLM Provider Not Available**:
```bash
Error: Ollama server not reachable at http://localhost:11434
Solution: Start Ollama with 'ollama serve'
```

**API Key Missing**:
```bash
Error: OPENAI_API_KEY environment variable not set
Solution: export OPENAI_API_KEY=sk-...
```

**Input Directory Empty**:
```bash
Error: No transcription files found in results/
Solution: Check directory path and ensure files have .json extension
```

**Checkpoint Corruption**:
```bash
Error: Checkpoint file corrupted
Solution: Delete checkpoint.json and restart the command
```

---

## Tips and Best Practices

### 1. Start Small

Test on sample data before processing full corpus:

```bash
# Test on 5 files first
mkdir samples && cp results/*.json samples/ | head -5
gtranscriber generate-cep-qa samples/ -o qa_test/
```

### 2. Use Checkpoints

Checkpoints enable automatic resume:

```bash
# Runs create checkpoints automatically
gtranscriber batch-transcribe input/catalog.csv --workers 4
# Creates: results/checkpoint.json

# Resume automatically if interrupted
gtranscriber batch-transcribe input/catalog.csv --workers 4
# Skips already processed files
```

### 3. Monitor Progress

Use pipeline tracking commands:

```bash
# List all runs
gtranscriber list-runs

# View latest run details
gtranscriber run-info latest --pipeline transcription

# Check run statistics
gtranscriber run-info latest --pipeline cep
```

### 4. Optimize Workers

Adjust workers based on available resources:

```bash
# CPU-bound tasks: Use available cores
gtranscriber generate-cep-qa results/ --workers $(nproc)

# Memory-constrained: Reduce workers
gtranscriber batch-transcribe catalog.csv --workers 2
```

### 5. Validate Quality

Always validate transcriptions before downstream tasks:

```bash
# Validate first
gtranscriber validate-transcriptions results/ --threshold 0.6

# Then generate QA
gtranscriber generate-cep-qa results/ --workers 4
```

---

**Document Version**: 2.0  
**Last Updated**: 2026-02-11  
**Status**: Aligned with codebase v0.1.0
