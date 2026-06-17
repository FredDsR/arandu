# CLI Commands Reference

Complete reference for all command-line interface commands in Arandu.

## Table of Contents

1. [Command Overview](#command-overview)
2. [Transcription Commands](#transcription-commands)
3. [QA Generation Commands](#qa-generation-commands)
4. [Judging Commands](#judging-commands)
5. [Knowledge Graph & RAG Commands (Phase C)](#knowledge-graph--rag-commands-phase-c)
6. [Utilities Commands](#utilities-commands)
7. [Usage Examples](#usage-examples)
8. [Common Patterns](#common-patterns)

---

## Command Overview

The Arandu CLI is built with [Typer](https://typer.tiangolo.com/) and provides rich terminal output using [Rich](https://rich.readthedocs.io/).

**Base Command**: `arandu`

**Command Categories**:
- **Transcription**: `transcribe`, `drive-transcribe`, `batch-transcribe`
- **QA Generation**: `generate-cep-qa`, `generate-non-answerable`
- **Judging**: `judge-transcription`, `judge-qa`, `judge-answers`
- **Knowledge Graph & RAG (Phase C)**: `chunk`, `build-kg`, `kg-link-passages`, `kg-build-retriever-index`, `retrieve`, `answer`, `emic-prepass`, `build-human-eval-sample`, `rag-analysis`
- **Utilities**: `refresh-auth`, `enrich-metadata`, `replicate`, `info`, `list-runs`, `run-info`, `rebuild-index`, `report`, `serve-report`

**Global Options**:
- `--help` - Show command help
- `--version` - Show application version

---

## Transcription Commands

### `transcribe`

Transcribe a local audio or video file.

**Usage**:
```bash
arandu transcribe FILE_PATH [OPTIONS]
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
arandu transcribe audio.mp3

# With custom model
arandu transcribe audio.mp3 --model-id openai/whisper-large-v3

# With quantization (reduced VRAM)
arandu transcribe audio.mp3 --quantize

# Force CPU execution
arandu transcribe audio.mp3 --cpu

# Specify language
arandu transcribe audio.mp3 --language pt

# Custom output location
arandu transcribe audio.mp3 -o results/transcription.json
```

---

### `drive-transcribe`

Transcribe a file from Google Drive. Downloads the file, transcribes it, and uploads the result to the same Drive folder.

**Usage**:
```bash
arandu drive-transcribe FILE_ID [OPTIONS]
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
arandu drive-transcribe 1abc123xyz --credentials credentials.json

# With custom model and quantization
arandu drive-transcribe 1abc123xyz --model-id openai/whisper-large-v3 --quantize
```

---

### `batch-transcribe`

Batch transcribe audio/video files from a catalog CSV with parallel processing and automatic checkpoint/resume capability.

**Usage**:
```bash
arandu batch-transcribe CATALOG_FILE [OPTIONS]
```

**Arguments**:
- `CATALOG_FILE` - Path to catalog CSV file with Google Drive file metadata

**Required CSV Columns**:
- `file_id` - Google Drive file ID
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
arandu batch-transcribe input/catalog.csv --workers 4

# With custom output directory
arandu batch-transcribe input/catalog.csv -o transcriptions/ --workers 2

# With quantization and custom model
arandu batch-transcribe input/catalog.csv \
    --model-id openai/whisper-large-v3 \
    --quantize \
    --workers 4

# Resume interrupted job (uses checkpoint automatically)
arandu batch-transcribe input/catalog.csv --workers 4

# With custom pipeline ID
arandu batch-transcribe input/catalog.csv --id my-project-001
```

---

## QA Generation Commands

### `generate-cep-qa`

Generate CEP (Cognitive Elicitation Pipeline) QA pairs from transcriptions with Bloom-level scaffolding. Generation and validation are now separate steps: this command only generates pairs. Run [`judge-qa`](#judge-qa) afterwards to evaluate them with LLM-as-a-Judge.

**Usage**:
```bash
arandu generate-cep-qa INPUT_DIR [OPTIONS]
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
| `--temperature` | | float | `0.7` | LLM temperature for generation (0.0-2.0) |
| `--ollama-url` | | str | `http://localhost:11434/v1` | Ollama API base URL |
| `--base-url` | | str | `None` | Custom base URL for OpenAI-compatible endpoints |
| `--language` | `-l` | str | `pt` | Language for prompts: 'pt' or 'en' |
| `--bloom-dist` | | str | `None` | Bloom level pair counts (e.g., 'remember:3,understand:1,analyze:1,evaluate:1') |
| `--jsonl/--no-jsonl` | | flag | `False` | Export QA pairs to JSONL format for training |
| `--id` | | str | Auto-resolved | Pipeline ID (auto-resolves transcription outputs) |

**Examples**:
```bash
# Basic usage with Ollama
arandu generate-cep-qa results/ -o qa_dataset/ --workers 4

# With custom Bloom distribution (integer pair counts per level)
arandu generate-cep-qa results/ \
    --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1"

# With OpenAI
arandu generate-cep-qa results/ \
    --provider openai \
    --model-id gpt-4o-mini \
    --workers 2

# Generate, then validate as a separate step
arandu generate-cep-qa results/ -o qa_dataset/ --workers 4
arandu judge-qa qa_dataset/ --model qwen3:14b

# Export to JSONL for KGQA training
arandu generate-cep-qa results/ \
    --jsonl

# English prompts
arandu generate-cep-qa results/ \
    --language en

# With pipeline ID
arandu generate-cep-qa results/ --id my-project-001
```

**Output Structure**:
```
qa_dataset/
├── 1abc123xyz_cep_qa.json
├── 2def456uvw_cep_qa.json
└── cep_qa_checkpoint.json
```

---

### `generate-non-answerable`

Generate a non-answerable benchmark from validated CEP pairs and the knowledge graph. Used to measure whether the RAG arms correctly abstain on questions the corpus cannot answer.

**Usage**:
```bash
arandu generate-non-answerable INPUT_DIR [OPTIONS]
```

Run `arandu generate-non-answerable --help` for the full option list.

---

## Judging Commands

The judging commands share the `ARANDU_JUDGE_VALIDATOR_*` environment fallbacks (model, provider, base URL) and the `--rejudge/--resume` resume semantics.

### `judge-transcription`

Judge transcription quality with a two-stage filter pipeline: pure-Python heuristics (always) plus optional LLM criteria. A record passes only when it clears every criterion in each filter stage; there is no aggregate score. See the [Transcription Validation guide](transcription-validation.md) for the full model.

**Usage**:
```bash
arandu judge-transcription INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing `*_transcription.json` files to judge

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--language` | `-l` | str | `pt` | Expected language code (e.g., 'pt', 'en') |
| `--validator-model` | | str | (none) | Model ID enabling the LLM filter stage. Falls back to `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `--validator-provider` | | str | inferred | `openai`, `ollama`, or `custom`. Falls back to `ARANDU_JUDGE_VALIDATOR_PROVIDER` |
| `--validator-base-url` | | str | inferred | Validator base URL. Falls back to `ARANDU_JUDGE_VALIDATOR_BASE_URL`, then `ARANDU_LLM_BASE_URL` |
| `--validator-temperature` | | float | `0.3` | Sampling temperature for LLM criteria. Falls back to `ARANDU_JUDGE_TEMPERATURE` |
| `--validator-max-tokens` | | int | `2048` | Max tokens for LLM criterion responses. Falls back to `ARANDU_JUDGE_MAX_TOKENS` |
| `--rejudge` / `--resume` | | flag | `--resume` | `--rejudge` re-evaluates every record; `--resume` skips already-judged records |

**Examples**:
```bash
# Heuristics only, update in-place
arandu judge-transcription results/

# Enable the LLM filter stage (language_drift + hallucination_loop)
arandu judge-transcription results/ --validator-model qwen3:14b

# Judge English transcriptions
arandu judge-transcription results/ --language en

# Force a fresh pass over every record
arandu judge-transcription results/ --validator-model qwen3:14b --rejudge
```

**Stages**:
- Heuristic filter (always): content length floor, script match, repetition, content density, segment quality
- LLM filter (optional): `language_drift`, `hallucination_loop`

---

### `judge-qa`

Judge CEP QA pairs with LLM-as-a-Judge across four criteria (faithfulness, Bloom calibration, informativeness, self-containedness). Each judged pair is persisted back into its `*_cep_qa.json` file via the pair's `validation` field; `is_valid` is derived from `validation.passed`. There is no aggregate side-file. This is the validation step that used to be folded into `generate-cep-qa`.

**Usage**:
```bash
arandu judge-qa INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing `*_cep_qa.json` files to judge

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model` | `-m` | str | (none) | Judge model ID. Required via flag or `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `--provider` | | str | inferred | `openai`, `ollama`, or `custom`. Falls back to `ARANDU_JUDGE_VALIDATOR_PROVIDER` |
| `--base-url` | | str | inferred | Custom/OpenAI-compatible URL. Falls back to `ARANDU_JUDGE_VALIDATOR_BASE_URL`, then `ARANDU_LLM_BASE_URL` |
| `--language` | `-l` | str | `pt` | Language for judge prompts: 'pt' or 'en' |
| `--files` | | int | All | Maximum number of QA files to sample |
| `--pairs` | | int | All | Maximum QA pairs to judge per file |
| `--rejudge` / `--resume` | | flag | `--resume` | `--rejudge` re-evaluates every sampled pair; `--resume` skips pairs already carrying a `validation` payload |

**Examples**:
```bash
# Defaults from ARANDU_JUDGE_* env vars
arandu judge-qa qa_dataset/

# Explicit Ollama model
arandu judge-qa qa_dataset/ --provider ollama --model qwen3:14b

# OpenAI-compatible custom endpoint
arandu judge-qa qa_dataset/ --provider custom --model gemini-2.5-flash \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai/

# Sample a subset, then force a fresh pass
arandu judge-qa qa_dataset/ --files 2 --pairs 3
arandu judge-qa qa_dataset/ --rejudge
```

---

### `judge-answers`

Run the gated answer judge over every `AnswerRecord` in a populated Phase C run, scoring generated RAG answers. See the [RAG evaluation docs](rag-analysis.md) for the scoring model.

**Usage**:
```bash
arandu judge-answers RUN_ID [OPTIONS]
```

Run `arandu judge-answers --help` for the full option list.

---

## Knowledge Graph & RAG Commands (Phase C)

These commands implement the Phase C retrieval-augmented-generation evaluation chain: build a knowledge graph from transcriptions, link passages, build retriever indices, retrieve, answer, and analyze. They operate on a populated run identified by a pipeline/run ID. Each command exposes its full option set via `--help`; the summaries below cover the common workflow.

| Command | Description |
|---------|-------------|
| `chunk` | Build `ChunkSets` across one or more chunker views |
| `build-kg` | Build a knowledge graph from transcription records |
| `kg-link-passages` | Map atlas-rag passages back to char offsets in source `EnrichedRecord` space |
| `kg-build-retriever-index` | Build the atlas-rag retriever's precomputed index for a run |
| `retrieve` | Run Phase C retrievers over a populated run |
| `answer` | Run the Answerer LLM over every `RetrievalRecord` in a populated run |
| `emic-prepass` | Score canonical-approved CEP pairs for emic validity (ordinal 1-5) |
| `build-human-eval-sample` | Build the stratified human-comparison sample (80 pairs) for a run |
| `rag-analysis` | Aggregate judged answers and emit `report.json` + `tables.md` |

**Typical Phase C chain**:
```bash
# Build the KG and retriever index for a run
arandu build-kg results/ --id project-001
arandu kg-link-passages project-001
arandu kg-build-retriever-index project-001

# Retrieve, answer, judge, analyze
arandu retrieve project-001
arandu answer project-001
arandu judge-answers project-001
arandu rag-analysis project-001
```

---

## Utilities Commands

### `refresh-auth`

Fully refresh Google OAuth2 authentication token. Deletes existing token and initiates fresh OAuth2 authorization flow.

**Usage**:
```bash
arandu refresh-auth [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--credentials` | `-c` | Path | `credentials.json` | Path to Google OAuth2 credentials file |
| `--token` | `-t` | Path | `token.json` | Path to token file to refresh |

**Example**:
```bash
arandu refresh-auth --credentials credentials.json --token token.json
```

---

### `info`

Display system information and hardware capabilities.

**Usage**:
```bash
arandu info
```

**Output**:
- Application version
- Device type (CPU/CUDA/MPS)
- CUDA/MPS availability
- PyTorch version and configuration
- GPU memory information (if available)

**Example**:
```bash
arandu info
```

---

### `list-runs`

List all pipeline runs with status and metadata.

**Usage**:
```bash
arandu list-runs [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--pipeline` | `-p` | str | All pipelines | Filter by pipeline type: transcription, qa, cep, kg, evaluation |
| `--results-dir` | `-r` | Path | `./results` | Base results directory |

**Examples**:
```bash
# List all runs
arandu list-runs

# Filter by pipeline type
arandu list-runs --pipeline transcription

# Custom results directory
arandu list-runs --results-dir /path/to/results
```

---

### `run-info`

Display detailed information about a specific run including execution environment, hardware info, configuration, and processing statistics.

**Usage**:
```bash
arandu run-info RUN_ID [OPTIONS]
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
arandu run-info transcription_20260211_143022

# Display latest transcription run
arandu run-info latest --pipeline transcription

# Display latest CEP run
arandu run-info latest --pipeline cep
```

---

### `rebuild-index`

Rebuild index.json from existing run directories by scanning all pipeline ID directories for run_metadata.json files.

**Usage**:
```bash
arandu rebuild-index [OPTIONS]
```

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--results-dir` | `-r` | Path | `./results` | Base results directory |

**Example**:
```bash
arandu rebuild-index --results-dir /path/to/results
```

---

### Other utilities

Concise reference for the remaining run-management and reporting commands. Run `arandu <command> --help` for the full option list.

| Command | Description |
|---------|-------------|
| `replicate SOURCE_PIPELINE_ID` | Replicate (clone) an existing pipeline run to a new ID |
| `enrich-metadata INPUT_DIR OUTPUT_DIR` | Enrich existing transcription JSONs with source metadata |
| `report` | Generate interactive HTML dashboard and PNG charts for pipeline results |
| `serve-report RESULTS_DIR` | Launch interactive dashboard for pipeline results exploration |

---

## Usage Examples

### End-to-End Pipeline

Complete pipeline from transcription to CEP QA generation:

```bash
# Step 1: Batch transcribe files
arandu batch-transcribe input/catalog.csv \
    --workers 4 \
    --quantize \
    --id etno-001

# Step 2: Judge transcription quality
arandu judge-transcription results/ \
    --validator-model qwen3:14b

# Step 3: Generate CEP QA pairs
arandu generate-cep-qa results/ \
    --workers 4 \
    --language pt \
    --id etno-001

# Step 4: Judge the generated QA pairs
arandu judge-qa qa_dataset/ --model qwen3:14b

# Step 5: List all runs
arandu list-runs

# Step 6: View run details
arandu run-info latest --pipeline cep
```

### Resume Interrupted Job

All batch commands support automatic checkpointing:

```bash
# Start batch transcription
arandu batch-transcribe input/catalog.csv --workers 4

# If interrupted, resume automatically
arandu batch-transcribe input/catalog.csv --workers 4
# Will skip already processed files

# CEP generation also supports resume
arandu generate-cep-qa results/ --workers 4
```

### Custom LLM Configuration

Use different LLM providers and models:

```bash
# With Ollama (default)
arandu generate-cep-qa results/ \
    --provider ollama \
    --model-id qwen3:14b \
    --workers 4

# With OpenAI
export OPENAI_API_KEY=sk-...
arandu generate-cep-qa results/ \
    --provider openai \
    --model-id gpt-4o-mini \
    --workers 2

# With custom OpenAI-compatible endpoint
arandu generate-cep-qa results/ \
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
export ARANDU_QA_PROVIDER=ollama
arandu generate-cep-qa results/ --provider openai
```

### Environment Variables

Set defaults via environment instead of CLI:

```bash
# Transcription settings
export ARANDU_MODEL_ID=openai/whisper-large-v3
export ARANDU_WORKERS=4
export ARANDU_QUANTIZE=true

# QA settings
export ARANDU_QA_PROVIDER=ollama
export ARANDU_QA_MODEL_ID=qwen3:14b

# Judge settings (used by judge-qa and judge-transcription)
export ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
export ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama

# Now run with defaults
arandu batch-transcribe input/catalog.csv
arandu generate-cep-qa results/
```

### Pipeline ID Tracking

Use consistent pipeline IDs across related steps:

```bash
# All steps use same pipeline ID
PIPELINE_ID="etno-project-001"

arandu batch-transcribe input/catalog.csv --id $PIPELINE_ID
arandu judge-transcription results/
arandu generate-cep-qa results/ --id $PIPELINE_ID

# View all runs for this pipeline
arandu list-runs
arandu run-info $PIPELINE_ID
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
arandu generate-cep-qa samples/ -o qa_test/
```

### 2. Use Checkpoints

Checkpoints enable automatic resume:

```bash
# Runs create checkpoints automatically
arandu batch-transcribe input/catalog.csv --workers 4
# Creates: results/checkpoint.json

# Resume automatically if interrupted
arandu batch-transcribe input/catalog.csv --workers 4
# Skips already processed files
```

### 3. Monitor Progress

Use pipeline tracking commands:

```bash
# List all runs
arandu list-runs

# View latest run details
arandu run-info latest --pipeline transcription

# Check run statistics
arandu run-info latest --pipeline cep
```

### 4. Optimize Workers

Adjust workers based on available resources:

```bash
# CPU-bound tasks: Use available cores
arandu generate-cep-qa results/ --workers $(nproc)

# Memory-constrained: Reduce workers
arandu batch-transcribe catalog.csv --workers 2
```

### 5. Judge Quality

Always judge transcriptions before downstream tasks:

```bash
# Judge first
arandu judge-transcription results/ --validator-model qwen3:14b

# Then generate QA
arandu generate-cep-qa results/ --workers 4
```

---

**Document Version**: 2.1  
**Last Updated**: 2026-06-16  
**Status**: Aligned with codebase v0.1.0
