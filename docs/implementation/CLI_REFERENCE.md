# CLI Commands Reference

Complete reference for all command-line interface commands in the Knowledge Graph Construction Pipeline.

## Table of Contents

1. [Command Overview](#command-overview)
2. [Existing Commands](#existing-commands)
3. [New Commands](#new-commands)
4. [Usage Examples](#usage-examples)
5. [Common Options](#common-options)

---

## Command Overview

The G-Transcriber CLI is built with [Typer](https://typer.tiangolo.com/) and provides rich terminal output using [Rich](https://rich.readthedocs.io/).

**Base Command**: `gtranscriber`

**Command Categories**:
- **Transcription** (existing): `transcribe`, `drive-transcribe`, `batch-transcribe`
- **QA Generation** (new): `generate-qa`
- **KG Construction** (new): `build-kg`
- **Evaluation** (new): `evaluate`
- **Utilities** (existing): `refresh-auth`, `info`

---

## Existing Commands

### `transcribe`

Transcribe a single local audio/video file.

**Usage**:
```bash
gtranscriber transcribe FILE_PATH [OPTIONS]
```

**Arguments**:
- `FILE_PATH` - Path to audio/video file

**Options**:
- `--model-id, -m` - Whisper model ID (default: from config)
- `--quantize` - Enable 8-bit quantization
- `--cpu` - Force CPU execution
- `--output, -o` - Output JSON file path

**Example**:
```bash
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3-turbo
```

### `drive-transcribe`

Transcribe a file from Google Drive.

**Usage**:
```bash
gtranscriber drive-transcribe DRIVE_FILE_ID [OPTIONS]
```

**Arguments**:
- `DRIVE_FILE_ID` - Google Drive file ID

**Options**:
- `--credentials` - Path to credentials.json
- `--token` - Path to token.json
- `--model-id, -m` - Whisper model ID
- `--output, -o` - Output JSON file path

**Example**:
```bash
gtranscriber drive-transcribe 1abc123xyz --credentials credentials.json
```

### `batch-transcribe`

Batch transcribe files from a CSV catalog.

**Usage**:
```bash
gtranscriber batch-transcribe CATALOG_FILE [OPTIONS]
```

**Arguments**:
- `CATALOG_FILE` - Path to CSV catalog

**Options**:
- `--credentials` - Path to credentials.json
- `--token` - Path to token.json
- `--output-dir, -o` - Output directory
- `--workers, -w` - Number of parallel workers
- `--model-id, -m` - Whisper model ID
- `--quantize` - Enable quantization
- `--cpu` - Force CPU execution

**Example**:
```bash
gtranscriber batch-transcribe input/catalog.csv --workers 4 --quantize
```

### `refresh-auth`

Refresh Google OAuth2 token.

**Usage**:
```bash
gtranscriber refresh-auth [OPTIONS]
```

**Options**:
- `--credentials` - Path to credentials.json
- `--token` - Path to token.json

**Example**:
```bash
gtranscriber refresh-auth --credentials credentials.json
```

### `info`

Display system hardware information.

**Usage**:
```bash
gtranscriber info
```

**Output**: CPU, GPU, memory, and device information

---

## New Commands

### `generate-qa`

Generate synthetic QA dataset from transcriptions.

**Usage**:
```bash
gtranscriber generate-qa INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing transcription JSON files (EnrichedRecord format)

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output-dir` | `-o` | Path | `qa_dataset` | Output directory for QA dataset |
| `--provider` | `-p` | str | `ollama` | LLM provider: openai, anthropic, ollama |
| `--model-id` | `-m` | str | `llama3.1:8b` | Model ID for generation |
| `--api-key` | | str | None | API key (overrides env) |
| `--ollama-url` | | str | `http://localhost:11434` | Ollama API URL |
| `--workers` | `-w` | int | `2` | Number of parallel workers |
| `--questions` | `-q` | int | `10` | Questions per document |
| `--strategy` | | str | `factual` | Question strategy (repeatable) |
| `--temperature` | | float | `0.7` | Generation temperature |
| `--checkpoint` | | Path | `qa_checkpoint.json` | Checkpoint file path |

**Question Strategies**:
- `factual` - Who, what, when, where questions
- `conceptual` - Why, how questions
- `temporal` - Time-based questions
- `entity` - Entity-focused questions

**Examples**:

```bash
# Basic usage with Ollama
gtranscriber generate-qa results/ -o qa_dataset/ --workers 4

# Multiple strategies
gtranscriber generate-qa results/ \
    --strategy factual \
    --strategy conceptual \
    --strategy temporal \
    --questions 15

# With OpenAI
gtranscriber generate-qa results/ \
    --provider openai \
    --model-id gpt-4o-mini \
    --api-key sk-... \
    --workers 2

# With Claude
gtranscriber generate-qa results/ \
    --provider anthropic \
    --model-id claude-3-sonnet-20240229 \
    --questions 12
```

**Output Structure**:
```
qa_dataset/
├── qa_1abc123xyz.json
├── qa_2def456uvw.json
└── qa_checkpoint.json
```

**Progress Display**:
- Overall progress bar
- Current file being processed
- QA pairs generated per file
- Estimated time remaining

### `build-kg`

Build knowledge graphs from transcriptions using AutoSchemaKG.

**Usage**:
```bash
gtranscriber build-kg INPUT_DIR [OPTIONS]
```

**Arguments**:
- `INPUT_DIR` - Directory containing transcription JSON files

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output-dir` | `-o` | Path | `knowledge_graphs` | Output directory for KGs |
| `--provider` | `-p` | str | `ollama` | LLM provider |
| `--model-id` | `-m` | str | `llama3.1:8b` | Model ID |
| `--api-key` | | str | None | API key (overrides env) |
| `--ollama-url` | | str | `http://localhost:11434` | Ollama API URL |
| `--workers` | `-w` | int | `1` | Parallel workers |
| `--merge/--no-merge` | | bool | `True` | Merge into corpus graph |
| `--format` | `-f` | str | `graphml` | Output format: graphml (default), json |
| `--schema-mode` | | str | `dynamic` | Schema mode: dynamic, predefined |
| `--checkpoint` | | Path | `kg_checkpoint.json` | Checkpoint file path |

**Examples**:

```bash
# Basic usage
gtranscriber build-kg results/ -o knowledge_graphs/

# No merge (individual graphs only)
gtranscriber build-kg results/ --no-merge

# Export as GraphML
gtranscriber build-kg results/ --format graphml

# With OpenAI and multiple workers
gtranscriber build-kg results/ \
    --provider openai \
    --model-id gpt-4o \
    --workers 4 \
    --merge

# With predefined schema
gtranscriber build-kg results/ \
    --schema-mode predefined \
    --schema-file schema.json
```

**Output Structure**:
```
knowledge_graphs/
├── corpus_graph.graphml              # Merged graph (NetworkX-compatible)
├── corpus_graph_metadata.json        # Provenance metadata
├── individual/                       # Per-document graphs
│   ├── 1abc123xyz.graphml
│   └── 2def456uvw.graphml
└── checkpoints/
    └── kg_checkpoint.json
```

**Progress Display**:
- Triple extraction progress
- Schema induction progress
- Graph construction status
- Merge operation status

### `evaluate`

Evaluate knowledge elicitation quality.

**Usage**:
```bash
gtranscriber evaluate QA_DATASET TRANSCRIPTIONS [OPTIONS]
```

**Arguments**:
- `QA_DATASET` - Path to QA dataset directory
- `TRANSCRIPTIONS` - Path to transcriptions directory

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--kg-path` | `-k` | Path | None | Path to knowledge graph (GraphML) |
| `--output` | `-o` | Path | `evaluation_report.json` | Output report path |
| `--metric` | `-m` | str | `qa` | Metric to compute (repeatable) |
| `--embedding-model` | | str | From config | Sentence transformer model |

**Metric Options**:
- `qa` - QA-based metrics (EM, F1, BLEU)
- `entity` - Entity coverage metrics
- `relation` - Relation density metrics
- `semantic` - Semantic quality metrics

**Examples**:

```bash
# All metrics
gtranscriber evaluate qa_dataset/ results/ \
    --kg-path knowledge_graphs/corpus_graph.graphml \
    --output evaluation_report.json

# Specific metrics only
gtranscriber evaluate qa_dataset/ results/ \
    --metric qa \
    --metric entity

# With custom embedding model
gtranscriber evaluate qa_dataset/ results/ \
    --embedding-model sentence-transformers/all-mpnet-base-v2

# QA metrics only (no KG required)
gtranscriber evaluate qa_dataset/ results/ \
    --metric qa \
    --output qa_evaluation.json
```

**Output**:
- JSON report with all computed metrics
- Terminal display of key metrics
- Recommendations for improvement

**Progress Display**:
- Loading datasets
- Computing each metric category
- Final report summary

---

## Usage Examples

### End-to-End Pipeline

Complete pipeline from transcriptions to evaluation:

```bash
# Step 1: Generate QA dataset
gtranscriber generate-qa results/ \
    -o qa_dataset/ \
    --provider ollama \
    --workers 4 \
    --questions 12

# Step 2: Build knowledge graphs
gtranscriber build-kg results/ \
    -o knowledge_graphs/ \
    --provider ollama \
    --workers 2 \
    --merge

# Step 3: Evaluate quality
gtranscriber evaluate qa_dataset/ results/ \
    --kg-path knowledge_graphs/corpus_graph.graphml \
    --output evaluation_report.json

# Step 4: View report
cat evaluation_report.json | jq .
```

### Quick Testing on Sample Data

Test pipeline on 5 sample files:

```bash
# Create sample directory
mkdir -p samples/
cp results/enriched_*.json samples/ | head -5

# Generate QA (fast)
gtranscriber generate-qa samples/ \
    -o qa_test/ \
    --workers 2 \
    --questions 5

# Build KG (fast)
gtranscriber build-kg samples/ \
    -o kg_test/ \
    --workers 1

# Evaluate
gtranscriber evaluate qa_test/ samples/ \
    --kg-path kg_test/corpus_graph.graphml
```

### Resume Interrupted Job

Resume from checkpoint after failure:

```bash
# QA generation resumes automatically
gtranscriber generate-qa results/ \
    -o qa_dataset/ \
    --checkpoint qa_checkpoint.json

# KG construction resumes automatically
gtranscriber build-kg results/ \
    -o knowledge_graphs/ \
    --checkpoint kg_checkpoint.json
```

### Hybrid LLM Approach

Use different LLMs for different tasks:

```bash
# High-quality QA with Claude
gtranscriber generate-qa results/ \
    -o qa_dataset/ \
    --provider anthropic \
    --model-id claude-3-sonnet-20240229 \
    --questions 15

# Cost-effective KG with Ollama
gtranscriber build-kg results/ \
    -o knowledge_graphs/ \
    --provider ollama \
    --model-id llama3.1:70b
```

---

## Common Options

### Global Options

Available for all commands:

| Option | Description |
|--------|-------------|
| `--help` | Show command help |
| `--version` | Show version |
| `--verbose, -v` | Verbose output |
| `--quiet, -q` | Suppress non-error output |

**Usage**:
```bash
gtranscriber --help
gtranscriber --version
gtranscriber generate-qa --help
```

### Configuration Override

Command-line arguments override configuration:

```bash
# Config says ollama, but we override to openai
gtranscriber generate-qa results/ --provider openai
```

### Environment Variables

Set via environment instead of CLI:

```bash
export GTRANSCRIBER_QA_PROVIDER=openai
export GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini

# Now uses OpenAI
gtranscriber generate-qa results/
```

### Output Formats

Commands support different output formats:

```bash
# GraphML output (default, NetworkX-compatible)
gtranscriber build-kg results/ --format graphml

# JSON output (alternative)
gtranscriber build-kg results/ --format json

# Evaluation report as JSON
gtranscriber evaluate qa_dataset/ results/ -o report.json
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
Solution: Check directory path and file format
```

**Checkpoint Corruption**:
```bash
Error: Checkpoint file corrupted
Solution: Delete checkpoint.json and restart
```

### Verbose Mode

Get detailed error information:

```bash
gtranscriber generate-qa results/ --verbose
```

### Dry Run Mode (Future)

Preview without execution:

```bash
gtranscriber generate-qa results/ --dry-run
# Output: Would process 187 files, generate ~2244 QA pairs
```

---

## Tips and Best Practices

### 1. Start Small

Test on sample data before full corpus:

```bash
# Test on 5 files first
mkdir samples && cp results/*.json samples/ | head -5
gtranscriber generate-qa samples/ -o qa_test/
```

### 2. Use Checkpoints

Always let checkpoint files persist for resumability:

```bash
# Default checkpoint location
gtranscriber generate-qa results/ -o qa_dataset/
# Creates: qa_dataset/qa_checkpoint.json

# Resume automatically if interrupted
gtranscriber generate-qa results/ -o qa_dataset/
```

### 3. Monitor Progress

Use verbose mode to monitor processing:

```bash
gtranscriber build-kg results/ --verbose
```

### 4. Optimize Workers

Adjust workers based on available resources:

```bash
# CPU-bound (QA/KG): Use available cores
gtranscriber generate-qa results/ --workers $(nproc)

# Memory-constrained: Reduce workers
gtranscriber build-kg results/ --workers 2
```

### 5. Separate Concerns

Process in stages for better debugging:

```bash
# Stage 1: QA
gtranscriber generate-qa results/ -o qa_dataset/

# Stage 2: KG
gtranscriber build-kg results/ -o knowledge_graphs/

# Stage 3: Evaluation
gtranscriber evaluate qa_dataset/ results/
```

---

## Shell Completion

Enable shell completion for better UX:

### Bash

```bash
eval "$(_GTRANSCRIBER_COMPLETE=bash_source gtranscriber)"
```

### Zsh

```zsh
eval "$(_GTRANSCRIBER_COMPLETE=zsh_source gtranscriber)"
```

### Fish

```fish
eval (env _GTRANSCRIBER_COMPLETE=fish_source gtranscriber)
```

Add to your shell's RC file for persistence.

---

## Command Chaining

Chain commands with shell operators:

```bash
# Sequential execution
gtranscriber generate-qa results/ && \
gtranscriber build-kg results/ && \
gtranscriber evaluate qa_dataset/ results/

# Parallel execution (independent tasks)
gtranscriber generate-qa results/ &
gtranscriber build-kg results/ &
wait
gtranscriber evaluate qa_dataset/ results/
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
