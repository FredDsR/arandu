# G-Transcriber

[![.github/workflows/ci.yml](https://github.com/FredDsR/etno-kgc-preprocessing/actions/workflows/ci.yml/badge.svg)](https://github.com/FredDsR/etno-kgc-preprocessing/actions/workflows/ci.yml)
![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/94bfd9f7de8e4f16abcdc62811a81cd0/raw/tests-badge.json)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/94bfd9f7de8e4f16abcdc62811a81cd0/raw/coverage-badge.json)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)

Automated transcription and knowledge graph construction pipeline for ethnographic audio/video archives.

## Overview

G-Transcriber is a comprehensive pipeline for processing ethnographic media collections. It transcribes audio/video files using state-of-the-art speech recognition, generates synthetic QA pairs for retrieval evaluation, and constructs knowledge graphs for semantic analysis.

## Features

### Transcription Pipeline
- **Flexible Model Support**: Use any Whisper model from Hugging Face Hub (e.g., `openai/whisper-large-v3`, `distil-whisper/distil-large-v3`)
- **Hardware Agnostic**: Automatic detection and optimization for CPU, CUDA GPU, or Apple Silicon MPS
- **Quantization Support**: 8-bit quantization for reduced VRAM usage on GPUs
- **Google Drive Integration**: Download files, transcribe, and upload results back to Drive
- **Resilient Transfers**: Resumable downloads/uploads with automatic retry logic, file size validation, and exponential backoff
- **Rich CLI**: Beautiful command-line interface with progress bars and structured output
- **Structured Output**: JSON output with transcription text, timestamps, and metadata
- **Transcription Quality Validation**: Heuristic-based quality scoring detecting wrong language/script, repetition, suspicious patterns, and empty content
- **Results Versioning**: Pipeline-ID-first directory layout with comprehensive run metadata and execution environment tracking

### CEP QA Generation
- **Cognitive Elicitation Pipeline (CEP)**: Generate QA pairs with Bloom's Taxonomy scaffolding (remember, understand, analyze, evaluate)
- **Reasoning Enrichment**: Automatic extraction of reasoning traces and multi-hop reasoning detection
- **LLM-as-a-Judge Validation**: Quality validation with faithfulness, Bloom calibration, and informativeness scoring
- **Scaffolding Context**: Progressive question generation where higher Bloom levels receive context from lower levels
- **Multi-Provider LLM Support**: Ollama (local), OpenAI, or custom OpenAI-compatible endpoints
- **Externalized Prompts**: Language-specific Markdown templates for maintainable prompt engineering

### KG Pipeline (Planned)
> **Status**: Schemas and configuration exist, but CLI commands and pipeline modules are not yet implemented.
- **Knowledge Graph Construction**: Extract entities and relations using AutoSchemaKG (planned)
- **Graph Evaluation**: Measure entity coverage, connectivity, and semantic coherence (planned)
- **Multilingual Support**: Portuguese, English, and Spanish extraction prompts (planned)

## Installation

```bash
# Using uv (recommended - project uses uv as build system)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Transcribe a Local File

```bash
gtranscriber transcribe audio.mp3
```

### Transcribe with Custom Model

```bash
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3
```

### Transcribe with Quantization (Reduced VRAM)

```bash
gtranscriber transcribe audio.mp3 --quantize
```

### Force CPU Execution

```bash
gtranscriber transcribe audio.mp3 --cpu
```

### Transcribe from Google Drive

```bash
gtranscriber drive-transcribe <file-id> --credentials credentials.json
```

### Batch Transcribe from Catalog

Transcribe all audio/video files from a catalog CSV with parallel processing:

```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json --workers 4
```

Advanced options:

```bash
# Use custom output directory
gtranscriber batch-transcribe input/catalog.csv -o transcriptions/ --workers 2

# Use different model with quantization
gtranscriber batch-transcribe input/catalog.csv --model-id openai/whisper-large-v3 --quantize --workers 4

# Resume interrupted job (uses checkpoint automatically)
gtranscriber batch-transcribe input/catalog.csv --workers 4
```

The batch transcribe command:
- Filters only audio/video files from the catalog
- Downloads files from Google Drive
- Extracts media duration and includes it in output
- Processes files in parallel with multiple model instances
- Automatically checkpoints progress for resumption
- Saves results as JSON files with full metadata

### Check System Information

```bash
gtranscriber info
```

### Validate Transcription Quality

```bash
# Validate and update in-place
gtranscriber validate-transcriptions results/

# Report only (no file updates)
gtranscriber validate-transcriptions results/ --report-only --threshold 0.6
```

### List Pipeline Runs

```bash
# List all runs
gtranscriber list-runs

# Filter by pipeline type
gtranscriber list-runs --pipeline cep

# View specific run details
gtranscriber run-info latest --pipeline transcription
```

## CEP QA Pipeline

Generate Cognitive Elicitation Pipeline (CEP) QA pairs from transcriptions with Bloom's Taxonomy scaffolding and LLM-as-a-Judge validation.

```bash
# Generate CEP QA pairs with Ollama (default)
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
gtranscriber generate-cep-qa results/ --no-validate
```

Configuration:

```bash
export GTRANSCRIBER_QA_PROVIDER=ollama                # ollama, openai, custom
export GTRANSCRIBER_QA_MODEL_ID=qwen3:14b             # Model for generation
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=10     # QA pairs per document
export GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1  # Ollama API URL
export GTRANSCRIBER_CEP_ENABLE_VALIDATION=true        # Enable LLM-as-a-Judge
export GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=qwen3:14b  # Validator model
```

Output: `qa_dataset/cep_qa_<gdrive_id>.json`

## KG Pipeline (Planned)

> **Status**: PLANNED - NOT YET IMPLEMENTED. Configuration schemas exist, but there are no CLI commands or pipeline modules for KG construction yet.

Knowledge graph construction using AutoSchemaKG for entity and relation extraction is planned for a future release. The configuration infrastructure is in place:

```bash
export GTRANSCRIBER_KG_PROVIDER=ollama          # ollama, openai, custom
export GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b     # Model for extraction
export GTRANSCRIBER_KG_LANGUAGE=pt              # pt, en, es
export GTRANSCRIBER_KG_MERGE_GRAPHS=true        # Merge into corpus graph
```

Expected output (when implemented): `knowledge_graphs/corpus_graph.graphml`

## Docker Compose Profiles

| Profile | Services | Pipeline |
|---------|----------|----------|
| `cep` | ollama, gtranscriber-cep | CEP QA Pipeline |
| `qa` | ollama, gtranscriber-qa | QA Pipeline |
| `kg` | ollama, gtranscriber-kg | KG Pipeline (planned) |
| `evaluate` | gtranscriber-eval | Evaluation (planned) |
| `cpu` | gtranscriber-cpu | Transcription (CPU) |
| `rocm` | gtranscriber-rocm | Transcription (AMD GPU) |

## SLURM Execution

SLURM scripts are organized by pipeline:

```bash
# Transcription Pipeline
sbatch scripts/slurm/transcription/batch_transcribe.slurm

# QA Pipeline
sbatch scripts/slurm/qa/generate_qa.slurm

# CEP Pipeline
sbatch scripts/slurm/cep/generate_cep_qa.slurm

# KG Pipeline (planned)
sbatch scripts/slurm/kg/build_kg.slurm

# Evaluation (planned)
sbatch scripts/slurm/evaluation/evaluate.slurm
```

## Configuration

The system can be configured via:

1. **Command-line arguments** (highest priority)
2. **Environment variables**:

### Transcription Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_MODEL_ID` | `openai/whisper-large-v3` | Whisper model ID |
| `GTRANSCRIBER_FORCE_CPU` | `false` | Force CPU execution |
| `GTRANSCRIBER_QUANTIZE` | `false` | Enable 8-bit quantization |
| `GTRANSCRIBER_CREDENTIALS` | `credentials.json` | Path to Google OAuth credentials |
| `GTRANSCRIBER_TOKEN` | `token.json` | Path to token file |
| `GTRANSCRIBER_WORKERS` | `1` | Number of parallel workers |

### QA/CEP Pipeline Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_QA_MODEL_ID` | `qwen3:14b` | Model for QA generation |
| `GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `GTRANSCRIBER_QA_TEMPERATURE` | `0.7` | LLM temperature |
| `GTRANSCRIBER_QA_OLLAMA_URL` | `http://localhost:11434/v1` | Ollama API base URL |
| `GTRANSCRIBER_CEP_ENABLE_VALIDATION` | `true` | Enable LLM-as-a-Judge validation |
| `GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID` | `qwen3:14b` | Validator model ID |

### KG Pipeline Settings (Planned)
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_KG_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_KG_MODEL_ID` | `llama3.1:8b` | Model for extraction |
| `GTRANSCRIBER_KG_LANGUAGE` | `pt` | Language code: `pt`, `en`, `es` |
| `GTRANSCRIBER_KG_MERGE_GRAPHS` | `true` | Merge into corpus graph |
| `GTRANSCRIBER_KG_OUTPUT_FORMAT` | `graphml` | Output format: `graphml`, `json` |

### Evaluation Settings (Planned)
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_EVAL_METRICS` | `qa,entity,relation,semantic` | Metrics to compute |
| `GTRANSCRIBER_EVAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Semantic embeddings model |

### Results & Quality Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_RESULTS_BASE_DIR` | `./results` | Base directory for versioned results |
| `GTRANSCRIBER_RESULTS_ENABLE_VERSIONING` | `true` | Enable versioned result directories |
| `GTRANSCRIBER_QUALITY_ENABLED` | `true` | Enable transcription quality validation |
| `GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD` | `0.5` | Minimum quality score (0.0-1.0) |

See [Configuration Guide](docs/user-guide/configuration.md) for complete reference.

## Output Format

Transcription results are saved as JSON files containing:

```json
{
  "gdrive_id": "...",
  "name": "audio.mp3",
  "mimeType": "audio/mpeg",
  "size_bytes": 12345678,
  "duration_milliseconds": 120000,
  "parents": ["parent_folder_id"],
  "web_content_link": "https://drive.google.com/...",
  "transcription_text": "...",
  "detected_language": "en",
  "language_probability": 0.95,
  "model_id": "openai/whisper-large-v3",
  "compute_device": "cuda:0",
  "processing_duration_sec": 45.2,
  "transcription_status": "completed",
  "created_at_enrichment": "2026-02-11T14:30:22",
  "segments": [
    {"text": "...", "start": 0.0, "end": 3.5}
  ],
  "transcription_quality": {
    "script_match_score": 1.0,
    "repetition_score": 0.95,
    "segment_quality_score": 1.0,
    "content_density_score": 0.85,
    "overall_score": 0.94,
    "issues_detected": [],
    "quality_rationale": "High quality transcription"
  },
  "is_valid": true
}
```

## Project Structure

```
arandu/
├── pyproject.toml
├── README.md
├── docker-compose.yml
├── Dockerfile
├── src/
│   └── gtranscriber/
│       ├── __init__.py
│       ├── main.py                      # CLI entrypoint (10 commands)
│       ├── config.py                    # 8 separate config classes
│       ├── schemas.py                   # Pydantic models (transcription, QA, CEP, KG, eval)
│       ├── core/
│       │   ├── drive.py                 # Google Drive integration
│       │   ├── engine.py                # Whisper engine
│       │   ├── hardware.py              # Hardware detection
│       │   ├── io.py                    # File operations
│       │   ├── media.py                 # Media file handling
│       │   ├── batch.py                 # Batch transcription logic
│       │   ├── checkpoint.py            # Checkpoint/resume support
│       │   ├── llm_client.py            # Unified LLM client
│       │   ├── qa_batch.py              # Batch QA generation logic
│       │   ├── results_manager.py       # Results versioning
│       │   ├── transcription_validator.py  # Quality validation
│       │   └── cep/                     # CEP modules
│       │       ├── cep_generator.py     # Main orchestrator
│       │       ├── bloom_scaffolding.py # Bloom taxonomy scaffolding
│       │       ├── reasoning.py         # Reasoning enrichment
│       │       └── validator.py         # LLM-as-a-Judge validation
│       └── utils/
│           ├── logger.py                # Rich logging utilities
│           ├── console.py               # Console utilities
│           └── ui.py                    # Progress bars & UI
├── prompts/
│   └── qa/
│       └── cep/                         # Externalized CEP prompts
│           ├── pt/                      # Portuguese templates
│           └── en/                      # English templates
├── scripts/
│   └── slurm/
│       ├── transcription/               # Transcription SLURM jobs
│       ├── qa/                          # QA generation SLURM jobs
│       ├── cep/                         # CEP SLURM jobs
│       ├── kg/                          # KG SLURM jobs (planned)
│       └── evaluation/                  # Evaluation SLURM jobs (planned)
└── docs/
    ├── README.md                        # Documentation index
    ├── user-guide/
    │   ├── getting-started.md
    │   ├── cli-reference.md
    │   ├── transcription.md
    │   ├── transcription-validation.md
    │   ├── qa-generation.md
    │   ├── cep-qa-generation.md
    │   ├── kg-construction.md           # Planned
    │   ├── evaluation.md                # Planned
    │   └── configuration.md
    ├── development/
    │   ├── architecture.md
    │   ├── schemas.md
    │   ├── dependencies.md
    │   ├── testing.md
    │   └── ci-cd.md
    └── deployment/
        ├── docker.md
        ├── slurm.md
        └── pcad.md
```

## Requirements

- Python >= 3.13
- PyTorch (CUDA 12.4 index)
- Transformers >= 4.57.3
- Google API Python Client >= 2.100.0
- OpenAI SDK >= 1.0.0 (supports Ollama and OpenAI-compatible endpoints)
- Accelerate >= 1.12.0
- bitsandbytes >= 0.49.1
- Tenacity >= 8.0.0
- Rich >= 13.0.0
- Typer >= 0.9.0
- Pydantic >= 2.0.0
- Pydantic Settings >= 2.0.0

Note: NetworkX, scikit-learn, sentence-transformers are planned dependencies for KG/Evaluation phases (not yet in pyproject.toml)

## Documentation

- [Documentation Index](docs/README.md)
- [Getting Started](docs/user-guide/getting-started.md)
- [CLI Reference](docs/user-guide/cli-reference.md)
- [Transcription Guide](docs/user-guide/transcription.md)
- [Transcription Validation Guide](docs/user-guide/transcription-validation.md)
- [QA Generation Guide](docs/user-guide/qa-generation.md)
- [CEP QA Generation Guide](docs/user-guide/cep-qa-generation.md)
- [KG Construction Guide](docs/user-guide/kg-construction.md) (Planned)
- [Evaluation Guide](docs/user-guide/evaluation.md) (Planned)
- [Configuration Reference](docs/user-guide/configuration.md)
- [Architecture](docs/development/architecture.md)
- [Schemas Reference](docs/development/schemas.md)

## License

MIT
