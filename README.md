# G-Transcriber

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

### QA Pipeline
- **QA Generation**: Generate synthetic question-answer pairs from transcriptions using LLMs
- **QA Evaluation**: Measure answer quality with Exact Match, F1, and BLEU scores
- **Multi-Provider LLM Support**: Ollama (local), OpenAI, or custom providers

### KG Pipeline
- **Knowledge Graph Construction**: Extract entities and relations using AutoSchemaKG
- **Graph Evaluation**: Measure entity coverage, connectivity, and semantic coherence
- **Multilingual Support**: Portuguese, English, and Spanish extraction prompts

## Installation

```bash
# Using pip
pip install -e .

# Using uv (recommended)
uv pip install -e .
```

## Usage

### Transcribe a Local File

```bash
gtranscriber transcribe audio.mp3
```

### Transcribe with Custom Model

```bash
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3-turbo
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
gtranscriber batch-transcribe input/catalog.csv --model-id openai/whisper-large-v3-turbo --quantize --workers 4

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

## QA Pipeline

Generate synthetic question-answer pairs from transcription results for training and evaluating retrieval systems.

```bash
# Generate QA pairs
docker compose --profile qa up

# Evaluate QA quality
GTRANSCRIBER_EVALUATION_METRICS=qa docker compose --profile evaluate up
```

Configuration:

```bash
export GTRANSCRIBER_QA_PROVIDER=ollama          # ollama, openai, custom
export GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b     # Model for generation
export GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10   # QA pairs per document
```

Output: `qa_dataset/qa_<gdrive_id>.json`

## KG Pipeline

Build knowledge graphs from transcription results using AutoSchemaKG for entity and relation extraction.

```bash
# Build knowledge graphs
docker compose --profile kg up

# Evaluate graph quality
GTRANSCRIBER_EVALUATION_METRICS=entity,relation,semantic docker compose --profile evaluate up
```

Configuration:

```bash
export GTRANSCRIBER_KG_PROVIDER=ollama          # ollama, openai, custom
export GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b     # Model for extraction
export GTRANSCRIBER_KG_LANGUAGE=pt              # pt, en, es
export GTRANSCRIBER_KG_MERGE_GRAPHS=true        # Merge into corpus graph
```

Output: `knowledge_graphs/corpus_graph.graphml`

## Docker Compose Profiles

| Profile | Services | Pipeline |
|---------|----------|----------|
| `qa` | ollama, gtranscriber-qa | QA Pipeline |
| `kg` | ollama, gtranscriber-kg | KG Pipeline |
| `evaluate` | gtranscriber-eval | Both (configurable) |
| `cpu` | gtranscriber-cpu | Transcription (CPU) |
| `rocm` | gtranscriber-rocm | Transcription (AMD GPU) |

## SLURM Execution

```bash
# QA Pipeline
sbatch scripts/slurm/run_qa_generation.slurm

# KG Pipeline
sbatch scripts/slurm/run_kg_construction.slurm

# Evaluation
sbatch scripts/slurm/run_evaluation.slurm
```

## Configuration

The system can be configured via:

1. **Command-line arguments** (highest priority)
2. **Environment variables**:

### Transcription Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_MODEL_ID` | `openai/whisper-large-v3-turbo` | Whisper model ID |
| `GTRANSCRIBER_FORCE_CPU` | `false` | Force CPU execution |
| `GTRANSCRIBER_QUANTIZE` | `false` | Enable 8-bit quantization |
| `GTRANSCRIBER_CREDENTIALS` | - | Path to Google OAuth credentials |
| `GTRANSCRIBER_TOKEN` | - | Path to token file |

### QA Pipeline Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_QA_MODEL_ID` | `llama3.1:8b` | Model for QA generation |
| `GTRANSCRIBER_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `GTRANSCRIBER_QA_TEMPERATURE` | `0.7` | LLM temperature |

### KG Pipeline Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_KG_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_KG_MODEL_ID` | `llama3.1:8b` | Model for extraction |
| `GTRANSCRIBER_KG_LANGUAGE` | `pt` | Language code: `pt`, `en`, `es` |
| `GTRANSCRIBER_KG_MERGE_GRAPHS` | `true` | Merge into corpus graph |
| `GTRANSCRIBER_KG_OUTPUT_FORMAT` | `graphml` | Output format: `graphml`, `json` |

### Evaluation Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_EVALUATION_METRICS` | `qa,entity,relation,semantic` | Metrics to compute |
| `GTRANSCRIBER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Semantic embeddings model |

See [Configuration Guide](docs/usage/CONFIGURATION.md) for complete reference.

## Output Format

Transcription results are saved as JSON files containing:

```json
{
  "gdrive_id": "...",
  "name": "audio.mp3",
  "mimeType": "audio/mpeg",
  "size_bytes": 12345678,
  "duration_milliseconds": 120000,
  "transcription_text": "...",
  "detected_language": "en",
  "language_probability": 0.95,
  "model_id": "openai/whisper-large-v3",
  "compute_device": "cuda:0",
  "processing_duration_sec": 45.2,
  "transcription_status": "completed",
  "segments": [
    {"text": "...", "start": 0.0, "end": 3.5}
  ]
}
```

## Project Structure

```
etno-kgc-preprocessing/
├── pyproject.toml
├── README.md
├── docker-compose.yml
├── Dockerfile
├── src/
│   └── gtranscriber/
│       ├── __init__.py
│       ├── main.py              # CLI entrypoint
│       ├── config.py            # Configuration (transcription + KGC)
│       ├── schemas.py           # Pydantic models (transcription + KGC)
│       ├── core/
│       │   ├── drive.py         # Google Drive integration
│       │   ├── engine.py        # Whisper engine
│       │   ├── hardware.py      # Hardware detection
│       │   ├── io.py            # File operations
│       │   └── llm_client.py    # Unified LLM client
│       └── utils/
│           ├── logger.py        # Rich logging
│           └── ui.py            # Progress bars
├── scripts/
│   └── slurm/
│       ├── grace.slurm                  # Transcription job
│       ├── run_qa_generation.slurm      # QA generation job
│       ├── run_kg_construction.slurm    # KG construction job
│       └── run_evaluation.slurm         # Evaluation job
└── docs/
    └── usage/
        ├── README.md            # Usage overview
        ├── QA_GENERATION.md     # QA generation guide
        ├── KG_CONSTRUCTION.md   # KG construction guide
        ├── EVALUATION.md        # Evaluation guide
        └── CONFIGURATION.md     # Configuration reference
```

## Requirements

- Python >= 3.13
- PyTorch
- Transformers
- Google API Python Client
- OpenAI SDK (for LLM integration)
- NetworkX (for knowledge graphs)
- Rich
- Typer
- Pydantic

## Documentation

- [Usage Overview](docs/usage/README.md)
- [QA Generation Guide](docs/usage/QA_GENERATION.md)
- [KG Construction Guide](docs/usage/KG_CONSTRUCTION.md)
- [Evaluation Guide](docs/usage/EVALUATION.md)
- [Configuration Reference](docs/usage/CONFIGURATION.md)

## License

MIT
