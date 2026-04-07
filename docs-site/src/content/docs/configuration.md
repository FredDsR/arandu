---
title: Configuration Reference
description: Complete reference for all configuration settings in the Arandu pipeline.
---

Complete reference for all configuration settings in the Arandu pipeline.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [TranscriberConfig](#transcriberconfig)
3. [QAConfig](#qaconfig)
4. [CEPConfig](#cepconfig)
5. [KGConfig](#kgconfig)
6. [EvaluationConfig](#evaluationconfig)
7. [LLMConfig](#llmconfig)
8. [ResultsConfig](#resultsconfig)
9. [TranscriptionQualityConfig](#transcriptionqualityconfig)
10. [Environment Variables](#environment-variables)
11. [Configuration Examples](#configuration-examples)

---

## Configuration System Overview

The Arandu project uses **Pydantic Settings** for configuration management with hierarchical loading:

1. **Command-line arguments** (highest priority)
2. **Environment variables** with config-specific prefixes
3. **`.env` file** in project root
4. **Default values** in `config.py` (lowest priority)

**Configuration File**: `src/arandu/config.py`

**Architecture**: The system uses **8 separate configuration classes**, each with its own environment variable prefix:
- `TranscriberConfig` - Prefix: `ARANDU_`
- `QAConfig` - Prefix: `ARANDU_QA_`
- `CEPConfig` - Prefix: `ARANDU_CEP_`
- `KGConfig` - Prefix: `ARANDU_KG_`
- `EvaluationConfig` - Prefix: `ARANDU_EVAL_`
- `LLMConfig` - No prefix (uses aliases like `OPENAI_API_KEY`)
- `ResultsConfig` - Prefix: `ARANDU_RESULTS_`
- `TranscriptionQualityConfig` - Prefix: `ARANDU_QUALITY_`

**Usage**:
```python
from arandu.config import TranscriberConfig, QAConfig

transcriber_config = TranscriberConfig()
qa_config = QAConfig()
print(transcriber_config.model_id)  # openai/whisper-large-v3
print(qa_config.provider)  # ollama
```

---

## TranscriberConfig

Configuration settings for the transcription pipeline.

**Environment Prefix**: `ARANDU_`

### Model Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_id` | `str` | `"openai/whisper-large-v3"` | Hugging Face model ID for Whisper transcription |
| `language` | `str \| None` | `None` | Language code (e.g., 'pt'). If None, auto-detect |
| `return_timestamps` | `bool` | `True` | Return timestamps for transcription segments |
| `chunk_length_s` | `int` | `30` | Audio chunk length in seconds |
| `stride_length_s` | `int` | `5` | Stride length in seconds between chunks |

### Hardware Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `force_cpu` | `bool` | `False` | Force CPU execution instead of GPU |
| `quantize` | `bool` | `False` | Enable 8-bit quantization to reduce VRAM |
| `quantize_bits` | `int` | `8` | Number of bits for quantization |

### Google Drive Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `credentials` | `str` | `"credentials.json"` | Path to Google OAuth2 credentials file |
| `token` | `str` | `"token.json"` | Path to Google OAuth2 token file |
| `scopes` | `list[str]` | `["https://www.googleapis.com/auth/drive"]` | OAuth2 scopes for Google Drive API |

**Note**: `credentials_file` and `token_file` are backward-compatible property aliases for `credentials` and `token`.

### Batch Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `workers` | `int` | `1` | Number of parallel workers for batch processing |
| `catalog_file` | `str` | `"catalog.csv"` | Name of the catalog CSV file |

### Path Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `input_dir` | `str` | `"./input"` | Directory containing input files |
| `results_dir` | `str` | `"./results"` | Directory for transcription results |
| `credentials_dir` | `str` | `"./"` | Directory containing credentials and token files |
| `hf_cache_dir` | `str` | `"./cache/huggingface"` | Hugging Face cache directory for model storage |

### Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `temp_dir` | `str` | `"/tmp/arandu"` (platform-specific) | Temporary directory for file processing |
| `max_retries` | `int` | `3` | Maximum number of retry attempts for failed operations |
| `retry_delay` | `float` | `1.0` | Delay in seconds between retry attempts |

**Example Configuration**:
```python
from arandu.config import TranscriberConfig

config = TranscriberConfig()
# Or with custom settings:
config = TranscriberConfig(
    model_id="openai/whisper-large-v3",
    force_cpu=False,
    workers=4
)
```

---

## QAConfig

Configuration settings for the QA generation pipeline.

**Environment Prefix**: `ARANDU_QA_`

### LLM Provider Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "openai", "ollama", "custom" |
| `model_id` | `str` | `"qwen3:14b"` | Model ID for QA generation |
| `ollama_url` | `str` | `"http://localhost:11434/v1"` | Ollama API base URL |
| `base_url` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints |

### Generation Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `questions_per_document` | `int` | `10` | Number of QA pairs to generate per document (min: 1, max: 50) |
| `temperature` | `float` | `0.7` | Temperature for QA generation LLM (range: 0.0-2.0) |
| `max_tokens` | `int` | `2048` | Max tokens for QA generation LLM (min: 1) |

### Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | `Path` | `Path("qa_dataset")` | Output directory for QA datasets |

### Language and Processing

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language code for QA generation prompts (ISO 639-1: 'en' or 'pt') |
| `workers` | `int` | `2` | Number of parallel workers for QA generation |

**Example Configuration**:
```python
from arandu.config import QAConfig

config = QAConfig(
    provider="ollama",
    model_id="qwen3:14b",
    questions_per_document=15,
    language="pt"
)
```

---

## CEPConfig

Configuration settings for the CEP (Cognitive Elicitation Pipeline) with Bloom's Taxonomy scaffolding and LLM-as-a-Judge validation.

**Environment Prefix**: `ARANDU_CEP_`

### Module Toggles

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_reasoning_traces` | `bool` | `True` | Enable reasoning trace generation for answers |
| `enable_validation` | `bool` | `True` | Enable LLM-as-a-Judge validation (requires additional LLM calls) |

### Module I - Bloom Scaffolding Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `bloom_levels` | `list[str]` | `["remember", "understand", "analyze", "evaluate"]` | Bloom levels for question generation |
| `bloom_distribution` | `dict[str, float]` | `{"remember": 0.2, "understand": 0.3, "analyze": 0.3, "evaluate": 0.2}` | Distribution per level (must sum to 1.0) |
| `enable_scaffolding_context` | `bool` | `True` | Pass previously generated QA pairs as context to higher Bloom levels |
| `max_scaffolding_pairs` | `int` | `10` | Max prior QA pairs to include as scaffolding context (min: 1, max: 50) |

**Valid Bloom Levels**: `remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`

### Module II - Reasoning Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_hop_count` | `int` | `3` | Maximum reasoning hops to detect for multi-hop questions (min: 1, max: 5) |

### Module III - LLM-as-a-Judge Validation Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `validator_provider` | `str` | `"ollama"` | LLM provider for validation: "openai", "ollama", "custom" |
| `validator_model_id` | `str` | `"qwen3:14b"` | Model ID for LLM-as-a-Judge validation |
| `validator_temperature` | `float` | `0.3` | Temperature for validator (low for consistent evaluation, range: 0.0-1.0) |
| `validation_threshold` | `float` | `0.6` | Minimum overall score to pass validation (range: 0.0-1.0) |

### Validation Scoring Weights

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `faithfulness_weight` | `float` | `0.4` | Weight for faithfulness score in overall calculation (range: 0.0-1.0) |
| `bloom_calibration_weight` | `float` | `0.3` | Weight for Bloom calibration score in overall calculation (range: 0.0-1.0) |
| `informativeness_weight` | `float` | `0.3` | Weight for informativeness score in overall calculation (range: 0.0-1.0) |

> **Note**: The three scoring weights (`faithfulness_weight`, `bloom_calibration_weight`, `informativeness_weight`) must sum to 1.0. A `@model_validator` enforces this constraint.

### Language Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language for CEP prompts (ISO 639-1: 'pt' or 'en') |

**Example Configuration**:
```python
from arandu.config import CEPConfig

config = CEPConfig(
    bloom_levels=["remember", "understand", "analyze"],
    bloom_distribution={"remember": 0.3, "understand": 0.4, "analyze": 0.3},
    enable_scaffolding_context=True,
    validation_threshold=0.7
)
```

---

## KGConfig

Configuration settings for the knowledge graph construction pipeline.

**Environment Prefix**: `ARANDU_KG_`

### LLM Provider Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "openai", "ollama", "custom" |
| `model_id` | `str` | `"llama3.1:8b"` | Model ID for KG construction |
| `ollama_url` | `str` | `"http://localhost:11434/v1"` | Ollama API base URL for KG construction |
| `base_url` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints |

### Backend Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `str` | `"atlas"` | KGC backend: `"atlas"` (AutoSchemaKG) |
| `backend_options` | `dict` | `{}` | Backend-specific options (e.g., `chunk_size`, `batch_size_triple`, `max_workers`) |

**Backend Validation**: Must match pattern `^(atlas)$`

### LLM Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `temperature` | `float` | `0.5` | Temperature for KG construction LLM (lower = more consistent, range: 0.0-2.0) |

### Language and Prompts

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language code for extraction prompts (ISO 639-1): `"pt"`, `"en"` |

Prompts are stored in `prompts/kg/atlas/` using language-keyed JSON files. The atlas backend loads the appropriate language at runtime.

### Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | `Path` | `Path("knowledge_graphs")` | Output directory for knowledge graphs |

**Example Configuration**:
```python
from arandu.config import KGConfig

config = KGConfig(
    backend="atlas",
    provider="ollama",
    model_id="qwen3:14b",
    language="pt",
    temperature=0.5,
    backend_options={"chunk_size": 4096, "max_workers": 4},
)
```

**Atlas Backend Options** (passed via `backend_options`):

| Option | Default | Description |
|--------|---------|-------------|
| `batch_size_triple` | `3` | Batch size for triple extraction |
| `batch_size_concept` | `16` | Batch size for concept generation |
| `chunk_size` | `8192` | Characters per text chunk |
| `max_new_tokens` | `2048` | Max tokens for LLM generation |
| `include_concept` | `true` | Whether to run concept generation |
| `max_workers` | `3` | Thread pool size for API calls |

---

## EvaluationConfig

Configuration settings for the evaluation pipeline.

**Environment Prefix**: `ARANDU_EVAL_`

### Metrics Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `metrics` | `list[str]` | `["qa", "entity", "relation", "semantic"]` | Metrics to compute |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Sentence transformer model for semantic embeddings |

**Valid Metrics**: `qa`, `entity`, `relation`, `semantic`
- `qa` - QA-based metrics (EM, F1, BLEU)
- `entity` - Entity coverage metrics
- `relation` - Relation density metrics
- `semantic` - Semantic quality metrics

### Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | `Path` | `Path("evaluation")` | Output directory for evaluation reports |

### Input Directory Overrides

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `qa_dir` | `Path` | `Path("qa_dataset")` | Directory containing QA dataset |
| `kg_dir` | `Path` | `Path("knowledge_graphs")` | Directory containing knowledge graphs |
| `results_dir` | `Path` | `Path("results")` | Directory containing transcription results |

**Example Configuration**:
```python
from arandu.config import EvaluationConfig

config = EvaluationConfig(
    metrics=["qa", "entity", "semantic"],
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## LLMConfig

Shared LLM configuration settings for API keys and shared LLM settings across pipelines.

**Environment Prefix**: None (uses field aliases)

### API Keys

| Setting | Type | Default | Alias | Description |
|---------|------|---------|-------|-------------|
| `openai_api_key` | `str \| None` | `None` | `OPENAI_API_KEY` | OpenAI API key |
| `base_url` | `str \| None` | `None` | `ARANDU_LLM_BASE_URL` | Custom base URL for OpenAI-compatible endpoints |

**Example Configuration**:
```python
from arandu.config import LLMConfig

config = LLMConfig()
# Loaded from OPENAI_API_KEY and ARANDU_LLM_BASE_URL env vars
```

**Environment Variables**:
```bash
export OPENAI_API_KEY=sk-...
export ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

---

## ResultsConfig

Configuration for versioned results management.

**Environment Prefix**: `ARANDU_RESULTS_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_dir` | `Path` | `Path("./results")` | Base directory for versioned results |
| `enable_versioning` | `bool` | `True` | Enable versioned result directories |

**Example Configuration**:
```python
from arandu.config import ResultsConfig

config = ResultsConfig(
    base_dir=Path("/data/results"),
    enable_versioning=True
)
```

---

## TranscriptionQualityConfig

Configuration for transcription quality validation with heuristic quality checks.

**Environment Prefix**: `ARANDU_QUALITY_`

### General Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable transcription quality validation |
| `quality_threshold` | `float` | `0.5` | Minimum quality score to mark transcription as valid (range: 0.0-1.0) |
| `expected_language` | `str` | `"pt"` | Expected language code (e.g., 'pt', 'en') |

### Scoring Weights

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `script_match_weight` | `float` | `0.35` | Weight for script/charset match check |
| `repetition_weight` | `float` | `0.30` | Weight for repetition detection |
| `segment_quality_weight` | `float` | `0.20` | Weight for segment pattern analysis |
| `content_density_weight` | `float` | `0.15` | Weight for content density check |

> **Note**: The four dimension weights must sum to 1.0. A `@model_validator` enforces this constraint at initialization.

### Validation Thresholds

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_non_latin_ratio` | `float` | `0.1` | Maximum ratio of non-Latin characters for Latin languages |
| `max_word_repetition_ratio` | `float` | `0.15` | Maximum ratio of most repeated word |
| `max_phrase_repetition_count` | `int` | `4` | Maximum allowed repetitions of same phrase |
| `suspicious_uniform_intervals` | `int` | `5` | Number of consecutive uniform 1-second intervals to flag |
| `min_words_per_minute` | `float` | `30.0` | Minimum words per minute threshold |
| `max_words_per_minute` | `float` | `300.0` | Maximum words per minute threshold |
| `max_empty_segment_ratio` | `float` | `0.2` | Maximum ratio of empty segments before flagging |
| `uniform_interval_tolerance` | `float` | `0.1` | Tolerance (±seconds) for detecting uniform 1-second intervals |

**Example Configuration**:
```python
from arandu.config import TranscriptionQualityConfig

config = TranscriptionQualityConfig(
    enabled=True,
    quality_threshold=0.6,
    expected_language="pt"
)
```

**See also**: [Transcription Validation Guide](/guides/transcription-validation/) for full usage details

---

## Environment Variables

Configuration settings are loaded from environment variables with config-specific prefixes.

### Prefix Reference

| Config Class | Prefix | Example |
|--------------|--------|---------|
| `TranscriberConfig` | `ARANDU_` | `ARANDU_MODEL_ID` |
| `QAConfig` | `ARANDU_QA_` | `ARANDU_QA_PROVIDER` |
| `CEPConfig` | `ARANDU_CEP_` | `ARANDU_CEP_ENABLE_VALIDATION` |
| `KGConfig` | `ARANDU_KG_` | `ARANDU_KG_PROVIDER` |
| `EvaluationConfig` | `ARANDU_EVAL_` | `ARANDU_EVAL_METRICS` |
| `LLMConfig` | (No prefix) | `OPENAI_API_KEY`, `ARANDU_LLM_BASE_URL` |
| `ResultsConfig` | `ARANDU_RESULTS_` | `ARANDU_RESULTS_BASE_DIR` |
| `TranscriptionQualityConfig` | `ARANDU_QUALITY_` | `ARANDU_QUALITY_ENABLED` |

### Format

```bash
<PREFIX><SETTING_NAME>=<value>
```

### Examples by Config Class

**TranscriberConfig** (`ARANDU_`):
```bash
export ARANDU_MODEL_ID=openai/whisper-large-v3
export ARANDU_FORCE_CPU=false
export ARANDU_WORKERS=4
export ARANDU_RETRY_DELAY=1.0
```

**QAConfig** (`ARANDU_QA_`):
```bash
export ARANDU_QA_PROVIDER=openai
export ARANDU_QA_MODEL_ID=gpt-4o-mini
export ARANDU_QA_QUESTIONS_PER_DOCUMENT=15
export ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
export ARANDU_QA_LANGUAGE=pt
```

**CEPConfig** (`ARANDU_CEP_`):
```bash
export ARANDU_CEP_ENABLE_VALIDATION=true
export ARANDU_CEP_BLOOM_LEVELS=remember,understand,analyze
export ARANDU_CEP_VALIDATION_THRESHOLD=0.7
export ARANDU_CEP_VALIDATOR_PROVIDER=ollama
```

**KGConfig** (`ARANDU_KG_`):
```bash
export ARANDU_KG_PROVIDER=openai
export ARANDU_KG_MODEL_ID=gpt-4o
export ARANDU_KG_MERGE_GRAPHS=true
export ARANDU_KG_LANGUAGE=pt
export ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
```

**EvaluationConfig** (`ARANDU_EVAL_`):
```bash
export ARANDU_EVAL_METRICS=qa,entity,relation,semantic
export ARANDU_EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**LLMConfig** (No prefix, uses aliases):
```bash
export OPENAI_API_KEY=sk-...
export ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

**ResultsConfig** (`ARANDU_RESULTS_`):
```bash
export ARANDU_RESULTS_BASE_DIR=/data/results
export ARANDU_RESULTS_ENABLE_VERSIONING=true
```

**TranscriptionQualityConfig** (`ARANDU_QUALITY_`):
```bash
export ARANDU_QUALITY_ENABLED=true
export ARANDU_QUALITY_QUALITY_THRESHOLD=0.6
export ARANDU_QUALITY_EXPECTED_LANGUAGE=pt
```

### API Keys

**Sensitive values should be set as environment variables only** (never commit to git):

```bash
export OPENAI_API_KEY=sk-...
export ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

These can also be set in `.env` file:
```bash
# .env file
OPENAI_API_KEY=sk-...
ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

**Note**: The `.env` file should be added to `.gitignore`.

---

## Configuration Examples

### Example 1: Local Development with Ollama

**.env file**:
```bash
# Transcription
ARANDU_MODEL_ID=openai/whisper-large-v3
ARANDU_WORKERS=2
ARANDU_RETRY_DELAY=1.0

# QA Generation
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
ARANDU_QA_QUESTIONS_PER_DOCUMENT=10
ARANDU_QA_LANGUAGE=pt

# CEP (Cognitive Elicitation Pipeline)
ARANDU_CEP_ENABLE_VALIDATION=true
ARANDU_CEP_VALIDATION_THRESHOLD=0.6

# KG Construction
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
ARANDU_KG_MERGE_GRAPHS=true
ARANDU_KG_OUTPUT_FORMAT=graphml
ARANDU_KG_LANGUAGE=pt

# Evaluation
ARANDU_EVAL_METRICS=qa,entity,relation,semantic

# Transcription Quality Validation
ARANDU_QUALITY_ENABLED=true
ARANDU_QUALITY_QUALITY_THRESHOLD=0.5
```

**CLI Usage**:
```bash
# QA generation (uses .env settings)
arandu generate-cep-qa results/

# Override specific settings
arandu generate-cep-qa results/ --questions 15 --provider ollama
```

### Example 2: Production with OpenAI API

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA Generation with OpenAI
ARANDU_QA_PROVIDER=openai
ARANDU_QA_MODEL_ID=gpt-4o-mini
ARANDU_QA_QUESTIONS_PER_DOCUMENT=12
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_LANGUAGE=en

# CEP with OpenAI
ARANDU_CEP_ENABLE_VALIDATION=true
ARANDU_CEP_VALIDATOR_PROVIDER=openai
ARANDU_CEP_VALIDATOR_MODEL_ID=gpt-4o-mini
ARANDU_CEP_VALIDATION_THRESHOLD=0.7

# KG Construction with OpenAI
ARANDU_KG_PROVIDER=openai
ARANDU_KG_MODEL_ID=gpt-4o
ARANDU_KG_TEMPERATURE=0.5
ARANDU_KG_MERGE_GRAPHS=true

# Results versioning
ARANDU_RESULTS_BASE_DIR=/data/transcriptions
ARANDU_RESULTS_ENABLE_VERSIONING=true

# Output directories
ARANDU_QA_OUTPUT_DIR=/data/qa_dataset
ARANDU_KG_OUTPUT_DIR=/data/knowledge_graphs
ARANDU_EVAL_OUTPUT_DIR=/data/evaluation
```

### Example 3: Hybrid Approach (OpenAI + Ollama)

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA with OpenAI (higher quality)
ARANDU_QA_PROVIDER=openai
ARANDU_QA_MODEL_ID=gpt-4o
ARANDU_QA_QUESTIONS_PER_DOCUMENT=15

# KG with Ollama (cost-effective)
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1

# Workers
ARANDU_WORKERS=4
```

### Example 4: SLURM Cluster Configuration

**SLURM script** (`run_qa_generation.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=arandu-qa
#SBATCH --partition=grace
#SBATCH --cpus-per-task=16

# Set configuration via environment
export ARANDU_QA_PROVIDER=ollama
export ARANDU_QA_MODEL_ID=qwen3:14b
export ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
export ARANDU_QA_WORKERS=8
export ARANDU_QA_QUESTIONS_PER_DOCUMENT=12

# Use $SCRATCH for I/O
export ARANDU_RESULTS_BASE_DIR=$SCRATCH/results
export ARANDU_QA_OUTPUT_DIR=$SCRATCH/qa_dataset

# Run via Docker
source scripts/slurm/job_common.sh
docker compose --profile qa up arandu-qa --abort-on-container-exit
```

### Example 5: Docker Compose Environment

**docker-compose.override.yml** (local overrides):
```yaml
version: '3.8'

services:
  arandu-qa:
    environment:
      - ARANDU_QA_PROVIDER=ollama
      - ARANDU_QA_MODEL_ID=qwen3:14b
      - ARANDU_QA_OLLAMA_URL=http://host.docker.internal:11434/v1
      - ARANDU_QA_WORKERS=4
      - ARANDU_QA_QUESTIONS_PER_DOCUMENT=10
    volumes:
      - ./results:/app/results:ro
      - ./qa_dataset:/app/qa_dataset:rw

  arandu-kg:
    environment:
      - ARANDU_KG_PROVIDER=ollama
      - ARANDU_KG_MODEL_ID=llama3.1:8b
      - ARANDU_KG_OLLAMA_URL=http://host.docker.internal:11434/v1
      - ARANDU_KG_MERGE_GRAPHS=true
      - ARANDU_KG_WORKERS=2
    volumes:
      - ./results:/app/results:ro
      - ./knowledge_graphs:/app/knowledge_graphs:rw
```

### Example 6: CEP with Advanced Bloom Scaffolding

**.env file**:
```bash
# CEP Settings
ARANDU_CEP_ENABLE_REASONING_TRACES=true
ARANDU_CEP_ENABLE_VALIDATION=true
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=15

# Bloom Levels (custom distribution)
ARANDU_CEP_BLOOM_LEVELS=remember,understand,analyze,evaluate

# LLM-as-a-Judge Validation
ARANDU_CEP_VALIDATOR_PROVIDER=ollama
ARANDU_CEP_VALIDATOR_MODEL_ID=qwen3:14b
ARANDU_CEP_VALIDATOR_TEMPERATURE=0.3
ARANDU_CEP_VALIDATION_THRESHOLD=0.7

# Scoring weights (must sum to 1.0)
ARANDU_CEP_FAITHFULNESS_WEIGHT=0.4
ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT=0.3
ARANDU_CEP_INFORMATIVENESS_WEIGHT=0.3
```

---

## Configuration Validation

The configuration system includes validation rules enforced by Pydantic.

### Type Validation

```python
# From QAConfig
questions_per_document: int = Field(
    default=10,
    ge=1,        # Must be >= 1
    le=50,       # Must be <= 50
)
```

### Pattern Validation

```python
# From KGConfig
output_format: str = Field(
    default="graphml",
    pattern="^(graphml|json)$"  # Must match pattern
)
```

### Custom Field Validation

```python
# From CEPConfig
@field_validator("bloom_levels")
@classmethod
def validate_bloom_levels(cls, v: list[str]) -> list[str]:
    valid_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    for level in v:
        if level not in valid_levels:
            raise ValueError(f"Invalid Bloom level: {level!r}")
    return v
```

### Model Validation (Cross-Field)

```python
# From TranscriptionQualityConfig
@model_validator(mode="after")
def validate_scoring_weights(self) -> TranscriptionQualityConfig:
    total = (
        self.script_match_weight
        + self.repetition_weight
        + self.segment_quality_weight
        + self.content_density_weight
    )
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Quality scoring weights must sum to 1.0, got {total:.3f}")
    return self
```

---

## Configuration Loading Order

Pydantic Settings loads configuration in the following priority order (highest to lowest):

1. **Command-line arguments** (highest priority):
   ```bash
   arandu generate-cep-qa results/ --provider ollama --questions 15
   ```

2. **Environment variables**:
   ```bash
   export ARANDU_QA_PROVIDER=openai
   export ARANDU_QA_MODEL_ID=gpt-4o-mini
   ```

3. **`.env` file** in project root:
   ```bash
   ARANDU_QA_PROVIDER=ollama
   ARANDU_QA_MODEL_ID=qwen3:14b
   ```

4. **Default values** in `config.py` (lowest priority):
   ```python
   provider: str = Field(default="ollama")
   model_id: str = Field(default="qwen3:14b")
   ```

**Result**: Command-line arguments override everything else. Environment variables override `.env` file and defaults.

---

## Best Practices

1. **Use `.env` for local development**
   - Easy to manage and version control (with .env.example)
   - Keep `.env` in `.gitignore`
   - Commit `.env.example` with safe defaults

2. **Use environment variables for production**
   - Better for CI/CD and containerized environments
   - Secrets management integration (e.g., Kubernetes Secrets)
   - No risk of committing sensitive data

3. **Use command-line arguments for one-off overrides**
   - Quick testing without changing configuration
   - Scripting and automation
   - Debugging specific settings

4. **Never commit API keys**
   - Always use environment variables or secrets management
   - Add `.env` to `.gitignore`
   - Use `.env.example` for documentation

5. **Document configuration changes**
   - Update `.env.example` when adding new settings
   - Add comments explaining non-obvious settings
   - Document valid values and ranges

6. **Validate configuration early**
   - Let Pydantic validation catch errors at startup
   - Use type hints and constraints
   - Write tests for custom validators

---

## Configuration File Template

**`.env.example`** (committed to git):
```bash
# ============================================================================
# Arandu Configuration Template
# Copy this file to .env and fill in your values
# ============================================================================

# ============================================================================
# TranscriberConfig (ARANDU_)
# ============================================================================

# Model settings
ARANDU_MODEL_ID=openai/whisper-large-v3
ARANDU_LANGUAGE=  # Optional: pt, en, etc. (auto-detect if not set)
ARANDU_CHUNK_LENGTH_S=30
ARANDU_STRIDE_LENGTH_S=5

# Hardware settings
ARANDU_FORCE_CPU=false
ARANDU_QUANTIZE=false

# Google Drive settings
ARANDU_CREDENTIALS=credentials.json
ARANDU_TOKEN=token.json

# Batch processing
ARANDU_WORKERS=2

# Paths
ARANDU_INPUT_DIR=./input
ARANDU_RESULTS_DIR=./results
ARANDU_HF_CACHE_DIR=./cache/huggingface

# Processing
ARANDU_MAX_RETRIES=3
ARANDU_RETRY_DELAY=1.0

# ============================================================================
# QAConfig (ARANDU_QA_)
# ============================================================================

# LLM Provider
ARANDU_QA_PROVIDER=ollama  # openai, ollama, custom
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
# ARANDU_QA_BASE_URL=  # For custom OpenAI-compatible endpoints

# Generation
ARANDU_QA_QUESTIONS_PER_DOCUMENT=10
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_MAX_TOKENS=2048

# Output
ARANDU_QA_OUTPUT_DIR=qa_dataset

# Language
ARANDU_QA_LANGUAGE=pt  # pt or en
ARANDU_QA_WORKERS=2

# ============================================================================
# CEPConfig (ARANDU_CEP_)
# ============================================================================

# Module toggles
ARANDU_CEP_ENABLE_REASONING_TRACES=true
ARANDU_CEP_ENABLE_VALIDATION=true

# Bloom scaffolding
ARANDU_CEP_BLOOM_LEVELS=remember,understand,analyze,evaluate
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=10

# Reasoning
ARANDU_CEP_MAX_HOP_COUNT=3

# LLM-as-a-Judge validation
ARANDU_CEP_VALIDATOR_PROVIDER=ollama
ARANDU_CEP_VALIDATOR_MODEL_ID=qwen3:14b
ARANDU_CEP_VALIDATOR_TEMPERATURE=0.3
ARANDU_CEP_VALIDATION_THRESHOLD=0.6

# Scoring weights (must sum to 1.0)
ARANDU_CEP_FAITHFULNESS_WEIGHT=0.4
ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT=0.3
ARANDU_CEP_INFORMATIVENESS_WEIGHT=0.3

# Language
ARANDU_CEP_LANGUAGE=pt

# ============================================================================
# KGConfig (ARANDU_KG_)
# ============================================================================

# LLM Provider
ARANDU_KG_PROVIDER=ollama  # openai, ollama, custom
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
# ARANDU_KG_BASE_URL=  # For custom OpenAI-compatible endpoints

# Graph settings
ARANDU_KG_MERGE_GRAPHS=true
ARANDU_KG_OUTPUT_FORMAT=graphml  # graphml or json
ARANDU_KG_SCHEMA_MODE=dynamic  # dynamic or predefined

# LLM settings
ARANDU_KG_TEMPERATURE=0.5

# Language and prompts
ARANDU_KG_LANGUAGE=pt
ARANDU_KG_PROMPT_PATH=prompts/pt_prompts.json

# Output
ARANDU_KG_OUTPUT_DIR=knowledge_graphs
ARANDU_KG_WORKERS=2

# ============================================================================
# EvaluationConfig (ARANDU_EVAL_)
# ============================================================================

# Metrics
ARANDU_EVAL_METRICS=qa,entity,relation,semantic
ARANDU_EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Output
ARANDU_EVAL_OUTPUT_DIR=evaluation

# Input directories (optional overrides)
# ARANDU_EVAL_QA_DIR=qa_dataset
# ARANDU_EVAL_KG_DIR=knowledge_graphs
# ARANDU_EVAL_RESULTS_DIR=results

# ============================================================================
# LLMConfig (No prefix - uses aliases)
# ============================================================================

# API Keys (SENSITIVE - do not commit)
# OPENAI_API_KEY=sk-...

# Custom endpoints
# ARANDU_LLM_BASE_URL=http://localhost:11434/v1

# ============================================================================
# ResultsConfig (ARANDU_RESULTS_)
# ============================================================================

ARANDU_RESULTS_BASE_DIR=./results
ARANDU_RESULTS_ENABLE_VERSIONING=true

# ============================================================================
# TranscriptionQualityConfig (ARANDU_QUALITY_)
# ============================================================================

# General
ARANDU_QUALITY_ENABLED=true
ARANDU_QUALITY_QUALITY_THRESHOLD=0.5
ARANDU_QUALITY_EXPECTED_LANGUAGE=pt

# Scoring weights (must sum to 1.0)
ARANDU_QUALITY_SCRIPT_MATCH_WEIGHT=0.35
ARANDU_QUALITY_REPETITION_WEIGHT=0.30
ARANDU_QUALITY_SEGMENT_QUALITY_WEIGHT=0.20
ARANDU_QUALITY_CONTENT_DENSITY_WEIGHT=0.15

# Thresholds (advanced - usually keep defaults)
# ARANDU_QUALITY_MAX_NON_LATIN_RATIO=0.1
# ARANDU_QUALITY_MAX_WORD_REPETITION_RATIO=0.15
# ARANDU_QUALITY_MAX_PHRASE_REPETITION_COUNT=4
# ARANDU_QUALITY_SUSPICIOUS_UNIFORM_INTERVALS=5
# ARANDU_QUALITY_MIN_WORDS_PER_MINUTE=30.0
# ARANDU_QUALITY_MAX_WORDS_PER_MINUTE=300.0
# ARANDU_QUALITY_MAX_EMPTY_SEGMENT_RATIO=0.2
# ARANDU_QUALITY_UNIFORM_INTERVAL_TOLERANCE=0.1
```

---

**Document Version**: 2.0  
**Last Updated**: 2025-01-24  
**Changes**: Complete rewrite to reflect actual implementation with 8 separate config classes
