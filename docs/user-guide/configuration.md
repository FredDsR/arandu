# Configuration Reference

Complete reference for all configuration settings in the G-Transcriber pipeline.

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

The G-Transcriber project uses **Pydantic Settings** for configuration management with hierarchical loading:

1. **Command-line arguments** (highest priority)
2. **Environment variables** with config-specific prefixes
3. **`.env` file** in project root
4. **Default values** in `config.py` (lowest priority)

**Configuration File**: `src/gtranscriber/config.py`

**Architecture**: The system uses **8 separate configuration classes**, each with its own environment variable prefix:
- `TranscriberConfig` - Prefix: `GTRANSCRIBER_`
- `QAConfig` - Prefix: `GTRANSCRIBER_QA_`
- `CEPConfig` - Prefix: `GTRANSCRIBER_CEP_`
- `KGConfig` - Prefix: `GTRANSCRIBER_KG_`
- `EvaluationConfig` - Prefix: `GTRANSCRIBER_EVAL_`
- `LLMConfig` - No prefix (uses aliases like `OPENAI_API_KEY`)
- `ResultsConfig` - Prefix: `GTRANSCRIBER_RESULTS_`
- `TranscriptionQualityConfig` - Prefix: `GTRANSCRIBER_QUALITY_`

**Usage**:
```python
from gtranscriber.config import TranscriberConfig, QAConfig

transcriber_config = TranscriberConfig()
qa_config = QAConfig()
print(transcriber_config.model_id)  # openai/whisper-large-v3
print(qa_config.provider)  # ollama
```

---

## TranscriberConfig

Configuration settings for the transcription pipeline.

**Environment Prefix**: `GTRANSCRIBER_`

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
| `temp_dir` | `str` | `"/tmp/gtranscriber"` (platform-specific) | Temporary directory for file processing |
| `max_retries` | `int` | `3` | Maximum number of retry attempts for failed operations |
| `retry_delay` | `float` | `1.0` | Delay in seconds between retry attempts |

**Example Configuration**:
```python
from gtranscriber.config import TranscriberConfig

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

**Environment Prefix**: `GTRANSCRIBER_QA_`

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
from gtranscriber.config import QAConfig

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

**Environment Prefix**: `GTRANSCRIBER_CEP_`

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
from gtranscriber.config import CEPConfig

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

**Environment Prefix**: `GTRANSCRIBER_KG_`

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
from gtranscriber.config import KGConfig

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

**Environment Prefix**: `GTRANSCRIBER_EVAL_`

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
from gtranscriber.config import EvaluationConfig

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
| `base_url` | `str \| None` | `None` | `GTRANSCRIBER_LLM_BASE_URL` | Custom base URL for OpenAI-compatible endpoints |

**Example Configuration**:
```python
from gtranscriber.config import LLMConfig

config = LLMConfig()
# Loaded from OPENAI_API_KEY and GTRANSCRIBER_LLM_BASE_URL env vars
```

**Environment Variables**:
```bash
export OPENAI_API_KEY=sk-...
export GTRANSCRIBER_LLM_BASE_URL=https://my-custom-endpoint/v1
```

---

## ResultsConfig

Configuration for versioned results management.

**Environment Prefix**: `GTRANSCRIBER_RESULTS_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_dir` | `Path` | `Path("./results")` | Base directory for versioned results |
| `enable_versioning` | `bool` | `True` | Enable versioned result directories |

**Example Configuration**:
```python
from gtranscriber.config import ResultsConfig

config = ResultsConfig(
    base_dir=Path("/data/results"),
    enable_versioning=True
)
```

---

## TranscriptionQualityConfig

Configuration for transcription quality validation with heuristic quality checks.

**Environment Prefix**: `GTRANSCRIBER_QUALITY_`

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
from gtranscriber.config import TranscriptionQualityConfig

config = TranscriptionQualityConfig(
    enabled=True,
    quality_threshold=0.6,
    expected_language="pt"
)
```

**See also**: [Transcription Validation Guide](transcription-validation.md) for full usage details

---

## Environment Variables

Configuration settings are loaded from environment variables with config-specific prefixes.

### Prefix Reference

| Config Class | Prefix | Example |
|--------------|--------|---------|
| `TranscriberConfig` | `GTRANSCRIBER_` | `GTRANSCRIBER_MODEL_ID` |
| `QAConfig` | `GTRANSCRIBER_QA_` | `GTRANSCRIBER_QA_PROVIDER` |
| `CEPConfig` | `GTRANSCRIBER_CEP_` | `GTRANSCRIBER_CEP_ENABLE_VALIDATION` |
| `KGConfig` | `GTRANSCRIBER_KG_` | `GTRANSCRIBER_KG_PROVIDER` |
| `EvaluationConfig` | `GTRANSCRIBER_EVAL_` | `GTRANSCRIBER_EVAL_METRICS` |
| `LLMConfig` | (No prefix) | `OPENAI_API_KEY`, `GTRANSCRIBER_LLM_BASE_URL` |
| `ResultsConfig` | `GTRANSCRIBER_RESULTS_` | `GTRANSCRIBER_RESULTS_BASE_DIR` |
| `TranscriptionQualityConfig` | `GTRANSCRIBER_QUALITY_` | `GTRANSCRIBER_QUALITY_ENABLED` |

### Format

```bash
<PREFIX><SETTING_NAME>=<value>
```

### Examples by Config Class

**TranscriberConfig** (`GTRANSCRIBER_`):
```bash
export GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3
export GTRANSCRIBER_FORCE_CPU=false
export GTRANSCRIBER_WORKERS=4
export GTRANSCRIBER_RETRY_DELAY=1.0
```

**QAConfig** (`GTRANSCRIBER_QA_`):
```bash
export GTRANSCRIBER_QA_PROVIDER=openai
export GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=15
export GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1
export GTRANSCRIBER_QA_LANGUAGE=pt
```

**CEPConfig** (`GTRANSCRIBER_CEP_`):
```bash
export GTRANSCRIBER_CEP_ENABLE_VALIDATION=true
export GTRANSCRIBER_CEP_BLOOM_LEVELS=remember,understand,analyze
export GTRANSCRIBER_CEP_VALIDATION_THRESHOLD=0.7
export GTRANSCRIBER_CEP_VALIDATOR_PROVIDER=ollama
```

**KGConfig** (`GTRANSCRIBER_KG_`):
```bash
export GTRANSCRIBER_KG_PROVIDER=openai
export GTRANSCRIBER_KG_MODEL_ID=gpt-4o
export GTRANSCRIBER_KG_MERGE_GRAPHS=true
export GTRANSCRIBER_KG_LANGUAGE=pt
export GTRANSCRIBER_KG_OLLAMA_URL=http://localhost:11434/v1
```

**EvaluationConfig** (`GTRANSCRIBER_EVAL_`):
```bash
export GTRANSCRIBER_EVAL_METRICS=qa,entity,relation,semantic
export GTRANSCRIBER_EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**LLMConfig** (No prefix, uses aliases):
```bash
export OPENAI_API_KEY=sk-...
export GTRANSCRIBER_LLM_BASE_URL=https://my-custom-endpoint/v1
```

**ResultsConfig** (`GTRANSCRIBER_RESULTS_`):
```bash
export GTRANSCRIBER_RESULTS_BASE_DIR=/data/results
export GTRANSCRIBER_RESULTS_ENABLE_VERSIONING=true
```

**TranscriptionQualityConfig** (`GTRANSCRIBER_QUALITY_`):
```bash
export GTRANSCRIBER_QUALITY_ENABLED=true
export GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD=0.6
export GTRANSCRIBER_QUALITY_EXPECTED_LANGUAGE=pt
```

### API Keys

**Sensitive values should be set as environment variables only** (never commit to git):

```bash
export OPENAI_API_KEY=sk-...
export GTRANSCRIBER_LLM_BASE_URL=https://my-custom-endpoint/v1
```

These can also be set in `.env` file:
```bash
# .env file
OPENAI_API_KEY=sk-...
GTRANSCRIBER_LLM_BASE_URL=https://my-custom-endpoint/v1
```

**Note**: The `.env` file should be added to `.gitignore`.

---

## Configuration Examples

### Example 1: Local Development with Ollama

**.env file**:
```bash
# Transcription
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3
GTRANSCRIBER_WORKERS=2
GTRANSCRIBER_RETRY_DELAY=1.0

# QA Generation
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=10
GTRANSCRIBER_QA_LANGUAGE=pt

# CEP (Cognitive Elicitation Pipeline)
GTRANSCRIBER_CEP_ENABLE_VALIDATION=true
GTRANSCRIBER_CEP_VALIDATION_THRESHOLD=0.6

# KG Construction
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_OLLAMA_URL=http://localhost:11434/v1
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=graphml
GTRANSCRIBER_KG_LANGUAGE=pt

# Evaluation
GTRANSCRIBER_EVAL_METRICS=qa,entity,relation,semantic

# Transcription Quality Validation
GTRANSCRIBER_QUALITY_ENABLED=true
GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD=0.5
```

**CLI Usage**:
```bash
# QA generation (uses .env settings)
gtranscriber generate-cep-qa results/

# Override specific settings
gtranscriber generate-cep-qa results/ --questions 15 --provider ollama
```

### Example 2: Production with OpenAI API

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA Generation with OpenAI
GTRANSCRIBER_QA_PROVIDER=openai
GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=12
GTRANSCRIBER_QA_TEMPERATURE=0.7
GTRANSCRIBER_QA_LANGUAGE=en

# CEP with OpenAI
GTRANSCRIBER_CEP_ENABLE_VALIDATION=true
GTRANSCRIBER_CEP_VALIDATOR_PROVIDER=openai
GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=gpt-4o-mini
GTRANSCRIBER_CEP_VALIDATION_THRESHOLD=0.7

# KG Construction with OpenAI
GTRANSCRIBER_KG_PROVIDER=openai
GTRANSCRIBER_KG_MODEL_ID=gpt-4o
GTRANSCRIBER_KG_TEMPERATURE=0.5
GTRANSCRIBER_KG_MERGE_GRAPHS=true

# Results versioning
GTRANSCRIBER_RESULTS_BASE_DIR=/data/transcriptions
GTRANSCRIBER_RESULTS_ENABLE_VERSIONING=true

# Output directories
GTRANSCRIBER_QA_OUTPUT_DIR=/data/qa_dataset
GTRANSCRIBER_KG_OUTPUT_DIR=/data/knowledge_graphs
GTRANSCRIBER_EVAL_OUTPUT_DIR=/data/evaluation
```

### Example 3: Hybrid Approach (OpenAI + Ollama)

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA with OpenAI (higher quality)
GTRANSCRIBER_QA_PROVIDER=openai
GTRANSCRIBER_QA_MODEL_ID=gpt-4o
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=15

# KG with Ollama (cost-effective)
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_OLLAMA_URL=http://localhost:11434/v1

# Workers
GTRANSCRIBER_WORKERS=4
```

### Example 4: SLURM Cluster Configuration

**SLURM script** (`run_qa_generation.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=grace
#SBATCH --cpus-per-task=16

# Set configuration via environment
export GTRANSCRIBER_QA_PROVIDER=ollama
export GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
export GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1
export GTRANSCRIBER_QA_WORKERS=8
export GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=12

# Use $SCRATCH for I/O
export GTRANSCRIBER_RESULTS_BASE_DIR=$SCRATCH/results
export GTRANSCRIBER_QA_OUTPUT_DIR=$SCRATCH/qa_dataset

# Run via Docker
source scripts/slurm/job_common.sh
docker compose --profile qa up gtranscriber-qa --abort-on-container-exit
```

### Example 5: Docker Compose Environment

**docker-compose.override.yml** (local overrides):
```yaml
version: '3.8'

services:
  gtranscriber-qa:
    environment:
      - GTRANSCRIBER_QA_PROVIDER=ollama
      - GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
      - GTRANSCRIBER_QA_OLLAMA_URL=http://host.docker.internal:11434/v1
      - GTRANSCRIBER_QA_WORKERS=4
      - GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=10
    volumes:
      - ./results:/app/results:ro
      - ./qa_dataset:/app/qa_dataset:rw

  gtranscriber-kg:
    environment:
      - GTRANSCRIBER_KG_PROVIDER=ollama
      - GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
      - GTRANSCRIBER_KG_OLLAMA_URL=http://host.docker.internal:11434/v1
      - GTRANSCRIBER_KG_MERGE_GRAPHS=true
      - GTRANSCRIBER_KG_WORKERS=2
    volumes:
      - ./results:/app/results:ro
      - ./knowledge_graphs:/app/knowledge_graphs:rw
```

### Example 6: CEP with Advanced Bloom Scaffolding

**.env file**:
```bash
# CEP Settings
GTRANSCRIBER_CEP_ENABLE_REASONING_TRACES=true
GTRANSCRIBER_CEP_ENABLE_VALIDATION=true
GTRANSCRIBER_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
GTRANSCRIBER_CEP_MAX_SCAFFOLDING_PAIRS=15

# Bloom Levels (custom distribution)
GTRANSCRIBER_CEP_BLOOM_LEVELS=remember,understand,analyze,evaluate

# LLM-as-a-Judge Validation
GTRANSCRIBER_CEP_VALIDATOR_PROVIDER=ollama
GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=qwen3:14b
GTRANSCRIBER_CEP_VALIDATOR_TEMPERATURE=0.3
GTRANSCRIBER_CEP_VALIDATION_THRESHOLD=0.7

# Scoring weights (must sum to 1.0)
GTRANSCRIBER_CEP_FAITHFULNESS_WEIGHT=0.4
GTRANSCRIBER_CEP_BLOOM_CALIBRATION_WEIGHT=0.3
GTRANSCRIBER_CEP_INFORMATIVENESS_WEIGHT=0.3
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
   gtranscriber generate-cep-qa results/ --provider ollama --questions 15
   ```

2. **Environment variables**:
   ```bash
   export GTRANSCRIBER_QA_PROVIDER=openai
   export GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
   ```

3. **`.env` file** in project root:
   ```bash
   GTRANSCRIBER_QA_PROVIDER=ollama
   GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
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
# G-Transcriber Configuration Template
# Copy this file to .env and fill in your values
# ============================================================================

# ============================================================================
# TranscriberConfig (GTRANSCRIBER_)
# ============================================================================

# Model settings
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3
GTRANSCRIBER_LANGUAGE=  # Optional: pt, en, etc. (auto-detect if not set)
GTRANSCRIBER_CHUNK_LENGTH_S=30
GTRANSCRIBER_STRIDE_LENGTH_S=5

# Hardware settings
GTRANSCRIBER_FORCE_CPU=false
GTRANSCRIBER_QUANTIZE=false

# Google Drive settings
GTRANSCRIBER_CREDENTIALS=credentials.json
GTRANSCRIBER_TOKEN=token.json

# Batch processing
GTRANSCRIBER_WORKERS=2

# Paths
GTRANSCRIBER_INPUT_DIR=./input
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_HF_CACHE_DIR=./cache/huggingface

# Processing
GTRANSCRIBER_MAX_RETRIES=3
GTRANSCRIBER_RETRY_DELAY=1.0

# ============================================================================
# QAConfig (GTRANSCRIBER_QA_)
# ============================================================================

# LLM Provider
GTRANSCRIBER_QA_PROVIDER=ollama  # openai, ollama, custom
GTRANSCRIBER_QA_MODEL_ID=qwen3:14b
GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1
# GTRANSCRIBER_QA_BASE_URL=  # For custom OpenAI-compatible endpoints

# Generation
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=10
GTRANSCRIBER_QA_TEMPERATURE=0.7
GTRANSCRIBER_QA_MAX_TOKENS=2048

# Output
GTRANSCRIBER_QA_OUTPUT_DIR=qa_dataset

# Language
GTRANSCRIBER_QA_LANGUAGE=pt  # pt or en
GTRANSCRIBER_QA_WORKERS=2

# ============================================================================
# CEPConfig (GTRANSCRIBER_CEP_)
# ============================================================================

# Module toggles
GTRANSCRIBER_CEP_ENABLE_REASONING_TRACES=true
GTRANSCRIBER_CEP_ENABLE_VALIDATION=true

# Bloom scaffolding
GTRANSCRIBER_CEP_BLOOM_LEVELS=remember,understand,analyze,evaluate
GTRANSCRIBER_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
GTRANSCRIBER_CEP_MAX_SCAFFOLDING_PAIRS=10

# Reasoning
GTRANSCRIBER_CEP_MAX_HOP_COUNT=3

# LLM-as-a-Judge validation
GTRANSCRIBER_CEP_VALIDATOR_PROVIDER=ollama
GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID=qwen3:14b
GTRANSCRIBER_CEP_VALIDATOR_TEMPERATURE=0.3
GTRANSCRIBER_CEP_VALIDATION_THRESHOLD=0.6

# Scoring weights (must sum to 1.0)
GTRANSCRIBER_CEP_FAITHFULNESS_WEIGHT=0.4
GTRANSCRIBER_CEP_BLOOM_CALIBRATION_WEIGHT=0.3
GTRANSCRIBER_CEP_INFORMATIVENESS_WEIGHT=0.3

# Language
GTRANSCRIBER_CEP_LANGUAGE=pt

# ============================================================================
# KGConfig (GTRANSCRIBER_KG_)
# ============================================================================

# LLM Provider
GTRANSCRIBER_KG_PROVIDER=ollama  # openai, ollama, custom
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_OLLAMA_URL=http://localhost:11434/v1
# GTRANSCRIBER_KG_BASE_URL=  # For custom OpenAI-compatible endpoints

# Graph settings
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=graphml  # graphml or json
GTRANSCRIBER_KG_SCHEMA_MODE=dynamic  # dynamic or predefined

# LLM settings
GTRANSCRIBER_KG_TEMPERATURE=0.5

# Language and prompts
GTRANSCRIBER_KG_LANGUAGE=pt
GTRANSCRIBER_KG_PROMPT_PATH=prompts/pt_prompts.json

# Output
GTRANSCRIBER_KG_OUTPUT_DIR=knowledge_graphs
GTRANSCRIBER_KG_WORKERS=2

# ============================================================================
# EvaluationConfig (GTRANSCRIBER_EVAL_)
# ============================================================================

# Metrics
GTRANSCRIBER_EVAL_METRICS=qa,entity,relation,semantic
GTRANSCRIBER_EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Output
GTRANSCRIBER_EVAL_OUTPUT_DIR=evaluation

# Input directories (optional overrides)
# GTRANSCRIBER_EVAL_QA_DIR=qa_dataset
# GTRANSCRIBER_EVAL_KG_DIR=knowledge_graphs
# GTRANSCRIBER_EVAL_RESULTS_DIR=results

# ============================================================================
# LLMConfig (No prefix - uses aliases)
# ============================================================================

# API Keys (SENSITIVE - do not commit)
# OPENAI_API_KEY=sk-...

# Custom endpoints
# GTRANSCRIBER_LLM_BASE_URL=http://localhost:11434/v1

# ============================================================================
# ResultsConfig (GTRANSCRIBER_RESULTS_)
# ============================================================================

GTRANSCRIBER_RESULTS_BASE_DIR=./results
GTRANSCRIBER_RESULTS_ENABLE_VERSIONING=true

# ============================================================================
# TranscriptionQualityConfig (GTRANSCRIBER_QUALITY_)
# ============================================================================

# General
GTRANSCRIBER_QUALITY_ENABLED=true
GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD=0.5
GTRANSCRIBER_QUALITY_EXPECTED_LANGUAGE=pt

# Scoring weights (must sum to 1.0)
GTRANSCRIBER_QUALITY_SCRIPT_MATCH_WEIGHT=0.35
GTRANSCRIBER_QUALITY_REPETITION_WEIGHT=0.30
GTRANSCRIBER_QUALITY_SEGMENT_QUALITY_WEIGHT=0.20
GTRANSCRIBER_QUALITY_CONTENT_DENSITY_WEIGHT=0.15

# Thresholds (advanced - usually keep defaults)
# GTRANSCRIBER_QUALITY_MAX_NON_LATIN_RATIO=0.1
# GTRANSCRIBER_QUALITY_MAX_WORD_REPETITION_RATIO=0.15
# GTRANSCRIBER_QUALITY_MAX_PHRASE_REPETITION_COUNT=4
# GTRANSCRIBER_QUALITY_SUSPICIOUS_UNIFORM_INTERVALS=5
# GTRANSCRIBER_QUALITY_MIN_WORDS_PER_MINUTE=30.0
# GTRANSCRIBER_QUALITY_MAX_WORDS_PER_MINUTE=300.0
# GTRANSCRIBER_QUALITY_MAX_EMPTY_SEGMENT_RATIO=0.2
# GTRANSCRIBER_QUALITY_UNIFORM_INTERVAL_TOLERANCE=0.1
```

---

**Document Version**: 2.0  
**Last Updated**: 2025-01-24  
**Changes**: Complete rewrite to reflect actual implementation with 8 separate config classes
