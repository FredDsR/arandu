# Configuration Reference

Complete reference for all configuration settings in the Arandu pipeline.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [TranscriberConfig](#transcriberconfig)
3. [QAConfig](#qaconfig)
4. [CEPConfig](#cepconfig)
5. [JudgeConfig](#judgeconfig)
6. [KGConfig](#kgconfig)
7. [EvaluationConfig](#evaluationconfig)
8. [LLMConfig](#llmconfig)
9. [ResultsConfig](#resultsconfig)
10. [Environment Variables](#environment-variables)
11. [Configuration Examples](#configuration-examples)

---

## Configuration System Overview

The Arandu project uses **Pydantic Settings** for configuration management with hierarchical loading:

1. **Command-line arguments** (highest priority)
2. **Environment variables** with config-specific prefixes
3. **`.env` file** in project root
4. **Default values** in each config class (lowest priority)

**Configuration modules**: there is no flat `config.py`. Each config class lives next to the domain it configures:

| Config Class | Module | Prefix |
|--------------|--------|--------|
| `TranscriberConfig` | `arandu.transcription.config` | `ARANDU_` |
| `QAConfig` | `arandu.qa.config` | `ARANDU_QA_` |
| `CEPConfig` | `arandu.qa.config` | `ARANDU_CEP_` |
| `JudgeConfig` | `arandu.qa.config` | `ARANDU_JUDGE_` |
| `KGConfig` | `arandu.kg.config` | `ARANDU_KG_` |
| `EvaluationConfig` | `arandu.shared.config` | `ARANDU_EVAL_` |
| `LLMConfig` | `arandu.shared.config` | (no prefix; uses aliases like `OPENAI_API_KEY`) |
| `ResultsConfig` | `arandu.shared.config` | `ARANDU_RESULTS_` |

**Usage**:
```python
from arandu.transcription.config import TranscriberConfig
from arandu.qa.config import QAConfig

transcriber_config = TranscriberConfig()
qa_config = QAConfig()
print(transcriber_config.model_id)  # openai/whisper-large-v3
print(qa_config.provider)  # ollama
```

---

## TranscriberConfig

Configuration settings for the transcription pipeline.

**Module**: `arandu.transcription.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_`

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
| `temp_dir` | `str` | `"<tmp>/arandu"` (platform-specific) | Temporary directory for file processing |
| `max_retries` | `int` | `3` | Maximum number of retry attempts for failed operations |
| `retry_delay` | `float` | `1.0` | Delay in seconds between retry attempts |

**Example Configuration**:
```python
from arandu.transcription.config import TranscriberConfig

config = TranscriberConfig()
# Or with custom settings:
config = TranscriberConfig(
    model_id="openai/whisper-large-v3",
    force_cpu=False,
    workers=4,
)
```

---

## QAConfig

Configuration settings for the QA generation pipeline.

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_QA_`

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
| `temperature` | `float` | `0.7` | Temperature for QA generation LLM (range: 0.0-2.0) |
| `max_tokens` | `int` | `8192` | Max tokens for QA generation LLM. Sized for thinking models (Qwen3, Gemini 2.5) whose reasoning tokens consume the budget before the JSON output |

> **Note**: The per-chunk ladder size is not a QAConfig knob. It is the sum of `CEPConfig.bloom_distribution` (the integer pair counts per Bloom level), so the distribution is the single source of truth.

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
from arandu.qa.config import QAConfig

config = QAConfig(
    provider="ollama",
    model_id="qwen3:14b",
    language="pt",
)
```

---

## CEPConfig

Configuration settings for the CEP (Cognitive Elicitation Pipeline) with Bloom's Taxonomy scaffolding. Generation and validation are separate steps: `generate-cep-qa` only generates pairs; `judge-qa` validates them using the scoring weights and threshold defined here.

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_CEP_`

### Module Toggles

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_reasoning_traces` | `bool` | `True` | Enable reasoning trace generation for answers |
| `enable_scaffolding_context` | `bool` | `True` | Pass previously generated QA pairs as context to higher Bloom levels |
| `enable_source_metadata_context` | `bool` | `True` | Include extracted source metadata (participant name, location, date) in CEP prompt context |

### Module I - Bloom Scaffolding Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `bloom_distribution` | `dict[str, int]` | `{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}` | Absolute number of QA pairs to generate at each Bloom level, per chunk (single source of truth). Counts must be non-negative; the total must be between 1 and `MAX_BLOOM_PAIRS_PER_CHUNK` (50). Locked at 3/1/1/1 for the thesis run. |
| `max_scaffolding_pairs` | `int` | `10` | Max prior QA pairs to include as scaffolding context (min: 1, max: 50) |

**Valid Bloom Levels**: `remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`

**Derived property**: `pairs_per_chunk` returns the sum of `bloom_distribution` values (the per-chunk ladder size). There is no `bloom_levels` field; the keys of `bloom_distribution` are the levels to generate.

### Module II - Reasoning Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_hop_count` | `int` | `3` | Maximum reasoning hops to detect for multi-hop questions (min: 1, max: 5) |
| `reasoning_max_tokens` | `int` | `8192` | Max tokens for reasoning enrichment responses (min: 128, max: 8192). Sized for thinking models |

### Judge Scoring Settings

These fields drive `judge-qa` (the four-criterion LLM-as-a-Judge over generated pairs). The validator *client* settings (model, provider, base URL) live in [`JudgeConfig`](#judgeconfig).

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `validation_threshold` | `float` | `0.6` | Minimum overall score to pass validation (range: 0.0-1.0) |
| `faithfulness_weight` | `float` | `0.30` | Weight for faithfulness score |
| `bloom_calibration_weight` | `float` | `0.25` | Weight for Bloom calibration score |
| `informativeness_weight` | `float` | `0.25` | Weight for informativeness score |
| `self_containedness_weight` | `float` | `0.20` | Weight for self-containedness score |

> **Note**: The four scoring weights must sum to 1.0. A `@model_validator` (`validate_scoring_weights`) enforces this constraint.

### Language Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language for CEP prompts (ISO 639-1: 'pt' or 'en') |

**Example Configuration**:
```python
from arandu.qa.config import CEPConfig

config = CEPConfig(
    bloom_distribution={"remember": 3, "understand": 2, "analyze": 1},
    enable_scaffolding_context=True,
    validation_threshold=0.7,
)
print(config.pairs_per_chunk)  # 6
```

---

## JudgeConfig

Configuration for the composable judge pipeline. Supplies the validator LLM client settings shared by the `judge-transcription` LLM filter stage and the `judge-qa` command. Replaces the removed `TranscriptionQualityConfig` (transcription validation is now a filter pipeline, not a weighted-score config; see the [Transcription Validation guide](transcription-validation.md)).

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_JUDGE_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language for judge criterion prompts (ISO 639-1: 'pt' or 'en') |
| `temperature` | `float` | `0.3` | Temperature for judge LLM (low for consistent evaluation, range: 0.0-1.0) |
| `max_tokens` | `int` | `8192` | Max tokens for judge responses (min: 128, max: 8192) |
| `validator_model` | `str \| None` | `None` | Model ID enabling the LLM stage (e.g. `qwen3:14b`, `gemini-2.5-flash`). When unset, `judge-transcription` runs heuristic-only and skips the LLM stage |
| `validator_provider` | `str \| None` | `None` | Provider: "openai", "ollama", or "custom". Inferred from `ARANDU_LLM_BASE_URL` (custom when set, else ollama) when unspecified |
| `validator_base_url` | `str \| None` | `None` | Base URL for the validator provider. Inherits `ARANDU_LLM_BASE_URL` only when the resolved provider is "custom" |

**Example Configuration**:
```python
from arandu.qa.config import JudgeConfig

config = JudgeConfig()
# Loaded from ARANDU_JUDGE_* env vars; e.g.
# ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
# ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
```

---

## KGConfig

Configuration settings for the knowledge graph construction pipeline.

**Module**: `arandu.kg.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_KG_`

### Backend Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `str` | `"atlas"` | KGC backend: `"atlas"` (AutoSchemaKG). Must match pattern `^(atlas)$` |
| `backend_options` | `dict` | `{}` | Backend-specific options passed through to the constructor (e.g., `chunk_size`, `batch_size_triple`, `max_workers`) |

### LLM Provider Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "openai", "ollama", "custom" |
| `model_id` | `str` | `"llama3.1:8b"` | Model ID for KG construction |
| `ollama_url` | `str` | `"http://localhost:11434/v1"` | Ollama API base URL for KG construction |
| `base_url` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints |

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
from arandu.kg.config import KGConfig

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

**Module**: `arandu.shared.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_EVAL_`

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
from arandu.shared.config import EvaluationConfig

config = EvaluationConfig(
    metrics=["qa", "entity", "semantic"],
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
```

---

## LLMConfig

Shared LLM configuration settings for API keys and shared LLM settings across pipelines.

**Module**: `arandu.shared.config` &nbsp;|&nbsp; **Environment Prefix**: None (uses field aliases)

### API Keys

| Setting | Type | Default | Alias | Description |
|---------|------|---------|-------|-------------|
| `openai_api_key` | `str \| None` | `None` | `OPENAI_API_KEY` | OpenAI API key |
| `base_url` | `str \| None` | `None` | `ARANDU_LLM_BASE_URL` | Custom base URL for OpenAI-compatible endpoints |

**Example Configuration**:
```python
from arandu.shared.config import LLMConfig

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

**Module**: `arandu.shared.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_RESULTS_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_dir` | `Path` | `Path("./results")` | Base directory for versioned results |
| `enable_versioning` | `bool` | `True` | Enable versioned result directories |

**Example Configuration**:
```python
from pathlib import Path

from arandu.shared.config import ResultsConfig

config = ResultsConfig(
    base_dir=Path("/data/results"),
    enable_versioning=True,
)
```

---

## Environment Variables

Configuration settings are loaded from environment variables with config-specific prefixes.

### Prefix Reference

| Config Class | Prefix | Example |
|--------------|--------|---------|
| `TranscriberConfig` | `ARANDU_` | `ARANDU_MODEL_ID` |
| `QAConfig` | `ARANDU_QA_` | `ARANDU_QA_PROVIDER` |
| `CEPConfig` | `ARANDU_CEP_` | `ARANDU_CEP_VALIDATION_THRESHOLD` |
| `JudgeConfig` | `ARANDU_JUDGE_` | `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `KGConfig` | `ARANDU_KG_` | `ARANDU_KG_PROVIDER` |
| `EvaluationConfig` | `ARANDU_EVAL_` | `ARANDU_EVAL_METRICS` |
| `LLMConfig` | (No prefix) | `OPENAI_API_KEY`, `ARANDU_LLM_BASE_URL` |
| `ResultsConfig` | `ARANDU_RESULTS_` | `ARANDU_RESULTS_BASE_DIR` |

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
export ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
export ARANDU_QA_LANGUAGE=pt
```

**CEPConfig** (`ARANDU_CEP_`):
```bash
export ARANDU_CEP_VALIDATION_THRESHOLD=0.7
export ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=15
export ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT=true
# Bloom distribution is a dict; set it as JSON
export ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'
```

**JudgeConfig** (`ARANDU_JUDGE_`):
```bash
export ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
export ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
export ARANDU_JUDGE_TEMPERATURE=0.3
```

**KGConfig** (`ARANDU_KG_`):
```bash
export ARANDU_KG_PROVIDER=openai
export ARANDU_KG_MODEL_ID=gpt-4o
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
ARANDU_QA_LANGUAGE=pt

# CEP (Cognitive Elicitation Pipeline)
ARANDU_CEP_VALIDATION_THRESHOLD=0.6

# Judge (validator client for judge-qa / judge-transcription)
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama

# KG Construction
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
ARANDU_KG_LANGUAGE=pt

# Evaluation
ARANDU_EVAL_METRICS=qa,entity,relation,semantic
```

**CLI Usage**:
```bash
# QA generation (uses .env settings)
arandu generate-cep-qa results/

# Validate the generated pairs (separate step)
arandu judge-qa qa_dataset/

# Override specific settings
arandu generate-cep-qa results/ --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1" --provider ollama
```

### Example 2: Production with OpenAI API

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA Generation with OpenAI
ARANDU_QA_PROVIDER=openai
ARANDU_QA_MODEL_ID=gpt-4o-mini
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_LANGUAGE=en

# Judge with OpenAI
ARANDU_JUDGE_VALIDATOR_PROVIDER=openai
ARANDU_JUDGE_VALIDATOR_MODEL=gpt-4o-mini
ARANDU_CEP_VALIDATION_THRESHOLD=0.7

# KG Construction with OpenAI
ARANDU_KG_PROVIDER=openai
ARANDU_KG_MODEL_ID=gpt-4o
ARANDU_KG_TEMPERATURE=0.5

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
services:
  arandu-qa:
    environment:
      - ARANDU_QA_PROVIDER=ollama
      - ARANDU_QA_MODEL_ID=qwen3:14b
      - ARANDU_QA_OLLAMA_URL=http://host.docker.internal:11434/v1
      - ARANDU_QA_WORKERS=4
    volumes:
      - ./results:/app/results:ro
      - ./qa_dataset:/app/qa_dataset:rw

  arandu-kg:
    environment:
      - ARANDU_KG_PROVIDER=ollama
      - ARANDU_KG_MODEL_ID=llama3.1:8b
      - ARANDU_KG_OLLAMA_URL=http://host.docker.internal:11434/v1
    volumes:
      - ./results:/app/results:ro
      - ./knowledge_graphs:/app/knowledge_graphs:rw
```

### Example 6: CEP with Advanced Bloom Scaffolding

**.env file**:
```bash
# CEP Settings
ARANDU_CEP_ENABLE_REASONING_TRACES=true
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=15

# Bloom distribution (integer pair counts per level; single source of truth)
ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'

# Judge scoring (used by judge-qa)
ARANDU_CEP_VALIDATION_THRESHOLD=0.7
ARANDU_CEP_FAITHFULNESS_WEIGHT=0.30
ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT=0.25
ARANDU_CEP_INFORMATIVENESS_WEIGHT=0.25
ARANDU_CEP_SELF_CONTAINEDNESS_WEIGHT=0.20

# Validator client
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_TEMPERATURE=0.3
```

---

## Configuration Validation

The configuration system includes validation rules enforced by Pydantic.

### Custom Field Validation

```python
# From CEPConfig — counts must be non-negative, reference valid Bloom levels,
# and total between 1 and MAX_BLOOM_PAIRS_PER_CHUNK (50)
@field_validator("bloom_distribution")
@classmethod
def validate_bloom_distribution(cls, v: dict[str, int]) -> dict[str, int]:
    ...  # each count >= 0, valid level, 1 <= sum <= 50
```

### Pattern Validation

```python
# From KGConfig
backend: str = Field(
    default="atlas",
    pattern="^(atlas)$",  # only the atlas backend is supported
)
```

### Model Validation (Cross-Field)

```python
# From CEPConfig — the four judge scoring weights must sum to 1.0
@model_validator(mode="after")
def validate_scoring_weights(self) -> CEPConfig:
    total = (
        self.faithfulness_weight
        + self.bloom_calibration_weight
        + self.informativeness_weight
        + self.self_containedness_weight
    )
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")
    return self
```

---

## Configuration Loading Order

Pydantic Settings loads configuration in the following priority order (highest to lowest):

1. **Command-line arguments** (highest priority):
   ```bash
   arandu generate-cep-qa results/ --provider ollama --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1"
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

4. **Default values** in each config class (lowest priority):
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
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_MAX_TOKENS=8192

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
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT=true

# Bloom scaffolding (integer pair counts per level; single source of truth)
ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=10

# Reasoning
ARANDU_CEP_MAX_HOP_COUNT=3

# Judge scoring (used by judge-qa; weights must sum to 1.0)
ARANDU_CEP_VALIDATION_THRESHOLD=0.6
ARANDU_CEP_FAITHFULNESS_WEIGHT=0.30
ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT=0.25
ARANDU_CEP_INFORMATIVENESS_WEIGHT=0.25
ARANDU_CEP_SELF_CONTAINEDNESS_WEIGHT=0.20

# Language
ARANDU_CEP_LANGUAGE=pt

# ============================================================================
# JudgeConfig (ARANDU_JUDGE_) - validator client for judge-qa / judge-transcription
# ============================================================================

ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama  # openai, ollama, custom
# ARANDU_JUDGE_VALIDATOR_BASE_URL=  # For custom OpenAI-compatible endpoints
ARANDU_JUDGE_TEMPERATURE=0.3
ARANDU_JUDGE_LANGUAGE=pt

# ============================================================================
# KGConfig (ARANDU_KG_)
# ============================================================================

# Backend
ARANDU_KG_BACKEND=atlas

# LLM Provider
ARANDU_KG_PROVIDER=ollama  # openai, ollama, custom
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
# ARANDU_KG_BASE_URL=  # For custom OpenAI-compatible endpoints

# LLM settings
ARANDU_KG_TEMPERATURE=0.5

# Language and prompts
ARANDU_KG_LANGUAGE=pt

# Output
ARANDU_KG_OUTPUT_DIR=knowledge_graphs

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
```

---

**Document Version**: 3.0  
**Last Updated**: 2026-06-16  
**Changes**: Synced to the per-domain config modules (no flat `config.py`); replaced the removed `TranscriptionQualityConfig` with `JudgeConfig`; CEP now uses `bloom_distribution` (no `bloom_levels`/`enable_validation`) with four judge weights; removed nonexistent KG `merge_graphs`/`output_format`/`schema_mode`/`prompt_path` fields.
