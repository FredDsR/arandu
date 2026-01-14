# Configuration Reference

Complete reference for all configuration settings in the Knowledge Graph Construction Pipeline.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [Existing Configuration](#existing-configuration)
3. [New Configuration Settings](#new-configuration-settings)
4. [Environment Variables](#environment-variables)
5. [Configuration Examples](#configuration-examples)

---

## Configuration System Overview

The G-Transcriber project uses **Pydantic Settings** for configuration management with hierarchical loading:

1. **Command-line arguments** (highest priority)
2. **Environment variables** with `GTRANSCRIBER_` prefix
3. **`.env` file** in project root
4. **Default values** in `config.py` (lowest priority)

**Configuration File**: `src/gtranscriber/config.py`

**Usage**:
```python
from gtranscriber.config import TranscriberConfig

config = TranscriberConfig()
print(config.qa_provider)  # Access settings
```

---

## Existing Configuration

These settings were already present in the transcription pipeline:

### Transcription Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_id` | `str` | `"openai/whisper-large-v3-turbo"` | Whisper model from Hugging Face |
| `force_cpu` | `bool` | `False` | Force CPU execution (no GPU) |
| `quantize` | `bool` | `False` | Enable 8-bit quantization |
| `chunk_length_s` | `int` | `30` | Audio chunk length in seconds |
| `stride_length_s` | `int` | `5` | Stride length for chunks |
| `batch_size` | `int` | `24` | Batch size for inference |

### Google Drive Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `credentials_file` | `Path` | `Path("credentials.json")` | OAuth2 credentials path |
| `token_file` | `Path` | `Path("token.json")` | OAuth2 token cache path |
| `max_retries` | `int` | `3` | Max retries for API calls |
| `retry_delay` | `float` | `2.0` | Delay between retries (seconds) |

### Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `workers` | `int` | `1` | Number of parallel workers |
| `temp_dir` | `Path` | `Path("/tmp/gtranscriber")` | Temporary file directory |

### Directory Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `input_dir` | `Path` | `Path("input")` | Input directory for catalogs |
| `results_dir` | `Path` | `Path("results")` | Output directory for results |
| `cache_dir` | `Path` | `Path(".cache/huggingface")` | HuggingFace cache directory |

---

## New Configuration Settings

### QA Generation Settings

Settings for synthetic QA dataset generation.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `qa_provider` | `str` | `"ollama"` | LLM provider: "openai", "anthropic", "ollama" |
| `qa_model_id` | `str` | `"llama3.1:8b"` | Model ID for QA generation |
| `qa_ollama_url` | `str` | `"http://localhost:11434"` | Ollama API base URL |
| `openai_api_key` | `str \| None` | `None` | OpenAI API key (from env) |
| `anthropic_api_key` | `str \| None` | `None` | Anthropic API key (from env) |
| `questions_per_document` | `int` | `10` | Number of QA pairs to generate per document |
| `qa_strategies` | `list[str]` | `["factual", "conceptual"]` | Question generation strategies |
| `qa_temperature` | `float` | `0.7` | Temperature for LLM generation |
| `qa_max_tokens` | `int` | `2048` | Max tokens for LLM generation |
| `qa_output_dir` | `Path` | `Path("qa_dataset")` | Output directory for QA datasets |

**Strategy Options**:
- `"factual"` - Who, what, when, where questions
- `"conceptual"` - Why, how questions
- `"temporal"` - Time-based questions
- `"entity"` - Entity-focused questions

**Example Configuration**:
```python
# In config.py
qa_provider: str = Field(
    default="ollama",
    description="LLM provider for QA generation"
)
qa_model_id: str = Field(
    default="llama3.1:8b",
    description="Model for QA generation"
)
questions_per_document: int = Field(
    default=10,
    ge=1,
    le=50,
    description="QA pairs per document"
)
```

### Knowledge Graph Construction Settings

Settings for KG construction using AutoSchemaKG.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `kg_provider` | `str` | `"ollama"` | LLM provider for entity/relation extraction |
| `kg_model_id` | `str` | `"llama3.1:8b"` | Model ID for KG construction |
| `kg_ollama_url` | `str` | `"http://localhost:11434"` | Ollama API base URL |
| `kg_merge_graphs` | `bool` | `True` | Merge individual graphs into corpus-level graph |
| `kg_output_format` | `str` | `"json"` | Export format: "json" or "graphml" |
| `kg_schema_mode` | `str` | `"dynamic"` | Schema mode: "dynamic" or "predefined" |
| `kg_temperature` | `float` | `0.5` | Temperature for LLM (lower = more consistent) |
| `kg_output_dir` | `Path` | `Path("knowledge_graphs")` | Output directory for KGs |

**Schema Modes**:
- `"dynamic"` - AutoSchemaKG infers schema from data (recommended)
- `"predefined"` - Use fixed schema (requires schema definition)

**Example Configuration**:
```python
# In config.py
kg_provider: str = Field(
    default="ollama",
    description="LLM provider for KG construction"
)
kg_merge_graphs: bool = Field(
    default=True,
    description="Merge individual graphs into corpus graph"
)
kg_output_format: str = Field(
    default="json",
    pattern="^(json|graphml)$",
    description="Graph export format"
)
```

### Evaluation Settings

Settings for knowledge elicitation metrics.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `evaluation_metrics` | `list[str]` | `["qa", "entity", "relation", "semantic"]` | Metrics to compute |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Model for semantic embeddings |
| `evaluation_output_dir` | `Path` | `Path("evaluation")` | Output directory for reports |

**Metric Options**:
- `"qa"` - QA-based metrics (EM, F1, BLEU)
- `"entity"` - Entity coverage metrics
- `"relation"` - Relation density metrics
- `"semantic"` - Semantic quality metrics

**Example Configuration**:
```python
# In config.py
evaluation_metrics: list[str] = Field(
    default=["qa", "entity", "relation", "semantic"],
    description="Metrics to compute during evaluation"
)
embedding_model: str = Field(
    default="sentence-transformers/all-MiniLM-L6-v2",
    description="Sentence transformer for embeddings"
)
```

### GraphRAG Settings (Future)

Settings for GraphRAG system (P3 Task 5).

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `graphrag_embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model for indexing |
| `graphrag_chunk_size` | `int` | `512` | Chunk size for text splitting |
| `graphrag_community_detection` | `str` | `"leiden"` | Algorithm: "leiden" or "louvain" |
| `graphrag_index_dir` | `Path` | `Path("graphrag_index")` | Index directory |

---

## Environment Variables

All configuration settings can be overridden via environment variables with the `GTRANSCRIBER_` prefix.

### Format

```bash
GTRANSCRIBER_<SETTING_NAME>=<value>
```

**Examples**:
```bash
# QA Generation
export GTRANSCRIBER_QA_PROVIDER=openai
export GTRANSCRIBER_QA_MODEL_ID=gpt-4
export GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=15

# KG Construction
export GTRANSCRIBER_KG_PROVIDER=anthropic
export GTRANSCRIBER_KG_MODEL_ID=claude-3-sonnet-20240229
export GTRANSCRIBER_KG_MERGE_GRAPHS=true

# Evaluation
export GTRANSCRIBER_EVALUATION_METRICS=qa,entity,relation
```

### API Keys

**Sensitive values should be set as environment variables only** (never commit to git):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

These can also be set in `.env` file:
```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Note**: The `.env` file should be added to `.gitignore`.

---

## Configuration Examples

### Example 1: Local Development with Ollama

**.env file**:
```bash
# Transcription (existing)
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3-turbo
GTRANSCRIBER_WORKERS=2

# QA Generation
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10

# KG Construction
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=json

# Evaluation
GTRANSCRIBER_EVALUATION_METRICS=qa,entity,relation,semantic
```

**CLI Usage**:
```bash
# QA generation (uses .env settings)
gtranscriber generate-qa results/

# Override specific settings
gtranscriber generate-qa results/ --questions 15 --provider ollama
```

### Example 2: Production with OpenAI API

**.env file**:
```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here

# QA Generation with OpenAI
GTRANSCRIBER_QA_PROVIDER=openai
GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=12
GTRANSCRIBER_QA_TEMPERATURE=0.7

# KG Construction with OpenAI
GTRANSCRIBER_KG_PROVIDER=openai
GTRANSCRIBER_KG_MODEL_ID=gpt-4o
GTRANSCRIBER_KG_TEMPERATURE=0.5
GTRANSCRIBER_KG_MERGE_GRAPHS=true

# Directories
GTRANSCRIBER_RESULTS_DIR=/data/transcriptions
GTRANSCRIBER_QA_OUTPUT_DIR=/data/qa_dataset
GTRANSCRIBER_KG_OUTPUT_DIR=/data/knowledge_graphs
```

### Example 3: Hybrid Approach (Claude + Ollama)

**.env file**:
```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-your-key-here

# QA with Claude (higher quality)
GTRANSCRIBER_QA_PROVIDER=anthropic
GTRANSCRIBER_QA_MODEL_ID=claude-3-sonnet-20240229
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=15

# KG with Ollama (cost-effective)
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:70b
GTRANSCRIBER_KG_OLLAMA_URL=http://localhost:11434

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
export GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
export GTRANSCRIBER_WORKERS=8
export GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=12

# Use $SCRATCH for I/O
export GTRANSCRIBER_RESULTS_DIR=$SCRATCH/results
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
      - GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
      - GTRANSCRIBER_QA_OLLAMA_URL=http://host.docker.internal:11434
      - GTRANSCRIBER_WORKERS=4
      - GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10
    volumes:
      - ./results:/app/results:ro
      - ./qa_dataset:/app/qa_dataset:rw

  gtranscriber-kg:
    environment:
      - GTRANSCRIBER_KG_PROVIDER=ollama
      - GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
      - GTRANSCRIBER_KG_MERGE_GRAPHS=true
      - GTRANSCRIBER_WORKERS=2
    volumes:
      - ./results:/app/results:ro
      - ./knowledge_graphs:/app/knowledge_graphs:rw
```

---

## Configuration Validation

The configuration system includes validation rules:

### Type Validation

```python
questions_per_document: int = Field(
    default=10,
    ge=1,        # Must be >= 1
    le=50,       # Must be <= 50
)
```

### Pattern Validation

```python
kg_output_format: str = Field(
    default="json",
    pattern="^(json|graphml)$"  # Must match pattern
)
```

### Custom Validation

```python
@field_validator("qa_strategies")
def validate_strategies(cls, v: list[str]) -> list[str]:
    valid = {"factual", "conceptual", "temporal", "entity"}
    for strategy in v:
        if strategy not in valid:
            raise ValueError(f"Invalid strategy: {strategy}")
    return v
```

---

## Configuration Loading Order

1. **Defaults** in `config.py`:
   ```python
   qa_provider: str = Field(default="ollama")
   ```

2. **`.env` file** (if exists):
   ```bash
   GTRANSCRIBER_QA_PROVIDER=openai
   ```

3. **Environment variables**:
   ```bash
   export GTRANSCRIBER_QA_PROVIDER=anthropic
   ```

4. **Command-line arguments** (highest priority):
   ```bash
   gtranscriber generate-qa ... --provider ollama
   ```

**Result**: Command-line arguments override everything else.

---

## Best Practices

1. **Use `.env` for local development**
   - Easy to manage and version control (with .env.example)
   - Keep `.env` in `.gitignore`

2. **Use environment variables for production**
   - Better for CI/CD and containerized environments
   - Secrets management integration

3. **Use command-line arguments for one-off overrides**
   - Quick testing without changing configuration
   - Scripting and automation

4. **Never commit API keys**
   - Always use environment variables or secrets management
   - Add `.env` to `.gitignore`

5. **Document configuration changes**
   - Update `.env.example` when adding new settings
   - Add comments explaining non-obvious settings

---

## Configuration File Template

**`.env.example`** (committed to git):
```bash
# ============================================================================
# G-Transcriber Configuration Template
# Copy this file to .env and fill in your values
# ============================================================================

# Transcription Settings
GTRANSCRIBER_MODEL_ID=openai/whisper-large-v3-turbo
GTRANSCRIBER_WORKERS=2
GTRANSCRIBER_QUANTIZE=false

# Google Drive Integration
GTRANSCRIBER_CREDENTIALS=credentials.json
GTRANSCRIBER_TOKEN=token.json

# QA Generation Settings
GTRANSCRIBER_QA_PROVIDER=ollama  # openai, anthropic, ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10

# API Keys (required for OpenAI/Anthropic)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# KG Construction Settings
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=json

# Evaluation Settings
GTRANSCRIBER_EVALUATION_METRICS=qa,entity,relation,semantic
GTRANSCRIBER_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Directories
GTRANSCRIBER_RESULTS_DIR=results
GTRANSCRIBER_QA_OUTPUT_DIR=qa_dataset
GTRANSCRIBER_KG_OUTPUT_DIR=knowledge_graphs
GTRANSCRIBER_EVALUATION_OUTPUT_DIR=evaluation
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
