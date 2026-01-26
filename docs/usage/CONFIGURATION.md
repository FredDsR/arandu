# Configuration Reference

Complete reference for all KGC pipeline configuration options.

## Configuration Methods

Settings are loaded in priority order (highest to lowest):

1. **Command-line arguments**
2. **Environment variables** (with pipeline-specific prefixes)
3. **`.env` file** in project root
4. **Default values**

## Configuration Classes

Each pipeline has a dedicated configuration class:

| Class | Env Prefix | Description |
|-------|------------|-------------|
| `TranscriberConfig` | `GTRANSCRIBER_` | Transcription pipeline |
| `QAConfig` | `GTRANSCRIBER_QA_` | QA generation pipeline |
| `KGConfig` | `GTRANSCRIBER_KG_` | KG construction pipeline |
| `EvaluationConfig` | `GTRANSCRIBER_EVAL_` | Evaluation pipeline |
| `LLMConfig` | (various) | Shared LLM settings |

## Environment Variables

### LLM Provider Settings

#### QA Generation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GTRANSCRIBER_QA_PROVIDER` | string | `ollama` | Provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_QA_MODEL_ID` | string | `llama3.1:8b` | Model identifier |
| `GTRANSCRIBER_QA_OLLAMA_URL` | string | `http://ollama:11434` | Ollama API URL |
| `GTRANSCRIBER_QA_TEMPERATURE` | float | `0.7` | LLM temperature (0.0-2.0) |
| `GTRANSCRIBER_QA_MAX_TOKENS` | int | `2048` | Max tokens for generation |
| `GTRANSCRIBER_QUESTIONS_PER_DOCUMENT` | int | `10` | QA pairs per document (1-50) |
| `GTRANSCRIBER_QA_STRATEGIES` | string | `factual,conceptual` | Comma-separated strategies |

#### KG Construction

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GTRANSCRIBER_KG_PROVIDER` | string | `ollama` | Provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_KG_MODEL_ID` | string | `llama3.1:8b` | Model identifier |
| `GTRANSCRIBER_KG_OLLAMA_URL` | string | `http://ollama:11434` | Ollama API URL |
| `GTRANSCRIBER_KG_TEMPERATURE` | float | `0.5` | LLM temperature (0.0-2.0) |
| `GTRANSCRIBER_KG_MERGE_GRAPHS` | bool | `true` | Merge into corpus graph |
| `GTRANSCRIBER_KG_OUTPUT_FORMAT` | string | `graphml` | Format: `graphml`, `json` |
| `GTRANSCRIBER_KG_SCHEMA_MODE` | string | `dynamic` | Mode: `dynamic`, `predefined` |
| `GTRANSCRIBER_KG_LANGUAGE` | string | `pt` | Language code (ISO 639-1) |
| `GTRANSCRIBER_KG_PROMPT_PATH` | string | `prompts/pt_prompts.json` | Prompt template path |

#### Shared LLM Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | - | OpenAI API key |
| `GTRANSCRIBER_LLM_BASE_URL` | string | - | Custom OpenAI-compatible URL |

### Evaluation Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GTRANSCRIBER_EVAL_METRICS` | string | `qa,entity,relation,semantic` | Metrics to compute |
| `GTRANSCRIBER_EVAL_EMBEDDING_MODEL` | string | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `GTRANSCRIBER_EVAL_OUTPUT_DIR` | path | `./evaluation` | Evaluation output |
| `GTRANSCRIBER_EVAL_QA_DIR` | path | `./qa_dataset` | QA dataset input |
| `GTRANSCRIBER_EVAL_KG_DIR` | path | `./knowledge_graphs` | KG input |
| `GTRANSCRIBER_EVAL_RESULTS_DIR` | path | `./results` | Transcription results |

### Directory Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GTRANSCRIBER_RESULTS_DIR` | path | `./results` | Transcription results |
| `GTRANSCRIBER_QA_DIR` | path | `./qa_dataset` | QA output directory |
| `GTRANSCRIBER_KG_DIR` | path | `./knowledge_graphs` | KG output directory |
| `GTRANSCRIBER_EVAL_DIR` | path | `./evaluation` | Evaluation output |
| `GTRANSCRIBER_HF_CACHE_DIR` | path | `./cache/huggingface` | HuggingFace cache |
| `OLLAMA_MODELS_DIR` | path | `./cache/ollama` | Ollama models cache |

### Processing Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GTRANSCRIBER_WORKERS` | int | `2` | Parallel workers |

### Ollama Service Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_PORT` | int | `11434` | Ollama API port |
| `OLLAMA_KEEP_ALIVE` | string | `5m` | Model keep-alive duration |

## SLURM Job Variables

Variables for SLURM job submission (short forms):

| Variable | Maps To | Default |
|----------|---------|---------|
| `QA_MODEL` | `GTRANSCRIBER_QA_MODEL_ID` | `llama3.1:8b` |
| `KG_MODEL` | `GTRANSCRIBER_KG_MODEL_ID` | `llama3.1:8b` |
| `QA_PROVIDER` | `GTRANSCRIBER_QA_PROVIDER` | `ollama` |
| `KG_PROVIDER` | `GTRANSCRIBER_KG_PROVIDER` | `ollama` |
| `WORKERS` | `GTRANSCRIBER_WORKERS` | `4` (QA), `8` (KG) |
| `QUESTIONS_PER_DOCUMENT` | `GTRANSCRIBER_QUESTIONS_PER_DOCUMENT` | `10` |
| `KG_LANGUAGE` | `GTRANSCRIBER_KG_LANGUAGE` | `pt` |
| `KG_MERGE_GRAPHS` | `GTRANSCRIBER_KG_MERGE_GRAPHS` | `true` |

## Configuration Files

### .env File

Create `.env` in project root:

```bash
# =============================================================================
# G-Transcriber Pipeline Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# QA Pipeline (GTRANSCRIBER_QA_ prefix)
# -----------------------------------------------------------------------------
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QA_OLLAMA_URL=http://ollama:11434
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=10
GTRANSCRIBER_QA_STRATEGIES=factual,conceptual
GTRANSCRIBER_QA_TEMPERATURE=0.7
GTRANSCRIBER_QA_OUTPUT_DIR=./qa_dataset

# -----------------------------------------------------------------------------
# KG Pipeline (GTRANSCRIBER_KG_ prefix)
# -----------------------------------------------------------------------------
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_OLLAMA_URL=http://ollama:11434
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=graphml
GTRANSCRIBER_KG_LANGUAGE=pt
GTRANSCRIBER_KG_TEMPERATURE=0.5
GTRANSCRIBER_KG_OUTPUT_DIR=./knowledge_graphs

# -----------------------------------------------------------------------------
# Evaluation Pipeline (GTRANSCRIBER_EVAL_ prefix)
# -----------------------------------------------------------------------------
GTRANSCRIBER_EVAL_METRICS=qa,entity,relation,semantic
GTRANSCRIBER_EVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GTRANSCRIBER_EVAL_OUTPUT_DIR=./evaluation

# -----------------------------------------------------------------------------
# Shared LLM Settings
# -----------------------------------------------------------------------------
# OPENAI_API_KEY=sk-...
# GTRANSCRIBER_LLM_BASE_URL=https://custom-api.example.com/v1

# -----------------------------------------------------------------------------
# Transcription Pipeline (GTRANSCRIBER_ prefix)
# -----------------------------------------------------------------------------
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_WORKERS=4
```

### Prompt Templates

Language-specific prompt templates in `prompts/`:

**prompts/pt_prompts.json**:
```json
{
  "pt": {
    "system": "Você é um assistente especializado em extração de conhecimento de textos em português...",
    "triple_extraction": "Extraia triplas de conhecimento (sujeito, predicado, objeto) do texto..."
  }
}
```

## Configuration Validation

The configuration system validates:

### Type Validation

```python
questions_per_document: int = Field(
    default=10,
    ge=1,       # >= 1
    le=50,      # <= 50
)
```

### Pattern Validation

```python
kg_output_format: str = Field(
    default="graphml",
    pattern="^(graphml|json)$"
)
```

### Custom Validation

```python
@field_validator("qa_strategies")
def validate_qa_strategies(cls, v):
    valid = {"factual", "conceptual", "temporal", "entity"}
    for strategy in v:
        if strategy not in valid:
            raise ValueError(f"Invalid strategy: {strategy}")
    return v
```

## Configuration Examples

### Local Development with Ollama

```bash
# .env
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.2:3b
GTRANSCRIBER_QA_WORKERS=2
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.2:3b
GTRANSCRIBER_KG_WORKERS=2
```

### Production with OpenAI

```bash
# .env
GTRANSCRIBER_QA_PROVIDER=openai
GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
GTRANSCRIBER_KG_PROVIDER=openai
GTRANSCRIBER_KG_MODEL_ID=gpt-4o
OPENAI_API_KEY=sk-...
```

### Hybrid (OpenAI for QA, Ollama for KG)

```bash
# .env
GTRANSCRIBER_QA_PROVIDER=openai
GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:70b
OPENAI_API_KEY=sk-...
```

### SLURM Cluster

```bash
# Submit with overrides (SLURM scripts use short-form variables)
QA_MODEL=llama3.1:70b \
WORKERS=16 \
QUESTIONS_PER_DOCUMENT=20 \
sbatch scripts/slurm/run_qa_generation.slurm
```

## Python Configuration Access

Each pipeline has its own configuration class with dedicated environment variable prefixes:

```python
from gtranscriber.config import (
    TranscriberConfig,  # GTRANSCRIBER_ prefix
    QAConfig,           # GTRANSCRIBER_QA_ prefix
    KGConfig,           # GTRANSCRIBER_KG_ prefix
    EvaluationConfig,   # GTRANSCRIBER_EVAL_ prefix
    LLMConfig,          # Shared LLM settings (OPENAI_API_KEY, etc.)
)

# Load pipeline-specific configurations
transcriber_config = TranscriberConfig()
qa_config = QAConfig()
kg_config = KGConfig()
eval_config = EvaluationConfig()
llm_config = LLMConfig()

# Access QA settings
print(f"QA Provider: {qa_config.provider}")
print(f"QA Model: {qa_config.model_id}")
print(f"Questions per doc: {qa_config.questions_per_document}")
print(f"QA Temperature: {qa_config.temperature}")

# Access KG settings
print(f"KG Provider: {kg_config.provider}")
print(f"KG Model: {kg_config.model_id}")
print(f"KG Language: {kg_config.language}")
print(f"Merge Graphs: {kg_config.merge_graphs}")

# Access evaluation settings
print(f"Metrics: {eval_config.metrics}")
print(f"Embedding Model: {eval_config.embedding_model}")

# Override for testing
qa_config = QAConfig(
    provider="openai",
    model_id="gpt-4o-mini",
    questions_per_document=5
)
```

### Helper Functions

```python
from gtranscriber.config import (
    get_transcriber_config,
    get_qa_config,
    get_kg_config,
    get_evaluation_config,
    get_llm_config,
)

# Get configurations using helper functions
qa_config = get_qa_config()
kg_config = get_kg_config()
```

## Docker Compose Override

Create `docker-compose.override.yml` for local overrides:

```yaml
services:
  gtranscriber-qa:
    environment:
      - GTRANSCRIBER_QA_MODEL_ID=llama3.2:3b
      - GTRANSCRIBER_WORKERS=2

  ollama:
    environment:
      - OLLAMA_KEEP_ALIVE=30m
```

---

**See also**: [QA Generation](QA_GENERATION.md) | [KG Construction](KG_CONSTRUCTION.md) | [Evaluation](EVALUATION.md)
