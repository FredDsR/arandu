# Dependencies Documentation

Complete documentation of all project dependencies for the Knowledge Graph Construction Pipeline.

## Table of Contents

1. [Existing Dependencies](#existing-dependencies)
2. [New Dependencies](#new-dependencies)
3. [Dependency Groups](#dependency-groups)
4. [Installation Instructions](#installation-instructions)
5. [Version Compatibility](#version-compatibility)
6. [License Information](#license-information)

---

## Existing Dependencies

Dependencies from the original transcription pipeline:

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `accelerate` | >=1.12.0 | Model acceleration and device management |
| `google-api-python-client` | latest | Google Drive API integration |
| `google-auth-httplib2` | latest | Google authentication |
| `google-auth-oauthlib` | latest | OAuth2 flow |
| `pydantic` | >=2.0.0 | Data validation, schemas, and JSON serialization |
| `pydantic-settings` | >=2.0.0 | Configuration management with env var support |
| `rich` | >=13.0.0 | Terminal UI and formatting |
| `sentencepiece` | >=0.2.1 | Tokenization for Whisper |
| `tenacity` | >=8.0.0 | Retry logic with exponential backoff |
| `transformers` | >=4.57.3 | Hugging Face transformers |
| `typer[all]` | >=0.9.0 | CLI framework |

### ML/AI Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | PyTorch deep learning framework |
| `torchaudio` | >=2.0.0 | Audio processing |

---

## New Dependencies

Dependencies added for P2 functionality:

### LLM Integration

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `openai` | >=1.0.0 | OpenAI API client | `llm_client.py` |
| `anthropic` | >=0.18.0 | Anthropic Claude API client | `llm_client.py` |
| `httpx` | >=0.27.0 | HTTP client for Ollama | `llm_client.py` |

### Knowledge Graph Construction

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `atlas-rag` | >=0.0.5 | AutoSchemaKG framework | `kg_builder.py` |
| `networkx` | >=3.1 | Graph data structures and algorithms | `kg_builder.py`, `metrics.py` |

### Evaluation and Metrics

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `scikit-learn` | >=1.3.0 | Machine learning metrics (F1, etc.) | `metrics.py` |
| `sentence-transformers` | >=2.2.0 | Semantic embeddings | `metrics.py`, `evaluator.py` |
| `nltk` | >=3.8.0 | NLP utilities (tokenization) | `metrics.py` |
| `sacrebleu` | >=2.3.0 | BLEU score calculation | `metrics.py` |

---

## Dependency Groups

### Production Dependencies

Required for running the application:

```toml
[project]
dependencies = [
    # Existing
    "accelerate>=1.12.0",
    "google-api-python-client",
    "google-auth-httplib2",
    "google-auth-oauthlib",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "rich>=13.0.0",
    "sentencepiece>=0.2.1",
    "tenacity>=8.0.0",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "transformers>=4.57.3",
    "typer[all]>=0.9.0",

    # New - LLM Integration
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "httpx>=0.27.0",

    # New - KG Construction
    "atlas-rag>=0.0.5",
    "networkx>=3.1",

    # New - Evaluation
    "scikit-learn>=1.3.0",
    "sentence-transformers>=2.2.0",
    "nltk>=3.8.0",
    "sacrebleu>=2.3.0",
]
```

### Development Dependencies

Optional dependencies for development:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]
```

### Documentation Dependencies

For building documentation:

```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.24.0",
]
```

### Testing Dependencies

For running tests:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "responses>=0.23.0",
    "faker>=20.0.0",
]
```

---

## Installation Instructions

### Basic Installation

Install production dependencies only:

```bash
pip install -e .
```

### Development Installation

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Complete Installation

Install all optional dependencies:

```bash
pip install -e ".[dev,docs,test]"
```

### Using uv (Recommended)

For faster installation:

```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

### Using Poetry (Alternative)

If using Poetry for dependency management:

```bash
poetry install
poetry install --with dev,docs,test
```

### Docker Installation

Dependencies are automatically installed in Docker:

```bash
docker compose build
```

---

## Version Compatibility

### Python Version

**Required**: Python >= 3.13

**Tested on**:
- Python 3.13.0
- Python 3.13.1

### PyTorch Version

**CUDA Support**:
- `torch>=2.0.0+cu124` (CUDA 12.4)
- `torchaudio>=2.0.0+cu124`

**ROCm Support** (AMD GPUs):
- `torch>=2.0.0+rocm5.7`

**CPU Only**:
- `torch>=2.0.0+cpu`

**Installation**:
```bash
# CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# ROCm
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# CPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Transformers Version Constraints

For **NV-embed-v2** support (optional):
- `transformers>=4.42.4,<=4.47.1`

For **general use**:
- `transformers>=4.57.3`

### NetworkX Version

Minimum version 3.1 required for:
- Modern graph algorithms
- Better performance
- JSON serialization improvements

### Sentence Transformers Compatibility

**Models Tested**:
- `sentence-transformers/all-MiniLM-L6-v2` (default)
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

---

## Detailed Dependency Information

### pydantic (>=2.0.0)

**Purpose**: Data validation, schema definitions, and JSON serialization

**Features Used**:
- `BaseModel` - All data schemas (QAPair, QARecord, EvaluationReport, etc.)
- `Field()` - Validation constraints (ge, le, pattern, default_factory)
- `@field_validator` - Custom field validation
- `@model_validator` - Cross-field validation
- `@computed_field` - Derived/calculated properties
- `model_dump_json()` - JSON serialization
- `model_validate_json()` - JSON deserialization
- `model_json_schema()` - JSON Schema export

**Why Pydantic over dataclasses**:
- Built-in validation with declarative constraints
- Automatic JSON serialization/deserialization with datetime support
- Better error messages with field paths
- Ecosystem alignment (OpenAI, Anthropic SDKs use Pydantic)
- Computed fields for derived values
- Rust-based validation core for performance (Pydantic v2)

**Documentation**: https://docs.pydantic.dev/latest/

### pydantic-settings (>=2.0.0)

**Purpose**: Configuration management with environment variable support

**Features Used**:
- `BaseSettings` - Configuration class with env var loading
- `SettingsConfigDict` - Configuration for env prefix, .env file support
- Automatic type coercion from string env vars

**Documentation**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### openai (>=1.0.0)

**Purpose**: OpenAI API client for GPT models

**Features Used**:
- Chat completions API
- Streaming responses
- Token usage tracking

**API Models**:
- `gpt-4o` (recommended)
- `gpt-4o-mini` (cost-effective)
- `gpt-4`
- `gpt-3.5-turbo`

**Authentication**: Requires `OPENAI_API_KEY` environment variable

**Documentation**: https://platform.openai.com/docs/api-reference

### anthropic (>=0.18.0)

**Purpose**: Anthropic Claude API client

**Features Used**:
- Messages API
- Streaming responses
- Token usage tracking

**API Models**:
- `claude-3-sonnet-20240229` (recommended)
- `claude-3-opus-20240229` (highest quality)
- `claude-3-haiku-20240307` (fastest)

**Authentication**: Requires `ANTHROPIC_API_KEY` environment variable

**Documentation**: https://docs.anthropic.com/claude/reference/

### httpx (>=0.27.0)

**Purpose**: HTTP client for Ollama API

**Features Used**:
- Async requests
- Timeout handling
- Connection pooling

**Default URL**: `http://localhost:11434`

**Documentation**: https://www.python-httpx.org/

### atlas-rag (>=0.0.5)

**Purpose**: AutoSchemaKG framework for knowledge graph construction

**Features Used**:
- Triple extraction
- Dynamic schema induction
- Graph construction
- NetworkX integration

**Key Modules**:
- `atlas_rag.kg_construction`
- `atlas_rag.llm_generator`
- `atlas_rag.utils`

**Documentation**: https://hkust-knowcomp.github.io/AutoSchemaKG/

**Paper**: https://arxiv.org/abs/2505.23628

### networkx (>=3.1)

**Purpose**: Graph data structures and algorithms

**Features Used**:
- Graph creation and manipulation
- Node and edge attributes
- Graph algorithms (connectivity, density)
- JSON serialization
- GraphML export

**Documentation**: https://networkx.org/documentation/stable/

### scikit-learn (>=1.3.0)

**Purpose**: Machine learning metrics

**Features Used**:
- F1 score calculation
- Precision and recall
- Token-level comparison

**Documentation**: https://scikit-learn.org/stable/

### sentence-transformers (>=2.2.0)

**Purpose**: Semantic embeddings for text

**Features Used**:
- Sentence embeddings
- Semantic similarity
- Coherence scoring

**Models Used**:
- `all-MiniLM-L6-v2` (384 dims, fast)
- `all-mpnet-base-v2` (768 dims, high quality)

**Documentation**: https://www.sbert.net/

### nltk (>=3.8.0)

**Purpose**: Natural language processing utilities

**Features Used**:
- Tokenization
- Word counting
- Text preprocessing

**Data Required**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Documentation**: https://www.nltk.org/

### sacrebleu (>=2.3.0)

**Purpose**: BLEU score calculation

**Features Used**:
- Sentence-level BLEU
- Corpus-level BLEU
- Multiple references

**Documentation**: https://github.com/mjpost/sacrebleu

---

## License Information

### MIT Licensed

- `openai`
- `httpx`
- `networkx`
- `nltk`
- `sacrebleu`

### Apache 2.0 Licensed

- `anthropic`
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `torch`

### BSD Licensed

- `pydantic`
- `typer`
- `rich`

### AutoSchemaKG License

- **License**: MIT
- **Citation Required**: Yes (if publishing research)
- **Commercial Use**: Allowed

**Citation**:
```bibtex
@article{huang2025autoschemakg,
  title={AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora},
  author={Huang, Haoyu and others},
  journal={arXiv preprint arXiv:2505.23628},
  year={2025}
}
```

---

## Dependency Size Information

### Installation Sizes

| Package | Disk Space |
|---------|------------|
| `torch` | ~2.5 GB |
| `transformers` | ~500 MB |
| `sentence-transformers` | ~400 MB |
| `atlas-rag` | ~50 MB |
| `openai` | ~5 MB |
| `anthropic` | ~3 MB |
| `networkx` | ~10 MB |
| `scikit-learn` | ~40 MB |
| `nltk` | ~20 MB + data |
| **Total** | **~3.5 GB** |

### Model Sizes (Downloaded at Runtime)

| Model | Size | Used By |
|-------|------|---------|
| Whisper Large V3 | ~3 GB | Transcription |
| all-MiniLM-L6-v2 | ~80 MB | Evaluation |
| Llama 3.1 8B (Ollama) | ~4.7 GB | QA/KG (if using Ollama) |

---

## Troubleshooting Dependencies

### Common Issues

**Issue**: `torch` installation fails with CUDA

**Solution**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Issue**: `atlas-rag` not found

**Solution**:
```bash
pip install atlas-rag --upgrade
```

**Issue**: `sentence-transformers` model download fails

**Solution**:
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Issue**: NLTK data not found

**Solution**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dependency Conflicts

**Conflict**: `transformers` version mismatch

**Resolution**:
```bash
pip install "transformers>=4.57.3" --upgrade
```

**Conflict**: `pydantic` v1 vs v2

**Resolution**: Project requires Pydantic v2
```bash
pip install "pydantic>=2.0.0" --upgrade
```

---

## Updating Dependencies

### Check for Updates

```bash
pip list --outdated
```

### Update All

```bash
pip install --upgrade pip
pip install -e . --upgrade
```

### Update Specific Package

```bash
pip install --upgrade openai
pip install --upgrade anthropic
```

### Lock Dependencies

For reproducible builds:

```bash
pip freeze > requirements.txt
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
