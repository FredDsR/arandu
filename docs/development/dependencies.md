# Dependencies Documentation

Complete documentation of all project dependencies for the Knowledge Graph Construction Pipeline.

## Table of Contents

1. [Existing Dependencies](#existing-dependencies)
2. [Optional Dependencies (Extras)](#optional-dependencies-extras)
3. [New Dependencies](#new-dependencies)
4. [Dependency Groups](#dependency-groups)
5. [Installation Instructions](#installation-instructions)
6. [Version Compatibility](#version-compatibility)
7. [License Information](#license-information)

---

## Existing Dependencies

Dependencies from the original transcription pipeline:

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `accelerate` | >=1.12.0 | Model acceleration and device management |
| `bitsandbytes` | >=0.49.1 | Quantization and memory optimization |
| `fastapi` | >=0.115.0 | Web framework backing the report dashboard server (`serve-report`) |
| `uvicorn[standard]` | >=0.34.0 | ASGI server for the dashboard |
| `google-api-python-client` | >=2.100.0 | Google Drive API integration |
| `google-auth-httplib2` | >=0.1.0 | Google authentication |
| `google-auth-oauthlib` | >=1.0.0 | OAuth2 flow |
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
| `torch` | (via uv sources) | PyTorch deep learning framework (CUDA 12.4) |
| `torchvision` | (via uv sources) | Vision processing utilities |
| `torchaudio` | (via uv sources) | Audio processing |

### RAG & NLP Dependencies

Core dependencies powering Phase C retrieval and chunking:

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | >=3.1 | Graph data structures and algorithms (KG loading, metrics, k-hop traversal) |
| `chonkie` | >=1.6.6 | Text chunking for the chunker views |
| `rank-bm25` | >=0.2.2 | BM25 lexical retriever arm |
| `spacy` | >=3.8.14 | Lemmatization for k-hop seed weighting (`shared/rag/retrievers/_khop_common.py`) |
| `pt-core-news-sm` | (pinned wheel) | spaCy Portuguese model used for k-hop seed lemmatization |
| `en-core-web-sm` | (pinned wheel) | spaCy English model for k-hop seed lemmatization |

---

## Optional Dependencies (Extras)

Install with `uv sync --extra <name>` (or `pip install -e ".[<name>]"`).

### `report`

| Package | Version | Purpose |
|---------|---------|---------|
| `plotly` | >=6.0.0 | Interactive charts for the HTML report |
| `jinja2` | >=3.1.0 | HTML templating |
| `kaleido` | >=0.2.1 | Static PNG export of Plotly charts |

### `kg`

| Package | Version | Purpose |
|---------|---------|---------|
| `atlas-rag` | >=0.0.5 | AutoSchemaKG knowledge graph construction backend |
| `pyarrow` | >=15,<21 | Columnar storage used by atlas-rag / datasets |
| `datasets` | >=2.18 | Dataset loading for atlas-rag |
| `faiss-cpu` | >=1.7 | Vector index for the atlas-rag retriever |

### `dashboard`

| Package | Version | Purpose |
|---------|---------|---------|
| `textual` | >=3.0.0 | Terminal UI dashboard |

---

## New Dependencies

Dependencies added for P2 functionality:

### LLM Integration

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `openai` | >=1.0.0 | OpenAI API client (also supports Ollama and other OpenAI-compatible endpoints) | `llm_client.py` |
| `httpx` | >=0.27.0 | HTTP client (used by openai SDK) | `llm_client.py` |

---

## Planned Dependencies (Not Yet Added)

The following dependencies are planned for future phases but are **not currently in `pyproject.toml`**:

> `atlas-rag` and `networkx` are no longer planned — both are installed (`networkx` is core; `atlas-rag` is in the `kg` extra). See the sections above.

### Evaluation and Metrics (Planned for Evaluation Phase)

| Package | Version | Purpose | Planned Module |
|---------|---------|---------|---------|
| `scikit-learn` | >=1.3.0 | Machine learning metrics (F1, etc.) | `metrics.py` |
| `sentence-transformers` | >=2.2.0 | Semantic embeddings | `metrics.py`, `evaluator.py` |
| `nltk` | >=3.8.0 | NLP utilities (tokenization) | `metrics.py` |
| `sacrebleu` | >=2.3.0 | BLEU score calculation | `metrics.py` |

> **Note**: These dependencies will be added to `pyproject.toml` when their respective implementation phases begin.

---

## Dependency Groups

### Production Dependencies

Required for running the application:

```toml
[project]
dependencies = [
    "accelerate>=1.12.0",
    "bitsandbytes>=0.49.1",
    "fastapi>=0.115.0",
    "google-api-python-client>=2.100.0",
    "google-auth-httplib2>=0.1.0",
    "google-auth-oauthlib>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "rich>=13.0.0",
    "sentencepiece>=0.2.1",
    "tenacity>=8.0.0",
    "transformers>=4.57.3",
    "typer[all]>=0.9.0",
    "uvicorn[standard]>=0.34.0",
    # LLM Integration (OpenAI SDK supports Ollama and other compatible endpoints)
    "openai>=1.0.0",
    "httpx>=0.27.0",
    # Graph loading, chunking, retrieval
    "networkx>=3.1",
    "chonkie>=1.6.6",
    "rank-bm25>=0.2.2",
    "spacy>=3.8.14",
    "pt-core-news-sm",
    "en-core-web-sm",
]

[project.optional-dependencies]
report = ["plotly>=6.0.0", "jinja2>=3.1.0", "kaleido>=0.2.1"]
kg = ["atlas-rag>=0.0.5", "pyarrow>=15,<21", "datasets>=2.18", "faiss-cpu>=1.7"]
dashboard = ["textual>=3.0.0"]
```

### Development Dependencies

Dependencies for code quality and linting:

```toml
[dependency-groups]
dev = [
    "ruff>=0.8.0",
]
```

### Testing Dependencies

Dependencies for running tests:

```toml
[dependency-groups]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
]
```

---

## Installation Instructions

### Build System

This project uses **uv** as the build backend and package manager:

```toml
[build-system]
requires = ["uv_build>=0.9.26,<0.10.0"]
build-backend = "uv_build"
```

### Basic Installation

Install production dependencies:

```bash
uv sync
```

### Development Installation

Install with development dependencies:

```bash
uv sync --group dev
```

### Testing Installation

Install with test dependencies:

```bash
uv sync --group test
```

### Complete Installation

Install all dependency groups:

```bash
uv sync --all-groups
```

### Alternative: pip Installation

If not using uv:

```bash
pip install -e .
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

**CUDA Support** (Configured via `[tool.uv.sources]`):

The project is configured to use PyTorch with CUDA 12.4 support. This is managed through uv's custom index configuration:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
priority = "supplemental"

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
```

When running `uv sync`, PyTorch packages are automatically installed from the CUDA 12.4 index.

**Manual Installation** (if not using uv):

```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ROCm (AMD GPUs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# CPU Only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Transformers Version Constraints

For **general use**:
- `transformers>=4.57.3`

### bitsandbytes Version

**Purpose**: Quantization and memory optimization for LLMs

**Minimum version**: >=0.49.1

**Features Used**:
- 8-bit and 4-bit quantization
- Memory-efficient model loading
- CUDA optimization

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
- Ecosystem alignment (OpenAI SDK uses Pydantic)
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

### httpx (>=0.27.0)

**Purpose**: HTTP client for Ollama API

**Features Used**:
- Async requests
- Timeout handling
- Connection pooling

**Default URL**: `http://localhost:11434`

**Documentation**: https://www.python-httpx.org/

---

## Dependency Information

> **Note**: `atlas-rag` (kg extra) and `networkx` (core) are installed and in use. `scikit-learn`, `sentence-transformers`, `nltk`, and `sacrebleu` are still planned for the evaluation phase.

### atlas-rag (>=0.0.5) — installed (`kg` extra)

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

### networkx (>=3.1) — installed (core)

**Purpose**: Graph data structures and algorithms

**Features Used**:
- Graph creation and manipulation
- Node and edge attributes
- Graph algorithms (connectivity, density)
- k-hop traversal for the `khop_passage` / `khop_triple` retriever arms
- GraphML export

**Documentation**: https://networkx.org/documentation/stable/

### scikit-learn (>=1.3.0)

**Purpose**: Machine learning metrics

**Features Planned**:
- F1 score calculation
- Precision and recall
- Token-level comparison

**Documentation**: https://scikit-learn.org/stable/

### sentence-transformers (>=2.2.0)

**Purpose**: Semantic embeddings for text

**Features Planned**:
- Sentence embeddings
- Semantic similarity
- Coherence scoring

**Models Planned**:
- `all-MiniLM-L6-v2` (384 dims, fast)
- `all-mpnet-base-v2` (768 dims, high quality)

**Documentation**: https://www.sbert.net/

### nltk (>=3.8.0)

**Purpose**: Natural language processing utilities

**Features Planned**:
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

**Features Planned**:
- Sentence-level BLEU
- Corpus-level BLEU
- Multiple references

**Documentation**: https://github.com/mjpost/sacrebleu

---

## License Information

### MIT Licensed

- `openai`
- `httpx`
- `pydantic`
- `typer`
- `rich`
- `networkx`
- `atlas-rag`
- `rank-bm25`
- `fastapi`

### Apache 2.0 Licensed

- `transformers`
- `torch`

### BSD Licensed

- `accelerate`
- `sentencepiece`

---

## Planned Dependency Licenses

> **Note**: For dependencies not yet added to the project.

### MIT Licensed (Planned)

- `nltk`
- `sacrebleu`

### Apache 2.0 Licensed (Planned)

- `sentence-transformers`
- `scikit-learn`

### AutoSchemaKG Citation

If using `atlas-rag` for published research, citation is required:

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

### Installation Sizes (Current Dependencies)

| Package | Disk Space |
|---------|------------|
| `torch` | ~2.5 GB |
| `transformers` | ~500 MB |
| `bitsandbytes` | ~50 MB |
| `openai` | ~5 MB |
| `accelerate` | ~30 MB |
| Other packages | ~100 MB |
| **Total** | **~3.2 GB** |

### Additional Sizes (Planned Dependencies)

| Package | Disk Space |
|---------|------------|
| `sentence-transformers` | ~400 MB |
| `atlas-rag` | ~50 MB |
| `networkx` | ~10 MB |
| `scikit-learn` | ~40 MB |
| `nltk` | ~20 MB + data |
| **Total (with planned)** | **~3.7 GB** |

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
# With uv (automatic via uv sources)
uv sync

# Manual installation
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Issue**: `bitsandbytes` installation fails

**Solution**:
```bash
# Ensure CUDA is available
nvidia-smi

# Reinstall
pip install bitsandbytes --upgrade
```

### Dependency Conflicts

**Conflict**: `pydantic` v1 vs v2

**Resolution**: Project requires Pydantic v2
```bash
pip install "pydantic>=2.0.0" --upgrade
```

**Conflict**: `transformers` version mismatch

**Resolution**:
```bash
pip install "transformers>=4.57.3" --upgrade
```

---

## Updating Dependencies

### Check for Updates

```bash
uv lock --upgrade
```

### Update All Dependencies

```bash
uv sync --upgrade
```

### Update Specific Package

```bash
uv add openai --upgrade
```

### Lock Dependencies

For reproducible builds, `uv.lock` is automatically maintained by uv.

---

**Document Version**: 1.1
**Last Updated**: 2026-06-17
