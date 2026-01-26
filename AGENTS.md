# AI Agent Development Guidelines

This document provides instructions for AI agents to effectively develop and maintain this codebase.

## Project Overview

**G-Transcriber** is a Python-based pipeline for processing ethnographic audio/video archives. It consists of three main pipelines:

1. **Transcription Pipeline**: Whisper-based ASR for audio/video transcription
2. **QA Pipeline**: Synthetic question-answer pair generation from transcriptions
3. **KG Pipeline**: Knowledge graph construction using LLMs

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13+ |
| Package Manager | uv (preferred), pip |
| CLI Framework | Typer with Rich for output |
| Data Validation | Pydantic v2, pydantic-settings |
| ML/AI | PyTorch, Transformers (Whisper), OpenAI SDK |
| External APIs | Google Drive API, Ollama, OpenAI |
| Graph Processing | NetworkX (GraphML format) |
| Containerization | Docker, Docker Compose |
| HPC | SLURM job scripts |
| Linting | Ruff |

## Project Structure

```
etno-kgc-preprocessing/
├── src/gtranscriber/           # Main package
│   ├── __init__.py             # Package metadata (__version__)
│   ├── main.py                 # CLI entrypoint (Typer app)
│   ├── config.py               # Pydantic Settings configurations
│   ├── schemas.py              # Pydantic models for data validation
│   ├── core/                   # Core functionality
│   │   ├── batch.py            # Batch processing with checkpointing
│   │   ├── checkpoint.py       # Progress checkpoint management
│   │   ├── drive.py            # Google Drive API integration
│   │   ├── engine.py           # Whisper transcription engine
│   │   ├── hardware.py         # Hardware detection (CPU/CUDA/MPS)
│   │   ├── io.py               # File I/O operations
│   │   └── llm_client.py       # Unified LLM client (OpenAI/Ollama)
│   └── utils/                  # Utilities
│       ├── console.py          # Rich console setup
│       ├── logger.py           # Rich logging integration
│       └── ui.py               # Progress bars and UI components
├── scripts/slurm/              # SLURM job scripts for HPC clusters
├── docs/                       # Documentation
├── docker-compose.yml          # Service definitions
├── Dockerfile                  # Multi-stage Docker build
└── pyproject.toml              # Project configuration and dependencies
```

## Code Patterns and Conventions

### Configuration Pattern

Use `pydantic-settings` for configuration with environment variable support:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MyConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    my_setting: str = Field(
        default="default_value",
        description="Clear description of the setting",
    )
```

### Schema Pattern

Use Pydantic models for all data structures with comprehensive validation:

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class MySchema(BaseModel):
    required_field: str = Field(..., description="Description")
    optional_field: int | None = Field(None, description="Description")

    @field_validator("required_field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        # Validation logic
        return v

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MySchema":
        """Load from JSON file."""
        return cls.model_validate_json(Path(path).read_text())
```

### CLI Pattern

Use Typer with Rich integration and environment variable defaults:

```python
from typing import Annotated
import typer
from gtranscriber.config import get_my_config

_config = get_my_config()

@app.command()
def my_command(
    required_arg: Annotated[
        str,
        typer.Argument(help="Description of argument."),
    ],
    optional_flag: Annotated[
        bool,
        typer.Option(
            "--flag",
            "-f",
            help="Description. Can be set via GTRANSCRIBER_FLAG env var.",
        ),
    ] = _config.flag,
) -> None:
    """Command docstring shown in help."""
    # Implementation
```

### LLM Client Pattern

Use the unified `LLMClient` that supports multiple providers:

```python
from gtranscriber.core.llm_client import create_llm_client, LLMProvider

# Ollama (local)
client = create_llm_client("ollama", "llama3.1:8b")

# OpenAI
client = create_llm_client(LLMProvider.OPENAI, "gpt-4", api_key="sk-...")

# Custom endpoint
client = create_llm_client("custom", "model", base_url="http://localhost:8000/v1")

response = client.generate(
    prompt="Question here",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
)
```

### Error Handling Pattern

Use Rich console utilities for user-facing messages:

```python
from gtranscriber.utils.logger import print_error, print_success, print_info, print_warning

try:
    # Operation
    print_success("Operation completed successfully")
except SomeError as e:
    print_error(f"Operation failed: {e}")
    raise typer.Exit(code=1) from e
```

### Retry Pattern

Use tenacity for operations that may fail transiently:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def unreliable_operation():
    # Implementation
```

## Type Annotations

### Required Annotations

This project uses strict type annotations enforced by Ruff:

- All function arguments must have type annotations (`ANN001`)
- All public functions must have return type annotations (`ANN201`)
- All private functions must have return type annotations (`ANN202`)

**Exception**: CLI command functions in `main.py` are exempt from these rules.

### Type Annotation Patterns

```python
from __future__ import annotations  # Always use at top of file

from typing import TYPE_CHECKING, Literal
from pathlib import Path

if TYPE_CHECKING:
    from typing import Self  # For return type in methods

# Use | for unions (Python 3.10+ syntax)
def process(value: str | None) -> list[str]:
    ...

# Use Literal for constrained string values
status: Literal["pending", "completed", "failed"]
```

## Imports Organization

Ruff enforces import ordering via isort rules:

```python
# 1. Standard library
from __future__ import annotations
import logging
from pathlib import Path

# 2. Third-party packages
import torch
from pydantic import BaseModel
from rich.console import Console

# 3. First-party (gtranscriber)
from gtranscriber.config import TranscriberConfig
from gtranscriber.schemas import EnrichedRecord
```

## Ruff Configuration

The project uses Ruff for linting with these rules enabled:

- `E`, `W`: pycodestyle errors and warnings
- `F`: pyflakes
- `I`: isort (import sorting)
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `UP`: pyupgrade
- `ANN001`, `ANN201`, `ANN202`: type annotations
- `SIM`: flake8-simplify
- `TCH`: flake8-type-checking
- `RUF`: Ruff-specific rules

**Line length**: 100 characters

Run linting with:
```bash
ruff check src/
ruff format src/
```

## Environment Variables

All configuration uses the `GTRANSCRIBER_` prefix. Key variables:

### Transcription
- `GTRANSCRIBER_MODEL_ID`: Whisper model (default: `openai/whisper-large-v3-turbo`)
- `GTRANSCRIBER_FORCE_CPU`: Force CPU execution
- `GTRANSCRIBER_QUANTIZE`: Enable 8-bit quantization
- `GTRANSCRIBER_WORKERS`: Parallel workers count

### QA Pipeline
- `GTRANSCRIBER_QA_PROVIDER`: `ollama`, `openai`, or `custom`
- `GTRANSCRIBER_QA_MODEL_ID`: LLM model for QA generation
- `GTRANSCRIBER_QA_OLLAMA_URL`: Ollama API URL

### KG Pipeline
- `GTRANSCRIBER_KG_PROVIDER`: `ollama`, `openai`, or `custom`
- `GTRANSCRIBER_KG_MODEL_ID`: LLM model for extraction
- `GTRANSCRIBER_KG_LANGUAGE`: Language code (`pt`, `en`, `es`)

### Shared
- `OPENAI_API_KEY`: OpenAI API key (when using OpenAI provider)

## Docker Compose Profiles

Run specific pipelines using profiles:

```bash
# Transcription (default, with GPU)
docker compose up

# CPU-only transcription
docker compose --profile cpu up

# QA generation (requires Ollama)
docker compose --profile qa up

# Knowledge graph construction
docker compose --profile kg up

# Evaluation
docker compose --profile evaluate up
```

## Testing Guidelines

When writing tests:

1. Place tests in `tests/` directory mirroring `src/` structure
2. Use pytest as the test framework
3. Mock external services (Google Drive API, LLM providers)
4. Test Pydantic models with edge cases

## Adding New Features

### New CLI Command

1. Add command function in `src/gtranscriber/main.py`
2. Use `Annotated` for all parameters with `typer.Argument` or `typer.Option`
3. Load defaults from config classes
4. Document with docstring (shown in `--help`)

### New Configuration

1. Add settings class in `config.py` extending `BaseSettings`
2. Use appropriate `env_prefix` (e.g., `GTRANSCRIBER_NEW_`)
3. Add factory function `get_new_config()`
4. Document in README.md configuration tables

### New Schema

1. Add Pydantic model in `schemas.py`
2. Include `save()` and `load()` class methods if persistent
3. Use `Field(...)` for required fields, `Field(default=...)` for optional
4. Add validators with `@field_validator` or `@model_validator`

### New Core Module

1. Create file in `src/gtranscriber/core/`
2. Use dataclasses for result objects
3. Use classes for stateful operations (engines, clients)
4. Provide convenience functions for simple use cases

## Common Pitfalls to Avoid

1. **Don't hardcode paths**: Use configuration or environment variables
2. **Don't skip type annotations**: Required by Ruff rules
3. **Don't use `print()`**: Use Rich console utilities instead
4. **Don't ignore error handling**: Use proper try/except with user-friendly messages
5. **Don't create unnecessary files**: Prefer editing existing code
6. **Don't skip validation**: Use Pydantic for all data structures
7. **Don't duplicate configuration**: Reuse existing config classes

## Helpful Commands

```bash
# Install dependencies (development)
uv pip install -e .

# Run CLI
gtranscriber --help
gtranscriber info
gtranscriber transcribe audio.mp3

# Lint code
ruff check src/
ruff format src/

# Build Docker image
docker compose build

# Run with Docker
docker compose up gtranscriber
```

## References

- [Typer Documentation](https://typer.tiangolo.com/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
