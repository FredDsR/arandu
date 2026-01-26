# GitHub Copilot Instructions

This is **G-Transcriber**, a Python 3.13+ project for processing ethnographic audio/video archives.

## Technology Stack

- Python 3.13+, uv package manager
- Typer + Rich for CLI
- Pydantic v2 for data validation
- PyTorch + Transformers for ML
- OpenAI SDK for LLM integration
- Ruff for linting (line length: 100)

## Code Style Requirements

### Type Annotations (Required)
```python
from __future__ import annotations

def process_file(path: str | Path, options: dict[str, Any] | None = None) -> list[str]:
    ...
```

### Configuration Pattern
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class MyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GTRANSCRIBER_", env_file=".env")
    setting: str = Field(default="value", description="Description")
```

### CLI Commands
```python
from typing import Annotated
import typer

@app.command()
def command(
    arg: Annotated[str, typer.Argument(help="Description")],
    flag: Annotated[bool, typer.Option("--flag", "-f")] = False,
) -> None:
    """Command help text."""
```

### User Output
```python
from gtranscriber.utils.logger import print_error, print_success, print_info
# Never use print() directly
```

## Import Order (isort)
1. Standard library
2. Third-party packages
3. First-party (gtranscriber)

## Key Directories

- `src/gtranscriber/` - Main package
- `src/gtranscriber/core/` - Core functionality (engine, drive, llm_client)
- `src/gtranscriber/utils/` - Utilities (logger, console, ui)
- `scripts/slurm/` - HPC job scripts

## Full Documentation

See [AGENTS.md](../AGENTS.md) for comprehensive guidelines.
