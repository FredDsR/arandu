# GitHub Copilot Instructions

**Read [AGENT.md](../AGENT.md) for comprehensive development guidelines.**

## Project: G-Transcriber

Python 3.13+ pipeline for ethnographic audio/video processing with Whisper ASR, LLM-based QA generation, and knowledge graph construction.

## Quick Essentials

### Mandatory Standards
- **Type annotations**: All function args and returns (`from __future__ import annotations`)
- **Docstrings**: Google style for all public functions/classes
- **Git**: Conventional commits and branches
- **Linting**: Ruff with 100 char line limit
- **No `print()`**: Use Rich utilities from `gtranscriber.utils.logger`

### Technology Stack
- **CLI**: Typer + Rich
- **Validation**: Pydantic v2
- **ML**: PyTorch, Transformers (Whisper)
- **LLM**: Unified `LLMClient` (never use provider SDKs directly)
- **Package Manager**: uv

### Import Order (isort)
1. Standard library
2. Third-party packages
3. First-party (gtranscriber)

### Pre-Commit
```bash
uv run ruff check --fix src/ && uv run ruff format src/ && uv run pytest
```

## Key Patterns

**Configuration**: Pydantic Settings with `GTRANSCRIBER_` env prefix
**Schemas**: Pydantic models with `@field_validator`
**CLI**: Typer with `Annotated` types, load defaults from config
**Error Handling**: Rich console utilities, `typer.Exit(code=1)` for CLI errors
