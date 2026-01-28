# Claude Code Instructions

This project is **G-Transcriber**, a Python-based pipeline for processing ethnographic audio/video archives with transcription, QA generation, and knowledge graph construction capabilities.

## Quick Reference

- **Language**: Python 3.13+
- **Package Manager**: uv (preferred)
- **CLI Framework**: Typer + Rich
- **Data Validation**: Pydantic v2
- **Linting**: Ruff (line length: 100)

## Key Guidelines

1. **Type annotations are required** - All functions need type hints (enforced by Ruff)
2. **Use pydantic-settings** for configuration with `GTRANSCRIBER_` env prefix
3. **Use Rich console** for user output, not `print()`
4. **Prefer editing existing files** over creating new ones
5. **Follow the patterns** in `src/gtranscriber/config.py` and `schemas.py`

## Essential Commands

```bash
# Lint and format
ruff check src/ && ruff format src/

# Run CLI
gtranscriber --help

# Docker - GPU transcription
docker compose --profile gpu build
docker compose --profile gpu up

# Docker - QA generation
docker compose --profile qa up
```

## Full Documentation

See [AGENTS.md](AGENTS.md) for comprehensive development guidelines including:
- Project structure and architecture
- Code patterns with examples
- Configuration and environment variables
- Docker Compose profiles
- Testing guidelines
