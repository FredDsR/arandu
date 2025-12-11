# GitHub Copilot Instructions for G-Transcriber

This document provides guidelines for GitHub Copilot when working on this repository.

## Project Overview

G-Transcriber is an automated transcription system for media files stored in Google Drive using Whisper ASR. The project integrates with Google Drive for seamless file management and uses state-of-the-art speech recognition models from Hugging Face.

## Technology Stack

- **Python Version**: 3.13+ (required)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended)
- **Linter/Formatter**: [ruff](https://github.com/astral-sh/ruff)
- **Key Dependencies**:
  - PyTorch for ML operations
  - Transformers for Whisper models
  - Google API Python Client for Drive integration
  - Rich for CLI UI
  - Typer for CLI framework
  - Pydantic for data validation

## Development Setup

### Package Management with uv

This project uses `uv` as the package manager. Key commands:

```bash
# Install dependencies in development mode
uv sync --dev

# Add a new dependency
uv add <package-name>

# Sync dependencies
uv sync
```

### Code Quality with ruff

Ruff is configured in `pyproject.toml` with specific linting rules. Key commands:

```bash
# Check code
ruff check .

# Format code
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

### Configuration

Ruff configuration in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.13
- Enabled rules: pycodestyle, pyflakes, isort, flake8-bugbear, flake8-comprehensions, pyupgrade, type annotations, flake8-simplify, flake8-type-checking, and Ruff-specific rules

## Code Style Guidelines

### Type Annotations

- **Required**: All function arguments should have type annotations (ANN001)
- **Required**: Public functions must have return type annotations (ANN201)
- **Required**: Private functions must have return type annotations (ANN202)
- **Exception**: CLI command functions in `src/gtranscriber/main.py` are exempt from exhaustive annotations

### Import Organization

- Use isort-compatible import ordering
- First-party imports: `gtranscriber` package

### Code Quality

- Follow pycodestyle (E/W rules)
- Use flake8-bugbear best practices
- Apply flake8-simplify suggestions
- Use comprehensions where appropriate (C4)
- Apply pyupgrade modern Python syntax (UP)
- Follow flake8-type-checking guidelines (TCH)

## Project Structure

```
gtranscriber/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project documentation
├── main.py                 # Legacy entrypoint (if exists)
└── src/
    └── gtranscriber/
        ├── __init__.py
        ├── main.py         # CLI entrypoint (Typer app)
        ├── config.py       # Configuration management
        ├── schemas.py      # Pydantic models
        ├── core/
        │   ├── drive.py    # Google Drive integration
        │   ├── engine.py   # Whisper transcription engine
        │   ├── hardware.py # Hardware detection (CPU/CUDA/MPS)
        │   └── io.py       # File operations
        └── utils/
            ├── console.py  # Console utilities
            ├── logger.py   # Rich logging utilities
            └── ui.py       # Progress bars and UI
```

## Development Workflow

### Adding New Features

1. Implement feature in appropriate module under `src/gtranscriber/`
2. Add type annotations to all functions
3. Run `ruff check --fix .` to fix auto-fixable issues
4. Run `ruff format .` to format code
5. Test manually with the CLI: `gtranscriber <command>`

### Working with Dependencies

1. **Check before adding**: Always verify if functionality can be achieved with existing dependencies
2. **Use uv**: Add dependencies with `uv add <package>`
3. **Lock file**: The `uv.lock` file tracks exact versions - commit changes to this file

### Hardware Compatibility

The application supports multiple compute devices:
- **CPU**: Default fallback
- **CUDA**: NVIDIA GPU support
- **MPS**: Apple Silicon support

When writing code that interacts with ML models:
- Use the hardware detection utilities in `core/hardware.py`
- Ensure code works across all device types
- Support optional 8-bit quantization for memory efficiency

## Testing

### Manual Testing

Use the CLI to test functionality:

```bash
# Test local transcription
gtranscriber transcribe <audio-file>

# Test with specific model
gtranscriber transcribe <audio-file> --model-id openai/whisper-large-v3-turbo

# Test with quantization
gtranscriber transcribe <audio-file> --quantize

# Test system info
gtranscriber info

# Test Google Drive integration
gtranscriber drive-transcribe <file-id> --credentials credentials.json
```

## Common Patterns

### Error Handling

- Use tenacity for retry logic on network operations
- Provide informative error messages using Rich console
- Handle hardware-specific errors gracefully

### Configuration

- Support both environment variables and CLI arguments
- CLI arguments take precedence over environment variables
- Use Pydantic for configuration validation

### CLI Development

- Use Typer for CLI commands
- Use Rich for beautiful output
- Provide progress bars for long-running operations
- Include helpful command descriptions and examples

## Best Practices

1. **Type Safety**: Always use type hints for function parameters and return values
2. **Code Quality**: Run ruff before committing changes
3. **Documentation**: Update docstrings for public APIs
4. **Dependencies**: Minimize external dependencies; use uv for management
5. **Compatibility**: Ensure code works on CPU, CUDA, and MPS devices
6. **Error Messages**: Provide clear, actionable error messages
7. **Configuration**: Support flexible configuration via CLI and environment variables
8. **Output**: Use structured JSON output for transcription results

## Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
