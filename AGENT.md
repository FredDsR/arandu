# AI Agent Development Guidelines

## Project Description

This project aims to implement the composable pipelines for the transcription of audio/video files from Google Drive, generation of Knowledge Graphs, QA validation datasets, and other evaluation methods. The research in question investigates the power of tacit knowledge elicitation using Knowledge Graphs generation with LLMs in the context of riverine people affected by critical climate events on southern Brazil.

**CRITICAL**: All guidelines are mandatory. Code that violates these standards will be rejected.

## Quick Start Checklist

Before making any changes:

1. **Git Workflow**: Use conventional commits (`feat(scope): description`) and conventional branches (`feature/description`)
2. **Code Style**: Pass Ruff linting (100 char line limit)
3. **Type Annotations**: Every function argument and return value must be typed
4. **Docstrings**: Use Google style for all public functions/classes
5. **Testing**: Write tests and run pytest before committing
6. **No `print()`**: Use Rich console utilities (`print_info`, `print_error`, etc.)

## Standard Development Workflow

**Follow this process for every task:**

1. **Create branch**: `git checkout -b <type>/<description>`
2. **Make changes**: Follow existing code patterns in the codebase
3. **Add docstrings**: Google style for all new/modified public functions
4. **Type annotations**: Ensure all arguments and returns are typed
5. **Lint and format**: `ruff check --fix src/ && ruff format src/`
6. **Test**: `pytest` (write tests if adding features)
7. **Commit**: Use conventional commit format
8. **Verify**: All checks pass before pushing

## Git Workflow

### Conventional Commits

**Format**: `<type>(<scope>): <description>`

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without behavior changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system or dependency changes
- `ci`: CI/CD configuration changes
- `chore`: Other changes (maintenance, tooling)
- `revert`: Revert a previous commit

**Scope** (optional): Component affected (`transcribe`, `qa`, `kg`, `cli`, `docker`)

**Description**: Brief summary in imperative mood ("add feature" not "added feature")

**Breaking changes**: Add `!` after type/scope and include `BREAKING CHANGE:` in footer

**Examples**:

```bash
feat(qa): add support for custom system prompts
fix(transcribe): resolve CUDA memory leak in batch processing
docs: update README with new environment variables
feat(cli)!: change transcribe command default model
```

### Conventional Branches

**Pattern**: `<type>/<short-description>`

**Types**: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`

**Rules**:

- Use lowercase with hyphens
- Keep descriptions short (2-4 words)
- Be descriptive but concise

**Examples**:

```bash
feature/ollama-streaming
fix/cuda-memory-leak
docs/api-reference
refactor/llm-client
```

## Documentation Standards

### Google Style Docstrings

All public functions, classes, and modules MUST use [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

**Structure**:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief summary in one line.

    Longer description if needed (optional).

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this exception occurs.

    Examples:
        >>> function_name(1, 2)
        3
    """
```

**Sections** (use as needed):

- **Args**: Function/method parameters
- **Returns**: Return value description
- **Raises**: Exceptions that may be raised
- **Yields**: For generator functions
- **Attributes**: Class attributes (public only)
- **Examples**: Usage examples (encouraged for non-trivial functions)

**Rules**:

1. First line is a brief summary (imperative mood)
2. Leave blank line after summary if adding details
3. Document all public functions, classes, and methods
4. Private functions can have shorter docstrings
5. Keep line length under 100 characters

## Type Annotations

**MANDATORY**: Strict type annotations enforced by Ruff.

**Requirements**:

- All function arguments MUST have type annotations (`ANN001`)
- All public functions MUST have return type annotations (`ANN201`)
- All private functions MUST have return type annotations (`ANN202`)

**Exception**: CLI command functions in `main.py` are exempt.

**Patterns**:

- Use `|` for unions: `str | None`
- Use `list[type]`, `dict[key, value]` for generics
- Import `from __future__ import annotations` at top of file
- Use `Literal` for constrained string values

## Design Principles

**MANDATORY**: Follow these principles when writing or modifying code.

### DRY (Don't Repeat Yourself)

Every piece of knowledge must have a single, unambiguous, authoritative representation. If you find yourself copying logic, extract it into a shared function, base class, or utility.

- Extract repeated logic into helper functions or shared modules
- Use Pydantic base models for common schema patterns
- Centralize configuration in `BaseSettings` classes
- Reuse existing utilities from `arandu.utils` before creating new ones

### SOLID

**Single Responsibility Principle (SRP)**: Each module, class, or function should have one reason to change. A pipeline step should not also handle logging configuration; a schema should not also manage file I/O unrelated to its data.

**Open/Closed Principle (OCP)**: Design modules that are open for extension but closed for modification. Use protocols, abstract base classes, or strategy patterns so new behavior (e.g., a new LLM provider) can be added without altering existing code.

**Liskov Substitution Principle (LSP)**: Subtypes must be substitutable for their base types. If a function accepts a `BaseProcessor`, any subclass must work without breaking expectations on inputs, outputs, or side effects.

**Interface Segregation Principle (ISP)**: Prefer small, focused interfaces over large ones. Don't force classes to implement methods they don't use. Use Python `Protocol` classes to define narrow contracts.

**Dependency Inversion Principle (DIP)**: Depend on abstractions, not concretions. High-level pipeline logic should not import low-level provider SDKs directly — use the unified `LLMClient` and injected dependencies.

### KISS (Keep It Simple, Stupid)

Prefer the simplest solution that meets the requirements. Avoid premature abstraction, unnecessary indirection, and clever tricks that obscure intent.

- Write straightforward, readable code over "elegant" but opaque alternatives
- Avoid deep inheritance hierarchies — prefer composition
- If a function needs a long explanation, it's probably too complex — split it

### YAGNI (You Aren't Gonna Need It)

Do not implement functionality until it is actually needed. Speculative features increase maintenance burden and code complexity.

- Don't add parameters, flags, or configuration for hypothetical future use cases
- Don't build abstractions for a single concrete implementation
- Remove dead code — don't comment it out "just in case"

### Composition Over Inheritance

Favor composing objects with well-defined interfaces over building deep inheritance trees. This project's pipeline architecture relies on composable steps, not monolithic class hierarchies.

- Use dependency injection to assemble behavior at runtime
- Prefer `Protocol` classes over abstract base classes when possible
- Build pipelines from small, reusable, independently testable components

### Separation of Concerns

Keep distinct responsibilities in distinct modules. Each layer of the system should handle its own concern without leaking into others.

- CLI layer: argument parsing, user interaction, output formatting
- Core layer: business logic, pipeline orchestration, LLM interactions
- Schema layer: data validation, serialization, model definitions
- Utils layer: cross-cutting concerns (logging, file helpers, retry logic)

### Law of Demeter (Principle of Least Knowledge)

A module should only talk to its immediate collaborators. Avoid chaining through objects to reach deeply nested state (`obj.a.b.c.do_thing()`).

- Pass required data directly rather than passing entire objects for a single attribute
- Keep function signatures focused on what the function actually needs

## Code Conventions

### Imports

**Order** (enforced by Ruff isort):

1. Standard library
2. Third-party packages
3. First-party (arandu)

### Configuration

- Use `pydantic-settings` with `BaseSettings`
- All env vars use `ARANDU_` prefix
- Use `Field()` with descriptions

### Schemas

- Use Pydantic models for all data structures
- Add validators with `@field_validator` or `@model_validator`
- Include `save()` and `load()` methods for persistent schemas

### CLI

- Use Typer with Rich integration
- Load defaults from config classes
- Use `Annotated` for all parameters

### LLM Interactions

- Always use unified `LLMClient` from `arandu.core.llm_client`
- Never use provider SDKs (OpenAI, Ollama) directly

### Error Handling

- Never use `print()` - use Rich console utilities
- Import from `arandu.utils.logger`: `print_error`, `print_success`, `print_info`, `print_warning`
- Use `typer.Exit(code=1)` for CLI errors

### Retry Logic

- Use `tenacity` for operations that may fail transiently
- Typical pattern: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`

## Linting with Ruff

**MANDATORY**: All code MUST pass Ruff checks before committing.

**Requirements**:

- Line length: 100 characters maximum
- Run before every commit: `uv run ruff check --fix src/ && uv run ruff format src/`

**Commands**:

```bash
# Check and auto-fix
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

## Testing

**Framework**: pytest

**Requirements**:

1. Place tests in `tests/` mirroring `src/` structure
2. Mock all external services (Google Drive, Ollama, OpenAI)
3. Test Pydantic model validation with invalid inputs
4. Test error paths, not just happy paths

**Commands**:

```bash
uv run pytest                              # Run all tests
uv run pytest tests/core/test_engine.py   # Run specific test
uv run pytest --cov=arandu          # Run with coverage
```

## Common Mistakes to Avoid

| ❌ Don't | ✅ Do |
| -------- | ----- |
| Hardcode paths | Use configuration or environment variables |
| Skip type annotations | Annotate all functions and arguments |
| Use `print()` for output | Use Rich console utilities |
| Ignore error handling | Use try/except with user-friendly messages |
| Skip validation | Use Pydantic models for data structures |
| Use provider SDKs directly | Use unified `LLMClient` |
| Commit without testing | Run `ruff check`, `ruff format`, and tests |
| Use inconsistent commits | Follow Conventional Commits |
| Duplicate logic across modules | Extract into shared utilities (DRY) |
| Build deep inheritance hierarchies | Use composition and `Protocol` classes (SOLID) |
| Add speculative features | Implement only what is needed now (YAGNI) |
| Write complex "clever" code | Keep it simple and readable (KISS) |
| Chain through nested objects | Pass required data directly (Law of Demeter) |

## Quick Reference

### Pre-Commit Checklist

```bash
# Run all quality checks
uv run ruff check --fix src/ && uv run ruff format src/ && uv run pytest
```

### Development Setup

```bash
uv sync
uv run arandu --help
```

### Key Standards Summary

| Aspect | Requirement | Example |
| ------ | ----------- | ------- |
| **Commits** | Conventional Commits | `feat(qa): add custom prompts` |
| **Branches** | `<type>/<description>` | `feature/ollama-streaming` |
| **Docstrings** | Google style | See Documentation Standards |
| **Type Hints** | All args and returns | `def foo(x: int) -> str:` |
| **Line Length** | 100 characters max | Enforced by Ruff |
| **Output** | Rich console only | `print_info()`, never `print()` |
| **LLM Calls** | Unified `LLMClient` | Never use provider SDKs directly |
| **Design** | DRY, SOLID, KISS, YAGNI | See Design Principles |

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [DRY Principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
- [KISS Principle](https://en.wikipedia.org/wiki/KISS_principle)
- [YAGNI Principle](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)
- [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter)
- [Composition Over Inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance)
