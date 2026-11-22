# Agent Guidelines

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes — it is the
single source of truth for setup, repository layout, coding standards, testing,
and git conventions for both human developers and coding agents.

## Quick Verification Before Finishing

```bash
uv run ruff check src/ tests/ analysis/ scripts/
uv run ruff format --check src/ tests/ analysis/ scripts/
uv run pytest
```
