# Contributing to Arandu

Thank you for your interest in contributing to Arandu. These guidelines apply
equally to human contributors and automated coding agents.

## Development Setup

Arandu requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras --all-groups   # install base package, optional extras, and dev/test tools
uv run arandu --help
```

> **Note on `uv sync`:** pairing `--all-extras` with `--all-groups` (or
> `--group test`) ensures `pytest-mock` and optional extras (`kg`, `report`,
> `dashboard`, `analysis`) are all available.

## Repository Layout

- `src/arandu/` — core package, split by concern:
  - `cli/` — Typer CLI entry points (argument parsing and Rich output only)
  - `transcription/`, `qa/`, `kg/`, `report/`, `metadata/` — domain pipelines
  - `shared/` — cross-domain schemas, configuration, `LLMClient`, judges, chunking, RAG
  - `utils/` — logging, paths, retries, concurrency helpers
- `prompts/` — versioned prompt templates and judge criterion configs
- `analysis/` — read-only analysis modules over finished runs (`results/<run-id>/`)
- `scripts/` — reusable operational utilities (`export_zenodo_release.py`, cost estimation, smoke checks)
- `docs/` — user guides, pipeline references, architecture, schemas, and annotator instruments
- `tests/` — pytest suite mirroring `src/`, `analysis/`, and `scripts/`

Cluster-specific deployment manifests (e.g., SLURM partition scripts) are kept
in the separate [`arandu-deploy`](https://github.com/FredDsR/arandu-deploy)
repository.

## Coding Standards

1. **Formatting and linting (Ruff):** 100-character line limit; imports sorted
   by standard library, third-party, first-party (`arandu`).
2. **Type annotations:** mandatory for all function arguments and return types
   (`from __future__ import annotations`, PEP 604 `X | Y` unions). CLI command
   functions in `src/arandu/cli/` are exempt.
3. **Docstrings:** Google style for all public modules, classes, and functions.
4. **Configuration & schemas:** use `pydantic-settings` (`BaseSettings` with
   `ARANDU_` env prefixes) and Pydantic v2 models with validators.
5. **LLM calls:** always route through the unified `LLMClient` in
   `arandu.shared.llm_client`; never call provider SDKs directly from domain
   modules.
6. **Console output:** inside `src/arandu/` and `scripts/`, use Rich helpers
   from `arandu.utils.logger` (`print_info`, `print_warning`, `print_error`,
   `print_success`) rather than bare `print()`.
7. **Persistence:** pipeline stages write under `results/<run-id>/<stage>/outputs/`
   via `ResultsManager` (see [architecture.md](docs/development/architecture.md)).
8. **Design principles:** keep modules focused (DRY, SOLID, KISS, YAGNI) and
   favor composition and `Protocol` interfaces over deep inheritance.

### Rules for `analysis/` and `scripts/`

- **`analysis/` modules must be read-only** and make no LLM calls. Run them as
  modules from the repo root (`uv run python -m analysis.<module> --id <run-id>`),
  never hardcode absolute paths or dataset identifiers, and register new modules
  in [analysis/README.md](analysis/README.md).
- **`scripts/` utilities** operate on or package runs. Register reusable tools
  in [scripts/README.md](scripts/README.md) and prefix smoke checks with
  `smoke_` (never `test_`).

## Testing

- Place tests in `tests/` mirroring the source path.
- Mock all external services (Google Drive, Ollama, OpenAI, Label Studio) and
  use `tmp_path` fixtures instead of real pipeline runs.
- Guard tests that rely on optional extras (`report`, `analysis`, `kg`) with
  `pytest.importorskip(...)` so the default CI suite skips them cleanly when an
  extra is absent.

## Quality Checks

Run before opening a pull request:

```bash
uv run ruff check --fix src/ tests/ analysis/ scripts/
uv run ruff format src/ tests/ analysis/ scripts/
uv run pytest
```

## Git Workflow

- **Branches:** `<type>/<short-description>` (e.g., `feature/dense-retriever`,
  `fix/chunk-boundaries`, `docs/cli-reference`).
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`<type>(<scope>): <imperative description>`, using `feat`, `fix`, `docs`,
  `refactor`, `test`, `perf`, `build`, `ci`, `chore`).
- Never commit raw interview data, participant identifiers, Google Drive
  catalogs/credentials (`.env`, `credentials.json`, `token.json`), or notebook
  outputs containing corpus excerpts.
