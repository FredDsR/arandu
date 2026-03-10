# Codebase Reorganization Status

**Branch:** `refactor/codebase-reorganization`
**Design doc:** `docs/plans/2026-03-04-codebase-reorganization-design.md`
**Full plan:** `docs/plans/2026-03-04-codebase-reorganization-plan.md`

## Completed (6/12 tasks)

| # | Commit | What moved |
|---|--------|-----------|
| 1 | `8cdb196` | Package scaffolding (cli/, shared/, transcription/, qa/, report/charts/) |
| 2 | `c5129ac` | checkpoint, hardware, io, llm_client, drive, results_manager → `shared/` |
| 3 | `5e8ae02` | `schemas.py` split → `shared/schemas.py`, `qa/schemas.py`, `kg/schemas.py` |
| 4 | `ca15155` | `config.py` split → `transcription/config.py`, `qa/config.py`, `kg/config.py`, `shared/config.py` |
| 5 | `a9e1f93` | engine, batch, validator, media → `transcription/` |
| 6 | `a78bf9d` | metadata/ → `arandu.metadata` (from `core/metadata/`) |

All 965 tests passing after each commit. Old `schemas.py` and `config.py` deleted.

## Remaining (6 tasks)

| # | Task | Key actions |
|---|------|------------|
| 7 | Move qa/ domain | `core/cep/` → `qa/cep/`, `core/judge/` → `qa/judge/`, `core/qa_batch.py` → `qa/batch.py`, rename `cep_generator.py` → `generator.py` |
| 8 | Move kg/ domain | `core/kg/` → `kg/`, merge `core/kg/schemas.py` (KGConstructionResult) into `kg/schemas.py` |
| 9 | Move report/ + split charts | `core/report/` → `report/`, split 996-line `charts.py` into 6 themed files under `report/charts/`, rename `api_schemas.py` → `schemas.py` |
| 10 | Split main.py into cli/ | 2,116-line `main.py` → `cli/app.py`, `cli/transcribe.py`, `cli/qa.py`, `cli/kg.py`, `cli/report.py`, `cli/manage.py`. Update pyproject.toml entry point to `arandu.cli.app:app` |
| 11 | Clean up core/ | Remove empty `core/` and `tests/core/`, update AGENTS.md paths |
| 12 | Final verification | Verify no `arandu.core` imports remain, CLI works, no circular imports |

## Import update strategy

Using Python scripts that map each symbol to its new module and do project-wide find-and-replace. After each move: `ruff check --fix --unsafe-fixes` then `pytest`.
