# Codebase Reorganization Design

**Date**: 2026-03-04
**Goal**: Reorganize the Arandu codebase for maintainability — clear domain boundaries, small focused files, minimal cross-domain coupling.
**Approach**: Pipeline-Centric Restructure (Approach A) with clean break (no backward-compatible re-exports).

## Design Principles Applied

- **Dependency Inversion (DIP)**: No domain imports from another domain. Cross-domain contracts live in `shared/`.
- **Single Responsibility**: Each module/file has one reason to change.
- **Separation of Concerns**: CLI, domain logic, schemas, config, and utilities are distinct layers.
- **KISS**: Simplest grouping that achieves isolation — no unnecessary abstractions.

## Current Pain Points

| File | Lines | Problem |
|------|-------|---------|
| `main.py` | 2,116 | Monolithic CLI with 15+ commands |
| `charts.py` | 996 | 16 chart functions in one file |
| `schemas.py` | 895 | 25+ models for all pipelines |
| `config.py` | 787 | All config classes in one file |
| `results_manager.py` | 664 | Large single-class module |
| `batch.py` | 654 | Mixed orchestration + task logic |
| Flat `core/` | — | No grouping by pipeline domain |

## Target Package Structure

```
src/arandu/
├── __init__.py
├── cli/                        # CLI layer (split from main.py)
│   ├── __init__.py
│   ├── app.py                  # Typer app, shared callbacks, version, info()
│   ├── transcribe.py           # transcribe, drive_transcribe, batch_transcribe, validate_transcriptions
│   ├── qa.py                   # generate_cep_qa
│   ├── kg.py                   # build_kg
│   ├── report.py               # report, serve_report
│   └── manage.py               # list_runs, run_info, replicate, refresh_auth, rebuild_index, enrich_metadata
│
├── transcription/              # Transcription pipeline domain
│   ├── __init__.py
│   ├── config.py               # TranscriberConfig, BatchConfig
│   ├── engine.py               # WhisperEngine
│   ├── batch.py                # run_batch_transcription, TranscriptionTask, load_catalog
│   ├── validator.py            # Transcription quality validation
│   └── media.py                # Audio/video extraction, validation, ffprobe
│
├── qa/                         # QA/CEP generation pipeline domain
│   ├── __init__.py
│   ├── config.py               # QAConfig, CEPConfig
│   ├── schemas.py              # QAPair, QAPairCEP, CriterionScore, ValidationScore,
│   │                           #   QAPairValidated, QARecordCEP
│   ├── batch.py                # run_batch_cep_generation, worker init, QAGenerationTask
│   ├── cep/                    # Cognitive Elicitation Pipeline
│   │   ├── __init__.py
│   │   ├── bloom_scaffolding.py
│   │   ├── generator.py        # (renamed from cep_generator.py)
│   │   ├── reasoning.py
│   │   └── validator.py
│   └── judge/                  # LLM-as-a-Judge
│       ├── __init__.py
│       ├── criterion.py
│       ├── registry.py
│       └── pipeline.py
│
├── kg/                         # Knowledge Graph pipeline domain
│   ├── __init__.py
│   ├── config.py               # KGConfig
│   ├── schemas.py              # KGMetadata, GraphConnectivity, EntityCoverageResult,
│   │                           #   RelationMetricsResult, SemanticQualityResult, EvaluationReport
│   ├── batch.py
│   ├── atlas_backend.py
│   ├── factory.py
│   └── protocol.py
│
├── report/                     # Reporting & visualization domain
│   ├── __init__.py
│   ├── schemas.py              # API response schemas (from api_schemas.py)
│   ├── api.py                  # REST API endpoints
│   ├── service.py              # Report service layer
│   ├── dataset.py              # Dataset loading/caching
│   ├── charts/                 # Split from 996-line charts.py
│   │   ├── __init__.py         # Re-exports all chart functions
│   │   ├── quality.py          # transcription_quality, quality_radar, confidence_distribution
│   │   ├── distribution.py     # bloom_distribution, participant_breakdown, location_treemap
│   │   ├── validation.py       # validation_scores, bloom_validation_heatmap, correlation_heatmap
│   │   ├── timeline.py         # pipeline_overview, run_timeline, funnel
│   │   ├── comparison.py       # cross_run_comparison, parallel_coordinates, location_quality
│   │   ├── multihop.py         # multihop_chart
│   │   └── style.py            # Chart styling
│   ├── collector.py
│   ├── exporter.py
│   ├── generator.py
│   └── templates/              # HTML/CSS templates (unchanged)
│
├── metadata/                   # Metadata extraction domain
│   ├── __init__.py
│   ├── extractor.py            # GDriveCatalogExtractor
│   ├── enrichment.py
│   └── protocol.py
│
├── shared/                     # Cross-cutting infrastructure
│   ├── __init__.py
│   ├── config.py               # LLMConfig, EvaluationConfig, ResultsConfig
│   ├── schemas.py              # InputRecord, TranscriptionSegment, TranscriptionQualityScore,
│   │                           #   SourceMetadata, EnrichedRecord, PipelineType, ReplicationInfo,
│   │                           #   PipelineMetadata, RunStatus, ExecutionEnvironment, HardwareInfo,
│   │                           #   ConfigSnapshot, RunMetadata
│   ├── llm_client.py           # Unified LLM client
│   ├── drive.py                # DriveClient + exception classes
│   ├── checkpoint.py           # CheckpointManager
│   ├── results_manager.py      # ResultsManager
│   ├── hardware.py             # GPU/device detection
│   └── io.py                   # File I/O utilities
│
└── utils/                      # Shared utilities (unchanged)
    ├── __init__.py
    ├── logger.py
    ├── text.py
    ├── console.py
    └── ui.py
```

## Dependency Rules

1. **No domain-to-domain imports** (except report/ which is a pure downstream consumer of qa/ and kg/ schemas)
2. **All domains may import from**: `shared/`, `utils/`
3. **`report/` may additionally import from**: `qa/schemas.py`, `kg/schemas.py` (directed, no cycles)
4. **`cli/` imports from**: all domains (it's the composition root)
5. **`shared/` and `utils/`**: import only from each other and stdlib/third-party

```
        cli/  (composition root)
       / | \ \ \
      v  v  v  v  v
transcription/  qa/  kg/  metadata/  report/
      \        |    |      /          / |
       v       v    v     v          v  v
            shared/              qa/ kg/
               |                (schemas only)
               v
            utils/
```

## Schema Distribution

### `shared/schemas.py` — Cross-domain contracts

Models consumed by 2+ pipeline domains:

- `InputRecord` — transcription + metadata
- `TranscriptionSegment` — embedded in EnrichedRecord
- `TranscriptionQualityScore` — embedded in EnrichedRecord
- `SourceMetadata` — transcription + metadata + KG
- `EnrichedRecord` — transcription + QA + KG
- `PipelineType`, `ReplicationInfo`, `PipelineMetadata` — pipeline infrastructure
- `RunStatus`, `ExecutionEnvironment`, `HardwareInfo`, `ConfigSnapshot`, `RunMetadata` — run tracking

### `qa/schemas.py` — QA-only models

- `QAPair`, `QAPairCEP`, `CriterionScore`, `ValidationScore`, `QAPairValidated`, `QARecordCEP`

### `kg/schemas.py` — KG-only models

- `KGMetadata`, `GraphConnectivity`, `EntityCoverageResult`, `RelationMetricsResult`, `SemanticQualityResult`, `EvaluationReport`

### `report/schemas.py` — API response models

- Existing `api_schemas.py` content (renamed)

## Config Distribution

| Class | Location | Used by |
|-------|----------|---------|
| `TranscriberConfig` | `transcription/config.py` | transcription pipeline |
| `BatchConfig` | `transcription/config.py` | transcription batch |
| `QAConfig` | `qa/config.py` | QA pipeline |
| `CEPConfig` | `qa/config.py` | CEP pipeline |
| `KGConfig` | `kg/config.py` | KG pipeline |
| `LLMConfig` | `shared/config.py` | QA, KG, CEP |
| `EvaluationConfig` | `shared/config.py` | QA judge, KG evaluation |
| `ResultsConfig` | `shared/config.py` | all pipelines |

## CLI Split (`main.py` 2,116 lines → `cli/`)

| File | Commands | Est. lines |
|------|----------|-----------|
| `cli/app.py` | Typer app, shared callbacks, `info()`, version | ~100 |
| `cli/transcribe.py` | `transcribe`, `drive_transcribe`, `batch_transcribe`, `validate_transcriptions` | ~600 |
| `cli/qa.py` | `generate_cep_qa` | ~300 |
| `cli/kg.py` | `build_kg` | ~200 |
| `cli/report.py` | `report`, `serve_report` | ~300 |
| `cli/manage.py` | `list_runs`, `run_info`, `replicate`, `refresh_auth`, `rebuild_index`, `enrich_metadata` | ~600 |

`app.py` creates the Typer app and registers subcommand modules. Entry point in `pyproject.toml` points to `cli.app:app`.

## Charts Decomposition (`charts.py` 996 lines → `report/charts/`)

| File | Chart functions | Est. lines |
|------|----------------|-----------|
| `charts/__init__.py` | Re-exports | ~30 |
| `charts/quality.py` | transcription_quality, quality_radar, confidence_distribution | ~200 |
| `charts/distribution.py` | bloom_distribution, participant_breakdown, location_treemap | ~200 |
| `charts/validation.py` | validation_scores, bloom_validation_heatmap, correlation_heatmap | ~200 |
| `charts/timeline.py` | pipeline_overview, run_timeline, funnel | ~180 |
| `charts/comparison.py` | cross_run_comparison, parallel_coordinates, location_quality | ~150 |
| `charts/multihop.py` | multihop_chart | ~60 |
| `charts/style.py` | Styling utilities | ~110 |

## Test Structure

Tests mirror the new source layout:

```
tests/
├── conftest.py
├── cli/
│   ├── test_transcribe.py
│   ├── test_qa.py
│   ├── test_kg.py
│   ├── test_report.py
│   └── test_manage.py
├── transcription/
│   ├── test_engine.py
│   ├── test_batch.py
│   ├── test_validator.py
│   ├── test_media.py
│   └── test_drive.py
├── qa/
│   ├── test_batch.py
│   ├── cep/
│   │   ├── test_bloom_scaffolding.py
│   │   ├── test_generator.py
│   │   ├── test_reasoning.py
│   │   └── test_validator.py
│   └── judge/
│       ├── test_criterion.py
│       ├── test_registry.py
│       └── test_pipeline.py
├── kg/
│   ├── test_batch.py
│   ├── test_atlas_backend.py
│   ├── test_factory.py
│   └── test_protocol.py
├── report/
│   ├── test_api.py
│   ├── test_service.py
│   ├── test_dataset.py
│   ├── test_collector.py
│   ├── test_exporter.py
│   ├── test_generator.py
│   ├── test_style.py
│   └── charts/
│       ├── test_quality.py
│       ├── test_distribution.py
│       ├── test_validation.py
│       ├── test_timeline.py
│       ├── test_comparison.py
│       └── test_multihop.py
├── metadata/
│   ├── test_extractor.py
│   ├── test_enrichment.py
│   └── test_protocol.py
├── shared/
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_llm_client.py
│   ├── test_drive.py
│   ├── test_checkpoint.py
│   ├── test_results_manager.py
│   ├── test_hardware.py
│   └── test_io.py
├── utils/
│   ├── test_logger.py
│   ├── test_ui.py
│   ├── test_text.py
│   └── test_console.py
└── scripts/
    └── test_import_results.py
```

## Migration Strategy

1. **Bottom-up**: Start with leaf modules (no internal dependents), move up
2. **Order**: `utils/` (unchanged) → `shared/` → `metadata/` → `transcription/` → `qa/` → `kg/` → `report/` → `cli/`
3. **Per module**: Move file → update imports in that file → update all importers → run tests
4. **Clean break**: No re-exports from old locations
5. **One domain at a time**: Each domain move is a separate commit for easy bisecting
