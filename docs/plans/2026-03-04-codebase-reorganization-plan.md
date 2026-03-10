# Codebase Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the Arandu codebase from a flat `core/` layout with monolithic files into pipeline-centric domains with strict dependency isolation.

**Architecture:** Pipeline-Centric with `shared/` for cross-domain contracts. Each pipeline (transcription, qa, kg, report, metadata) becomes a top-level package. `cli/` replaces the monolithic `main.py`. All cross-domain schemas and configs live in `shared/`.

**Tech Stack:** Python 3.13, Pydantic v2, Typer, pytest, Ruff

**Design doc:** `docs/plans/2026-03-04-codebase-reorganization-design.md`

---

## Migration Strategy

Each task moves one domain at a time. After each task, **all existing tests must pass** (with updated import paths). The order is bottom-up: leaf modules first, then modules that depend on them.

**Critical rule:** After every task, run `uv run pytest` and `uv run ruff check src/ tests/` to verify nothing is broken.

**Import update strategy:** For each moved file, use project-wide find-and-replace on the old import path to the new one, then verify with `ruff check` and `pytest`.

---

### Task 1: Create package scaffolding

**Files:**
- Create: `src/arandu/cli/__init__.py`
- Create: `src/arandu/shared/__init__.py`
- Create: `src/arandu/transcription/__init__.py`
- Create: `src/arandu/qa/__init__.py`
- Create: `src/arandu/report/__init__.py` (replace existing `core/report/__init__.py`)
- Create: `src/arandu/report/charts/__init__.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/shared/__init__.py`
- Create: `tests/transcription/__init__.py`
- Create: `tests/qa/__init__.py`
- Create: `tests/report/__init__.py`
- Create: `tests/report/charts/__init__.py`

**Step 1:** Create all new package directories and empty `__init__.py` files.

```bash
mkdir -p src/arandu/{cli,shared,transcription,qa}
mkdir -p src/arandu/report/charts
mkdir -p tests/{cli,shared,transcription,qa}
mkdir -p tests/report/charts

touch src/arandu/cli/__init__.py
touch src/arandu/shared/__init__.py
touch src/arandu/transcription/__init__.py
touch src/arandu/qa/__init__.py
touch src/arandu/report/charts/__init__.py
touch tests/cli/__init__.py
touch tests/shared/__init__.py
touch tests/transcription/__init__.py
touch tests/qa/__init__.py
touch tests/report/__init__.py
touch tests/report/charts/__init__.py
```

**Step 2:** Verify no tests break.

Run: `uv run pytest --tb=short -q`
Expected: All tests pass (no functional change).

**Step 3:** Commit.

```bash
git add -A
git commit -m "chore: create package scaffolding for codebase reorganization"
```

---

### Task 2: Move `shared/` infrastructure — leaf modules

Move modules with no arandu imports first: `checkpoint.py`, `media.py`. Then modules with minimal imports: `hardware.py`, `io.py`, `llm_client.py`, `drive.py`, `results_manager.py`.

**Files:**
- Move: `src/arandu/core/checkpoint.py` → `src/arandu/shared/checkpoint.py`
- Move: `src/arandu/core/hardware.py` → `src/arandu/shared/hardware.py`
- Move: `src/arandu/core/io.py` → `src/arandu/shared/io.py`
- Move: `src/arandu/core/llm_client.py` → `src/arandu/shared/llm_client.py`
- Move: `src/arandu/core/drive.py` → `src/arandu/shared/drive.py`
- Move: `src/arandu/core/results_manager.py` → `src/arandu/shared/results_manager.py`
- Move: `tests/core/test_checkpoint.py` → `tests/shared/test_checkpoint.py`
- Move: `tests/core/test_hardware.py` → `tests/shared/test_hardware.py`
- Move: `tests/core/test_io.py` → `tests/shared/test_io.py`
- Move: `tests/core/test_llm_client.py` → `tests/shared/test_llm_client.py`
- Move: `tests/core/test_drive.py` → `tests/shared/test_drive.py`
- Move: `tests/core/test_results_manager.py` → `tests/shared/test_results_manager.py`

**Step 1:** Move files with `git mv`.

```bash
git mv src/arandu/core/checkpoint.py src/arandu/shared/checkpoint.py
git mv src/arandu/core/hardware.py src/arandu/shared/hardware.py
git mv src/arandu/core/io.py src/arandu/shared/io.py
git mv src/arandu/core/llm_client.py src/arandu/shared/llm_client.py
git mv src/arandu/core/drive.py src/arandu/shared/drive.py
git mv src/arandu/core/results_manager.py src/arandu/shared/results_manager.py
git mv tests/core/test_checkpoint.py tests/shared/test_checkpoint.py
git mv tests/core/test_hardware.py tests/shared/test_hardware.py
git mv tests/core/test_io.py tests/shared/test_io.py
git mv tests/core/test_llm_client.py tests/shared/test_llm_client.py
git mv tests/core/test_drive.py tests/shared/test_drive.py
git mv tests/core/test_results_manager.py tests/shared/test_results_manager.py
```

**Step 2:** Update ALL imports across src/ and tests/.

Find and replace these import paths project-wide:

| Old path | New path |
|----------|----------|
| `arandu.core.checkpoint` | `arandu.shared.checkpoint` |
| `arandu.core.hardware` | `arandu.shared.hardware` |
| `arandu.core.io` | `arandu.shared.io` |
| `arandu.core.llm_client` | `arandu.shared.llm_client` |
| `arandu.core.drive` | `arandu.shared.drive` |
| `arandu.core.results_manager` | `arandu.shared.results_manager` |

Also update internal imports within the moved files (e.g., `hardware.py` has a lazy import of `arandu.config.TranscriberConfig` — this will be updated in a later task when config is split; for now just update the module path references for files that moved).

**Step 3:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run pytest --tb=short -q
```

Expected: All tests pass.

**Step 4:** Commit.

```bash
git add -A
git commit -m "refactor: move shared infrastructure to arandu.shared"
```

---

### Task 3: Split `schemas.py` into `shared/schemas.py`, `qa/schemas.py`, `kg/schemas.py`

This is the most import-intensive change. The current `arandu.schemas` is imported by almost every module.

**Files:**
- Create: `src/arandu/shared/schemas.py` — cross-domain models
- Create: `src/arandu/qa/schemas.py` — QA-only models
- Create: `src/arandu/kg/schemas.py` — KG-only models
- Modify: `src/arandu/schemas.py` — becomes a re-export shim temporarily, then delete
- Move: `tests/test_schemas.py` → split into `tests/shared/test_schemas.py`, `tests/qa/test_schemas.py`, `tests/kg/test_schemas.py`

**Step 1:** Create `src/arandu/shared/schemas.py` with these classes (copy from `schemas.py`):
- `InputRecord` (lines 17–64)
- `TranscriptionSegment` (lines 66–72)
- `TranscriptionQualityScore` (lines 74–96)
- `SourceMetadata` (lines 98–131)
- `EnrichedRecord` (lines 133–174)
- `BloomLevel` type alias (line 222)
- `_utc_now()` helper (lines 621–623)
- `PipelineType` (lines 626–634)
- `ReplicationInfo` (lines 636–647)
- `PipelineMetadata` (lines 649–687)
- `RunStatus` (lines 689–697)
- `ExecutionEnvironment` (lines 699–735)
- `HardwareInfo` (lines 737–783)
- `ConfigSnapshot` (lines 785–815)
- `RunMetadata` (lines 817–896)

Include all necessary imports (pydantic, datetime, pathlib, etc.).

**Step 2:** Create `src/arandu/qa/schemas.py` with these classes:
- `QAPair` (lines 181–213) — note: uses `BloomLevel` from shared
- `QAPairCEP` (lines 225–272) — extends `QAPair`, uses `SourceMetadata` from shared
- `CriterionScore` (lines 274–291)
- `ValidationScore` (lines 293–341) — uses `CriterionScore`
- `QAPairValidated` (lines 343–350) — extends `QAPairCEP`, uses `ValidationScore`
- `QARecordCEP` (lines 352–433) — uses `SourceMetadata`, `QAPairValidated`, `QAPairCEP`

Import cross-domain types from `arandu.shared.schemas`:
```python
from arandu.shared.schemas import BloomLevel, SourceMetadata
```

**Step 3:** Create `src/arandu/kg/schemas.py` with these classes:
- `KGMetadata` (lines 443–470)
- `GraphConnectivity` (lines 477–484)
- `EntityCoverageResult` (lines 486–503)
- `RelationMetricsResult` (lines 505–520) — uses `GraphConnectivity`
- `SemanticQualityResult` (lines 522–532)
- `EvaluationReport` (lines 534–614) — uses the above metric classes

No cross-domain imports needed — all KG schemas are self-contained.

**Step 4:** Delete `src/arandu/schemas.py`.

**Step 5:** Update ALL imports across src/ and tests/.

Every file that imports from `arandu.schemas` must be updated:

| Old import | New import |
|-----------|-----------|
| `from arandu.schemas import InputRecord` | `from arandu.shared.schemas import InputRecord` |
| `from arandu.schemas import EnrichedRecord` | `from arandu.shared.schemas import EnrichedRecord` |
| `from arandu.schemas import TranscriptionSegment` | `from arandu.shared.schemas import TranscriptionSegment` |
| `from arandu.schemas import SourceMetadata` | `from arandu.shared.schemas import SourceMetadata` |
| `from arandu.schemas import PipelineType` | `from arandu.shared.schemas import PipelineType` |
| `from arandu.schemas import RunMetadata` | `from arandu.shared.schemas import RunMetadata` |
| `from arandu.schemas import PipelineMetadata` | `from arandu.shared.schemas import PipelineMetadata` |
| `from arandu.schemas import QAPair` | `from arandu.qa.schemas import QAPair` |
| `from arandu.schemas import QAPairCEP` | `from arandu.qa.schemas import QAPairCEP` |
| `from arandu.schemas import QAPairValidated` | `from arandu.qa.schemas import QAPairValidated` |
| `from arandu.schemas import QARecordCEP` | `from arandu.qa.schemas import QARecordCEP` |
| `from arandu.schemas import CriterionScore` | `from arandu.qa.schemas import CriterionScore` |
| `from arandu.schemas import ValidationScore` | `from arandu.qa.schemas import ValidationScore` |
| `from arandu.schemas import KGMetadata` | `from arandu.kg.schemas import KGMetadata` |
| `from arandu.schemas import EvaluationReport` | `from arandu.kg.schemas import EvaluationReport` |
| etc. | (map each symbol to its new module) |

**Important:** Some files import multiple symbols in one `from arandu.schemas import (...)` statement. These must be split into multiple import lines if the symbols now live in different modules.

**Step 6:** Split `tests/test_schemas.py` into three files, moving test classes/functions to match the new schema locations.

**Step 7:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 8:** Commit.

```bash
git add -A
git commit -m "refactor: split schemas.py into shared/, qa/, and kg/ domain schemas"
```

---

### Task 4: Split `config.py` into domain configs

**Files:**
- Create: `src/arandu/shared/config.py` — LLMConfig, EvaluationConfig, ResultsConfig, TranscriptionQualityConfig
- Create: `src/arandu/transcription/config.py` — TranscriberConfig
- Create: `src/arandu/qa/config.py` — QAConfig, CEPConfig, JudgeConfig
- Create: `src/arandu/kg/config.py` — KGConfig
- Delete: `src/arandu/config.py`
- Split: `tests/test_config.py` → domain-specific test files

**Step 1:** Create each config file. All config classes are independent (`BaseSettings` with no cross-references), so they can be moved without dependency issues.

`src/arandu/shared/config.py`:
- `_get_default_temp_dir()` helper
- `LLMConfig` (lines 615–642)
- `EvaluationConfig` (lines 556–613)
- `ResultsConfig` (lines 644–666)
- `TranscriptionQualityConfig` (lines 668–743)
- Corresponding `get_*_config()` factory functions

`src/arandu/transcription/config.py`:
- `TranscriberConfig` (lines 28–143)
- `get_transcriber_config()`

`src/arandu/qa/config.py`:
- `QAConfig` (lines 145–224)
- `CEPConfig` (lines 226–421)
- `JudgeConfig` (lines 423–479)
- `get_qa_config()`, `get_cep_config()`, `get_judge_config()`

`src/arandu/kg/config.py`:
- `KGConfig` (lines 481–554)
- `get_kg_config()`

**Step 2:** Delete `src/arandu/config.py`.

**Step 3:** Update ALL imports project-wide. Map each config class to its new module:

| Old import | New import |
|-----------|-----------|
| `from arandu.config import TranscriberConfig` | `from arandu.transcription.config import TranscriberConfig` |
| `from arandu.config import QAConfig` | `from arandu.qa.config import QAConfig` |
| `from arandu.config import CEPConfig` | `from arandu.qa.config import CEPConfig` |
| `from arandu.config import JudgeConfig` | `from arandu.qa.config import JudgeConfig` |
| `from arandu.config import KGConfig` | `from arandu.kg.config import KGConfig` |
| `from arandu.config import LLMConfig` | `from arandu.shared.config import LLMConfig` |
| `from arandu.config import ResultsConfig` | `from arandu.shared.config import ResultsConfig` |
| `from arandu.config import EvaluationConfig` | `from arandu.shared.config import EvaluationConfig` |
| `from arandu.config import TranscriptionQualityConfig` | `from arandu.shared.config import TranscriptionQualityConfig` |
| `from arandu.config import get_judge_config` | `from arandu.qa.config import get_judge_config` |
| `from arandu.config import get_transcription_quality_config` | `from arandu.shared.config import get_transcription_quality_config` |

**Step 4:** Split `tests/test_config.py` into `tests/shared/test_config.py`, `tests/transcription/test_config.py`, `tests/qa/test_config.py`, `tests/kg/test_config.py`.

**Step 5:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 6:** Commit.

```bash
git add -A
git commit -m "refactor: split config.py into domain-specific config modules"
```

---

### Task 5: Move `transcription/` domain

**Files:**
- Move: `src/arandu/core/engine.py` → `src/arandu/transcription/engine.py`
- Move: `src/arandu/core/batch.py` → `src/arandu/transcription/batch.py`
- Move: `src/arandu/core/transcription_validator.py` → `src/arandu/transcription/validator.py`
- Move: `src/arandu/core/media.py` → `src/arandu/transcription/media.py`
- Move: `tests/core/test_engine.py` → `tests/transcription/test_engine.py`
- Move: `tests/core/test_batch.py` → `tests/transcription/test_batch.py`
- Move: `tests/core/test_transcription_validator.py` → `tests/transcription/test_validator.py`
- Move: `tests/core/test_media.py` → `tests/transcription/test_media.py`

**Step 1:** Move files with `git mv`.

```bash
git mv src/arandu/core/engine.py src/arandu/transcription/engine.py
git mv src/arandu/core/batch.py src/arandu/transcription/batch.py
git mv src/arandu/core/transcription_validator.py src/arandu/transcription/validator.py
git mv src/arandu/core/media.py src/arandu/transcription/media.py
git mv tests/core/test_engine.py tests/transcription/test_engine.py
git mv tests/core/test_batch.py tests/transcription/test_batch.py
git mv tests/core/test_transcription_validator.py tests/transcription/test_validator.py
git mv tests/core/test_media.py tests/transcription/test_media.py
```

**Step 2:** Update imports in moved files and all importers.

| Old path | New path |
|----------|----------|
| `arandu.core.engine` | `arandu.transcription.engine` |
| `arandu.core.batch` | `arandu.transcription.batch` |
| `arandu.core.transcription_validator` | `arandu.transcription.validator` |
| `arandu.core.media` | `arandu.transcription.media` |

Note: `batch.py` has a `BatchConfig` class defined inside it. This stays in `transcription/batch.py` (or optionally move to `transcription/config.py`).

**Step 3:** Update `transcription/batch.py` internal imports — it imports from many modules that have already moved:
- `arandu.shared.checkpoint` (already moved in Task 2)
- `arandu.shared.drive` (already moved in Task 2)
- `arandu.shared.results_manager` (already moved in Task 2)
- `arandu.shared.io` (already moved in Task 2)
- `arandu.transcription.engine` (just moved)
- `arandu.transcription.media` (just moved)
- `arandu.transcription.validator` (just moved)

**Step 4:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 5:** Commit.

```bash
git add -A
git commit -m "refactor: move transcription pipeline to arandu.transcription"
```

---

### Task 6: Move `metadata/` domain

**Files:**
- Move: `src/arandu/core/metadata/` → `src/arandu/metadata/`
- Move: `tests/core/metadata/` → `tests/metadata/`

**Step 1:** Move the entire directory.

```bash
# Remove old __init__.py from target if empty placeholder exists
rm -f src/arandu/metadata/__init__.py
git mv src/arandu/core/metadata/* src/arandu/metadata/
git mv tests/core/metadata/* tests/metadata/
```

**Step 2:** Update imports.

| Old path | New path |
|----------|----------|
| `arandu.core.metadata` | `arandu.metadata` |
| `arandu.core.metadata.extractor` | `arandu.metadata.extractor` |
| `arandu.core.metadata.enrichment` | `arandu.metadata.enrichment` |
| `arandu.core.metadata.protocol` | `arandu.metadata.protocol` |

**Step 3:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run pytest --tb=short -q
```

**Step 4:** Commit.

```bash
git add -A
git commit -m "refactor: move metadata domain to arandu.metadata"
```

---

### Task 7: Move `qa/` domain (cep/ + judge/ + qa_batch.py)

**Files:**
- Move: `src/arandu/core/cep/` → `src/arandu/qa/cep/`
- Move: `src/arandu/core/judge/` → `src/arandu/qa/judge/`
- Move: `src/arandu/core/qa_batch.py` → `src/arandu/qa/batch.py`
- Rename: `src/arandu/qa/cep/cep_generator.py` → `src/arandu/qa/cep/generator.py`
- Move: `tests/core/cep/` → `tests/qa/cep/`
- Move: `tests/core/judge/` → `tests/qa/judge/`
- Move: `tests/core/test_qa_batch.py` → `tests/qa/test_batch.py`

**Step 1:** Move files.

```bash
mkdir -p src/arandu/qa/{cep,judge}
mkdir -p tests/qa/{cep,judge}
git mv src/arandu/core/cep/* src/arandu/qa/cep/
git mv src/arandu/core/judge/* src/arandu/qa/judge/
git mv src/arandu/core/qa_batch.py src/arandu/qa/batch.py
# Rename cep_generator.py → generator.py
git mv src/arandu/qa/cep/cep_generator.py src/arandu/qa/cep/generator.py
git mv tests/core/cep/* tests/qa/cep/
git mv tests/core/judge/* tests/qa/judge/
git mv tests/core/test_qa_batch.py tests/qa/test_batch.py
```

**Step 2:** Update imports.

| Old path | New path |
|----------|----------|
| `arandu.core.cep` | `arandu.qa.cep` |
| `arandu.core.cep.bloom_scaffolding` | `arandu.qa.cep.bloom_scaffolding` |
| `arandu.core.cep.cep_generator` | `arandu.qa.cep.generator` |
| `arandu.core.cep.reasoning` | `arandu.qa.cep.reasoning` |
| `arandu.core.cep.validator` | `arandu.qa.cep.validator` |
| `arandu.core.judge` | `arandu.qa.judge` |
| `arandu.core.judge.criterion` | `arandu.qa.judge.criterion` |
| `arandu.core.judge.pipeline` | `arandu.qa.judge.pipeline` |
| `arandu.core.judge.registry` | `arandu.qa.judge.registry` |
| `arandu.core.qa_batch` | `arandu.qa.batch` |

Also update the `__init__.py` files for `qa/cep/` and `qa/judge/` to use new internal paths.

**Step 3:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 4:** Commit.

```bash
git add -A
git commit -m "refactor: move QA/CEP/Judge pipeline to arandu.qa"
```

---

### Task 8: Move `kg/` domain

**Files:**
- Move: `src/arandu/core/kg/batch.py` → `src/arandu/kg/batch.py`
- Move: `src/arandu/core/kg/atlas_backend.py` → `src/arandu/kg/atlas_backend.py`
- Move: `src/arandu/core/kg/factory.py` → `src/arandu/kg/factory.py`
- Move: `src/arandu/core/kg/protocol.py` → `src/arandu/kg/protocol.py`
- Move: `src/arandu/core/kg/__init__.py` → update `src/arandu/kg/__init__.py`
- Note: `src/arandu/kg/schemas.py` was already created in Task 3
- Note: `src/arandu/core/kg/schemas.py` (KGConstructionResult) — merge into `src/arandu/kg/schemas.py`
- Move: `tests/core/kg/` → `tests/kg/`

**Step 1:** Move files.

```bash
git mv src/arandu/core/kg/batch.py src/arandu/kg/batch.py
git mv src/arandu/core/kg/atlas_backend.py src/arandu/kg/atlas_backend.py
git mv src/arandu/core/kg/factory.py src/arandu/kg/factory.py
git mv src/arandu/core/kg/protocol.py src/arandu/kg/protocol.py
```

Merge `src/arandu/core/kg/schemas.py` (which contains `KGConstructionResult`) into `src/arandu/kg/schemas.py` (which already has `KGMetadata` etc. from Task 3).

```bash
git mv tests/core/kg/* tests/kg/
```

Update `src/arandu/kg/__init__.py` with the re-exports from the old `core/kg/__init__.py`.

**Step 2:** Update imports.

| Old path | New path |
|----------|----------|
| `arandu.core.kg` | `arandu.kg` |
| `arandu.core.kg.batch` | `arandu.kg.batch` |
| `arandu.core.kg.atlas_backend` | `arandu.kg.atlas_backend` |
| `arandu.core.kg.factory` | `arandu.kg.factory` |
| `arandu.core.kg.protocol` | `arandu.kg.protocol` |
| `arandu.core.kg.schemas` | `arandu.kg.schemas` |

**Step 3:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run pytest --tb=short -q
```

**Step 4:** Commit.

```bash
git add -A
git commit -m "refactor: move KG pipeline to arandu.kg"
```

---

### Task 9: Move `report/` domain + split `charts.py`

**Files:**
- Move: `src/arandu/core/report/*.py` → `src/arandu/report/`
- Move: `src/arandu/core/report/templates/` → `src/arandu/report/templates/`
- Split: `src/arandu/core/report/charts.py` → `src/arandu/report/charts/` (multiple files)
- Rename: `src/arandu/core/report/api_schemas.py` → `src/arandu/report/schemas.py`
- Move: `tests/core/report/` → `tests/report/`

**Step 1:** Move non-chart files.

```bash
git mv src/arandu/core/report/api.py src/arandu/report/api.py
git mv src/arandu/core/report/api_schemas.py src/arandu/report/schemas.py
git mv src/arandu/core/report/collector.py src/arandu/report/collector.py
git mv src/arandu/core/report/dataset.py src/arandu/report/dataset.py
git mv src/arandu/core/report/exporter.py src/arandu/report/exporter.py
git mv src/arandu/core/report/generator.py src/arandu/report/generator.py
git mv src/arandu/core/report/service.py src/arandu/report/service.py
git mv src/arandu/core/report/style.py src/arandu/report/charts/style.py
cp -r src/arandu/core/report/templates src/arandu/report/templates
git add src/arandu/report/templates
git rm -r src/arandu/core/report/templates
```

**Step 2:** Split `charts.py` (996 lines) into themed modules.

Read `src/arandu/core/report/charts.py` and distribute functions:

`src/arandu/report/charts/quality.py`:
- `create_transcription_quality_chart()` (lines 210–280)
- `create_quality_radar_chart()` (lines 390–445)
- `create_confidence_distribution_chart()` (lines 178–209)

`src/arandu/report/charts/distribution.py`:
- `create_bloom_distribution_chart()` (lines 92–132)
- `create_participant_breakdown_chart()` (lines 555–610)
- `create_location_treemap()` (lines 611–660)

`src/arandu/report/charts/validation.py`:
- `create_validation_scores_chart()` (lines 133–177)
- `create_bloom_validation_heatmap()` (lines 661–735)
- `create_correlation_heatmap()` (lines 323–389)

`src/arandu/report/charts/timeline.py`:
- `create_pipeline_overview_chart()` (lines 30–91)
- `create_run_timeline_chart()` (lines 510–554)
- `create_funnel_chart()` (lines 878–947)

`src/arandu/report/charts/comparison.py`:
- `create_cross_run_comparison()` (lines 736–802)
- `create_parallel_coordinates_chart()` (lines 446–509)
- `create_location_quality_chart()` (lines 803–846)

`src/arandu/report/charts/multihop.py`:
- `create_multihop_chart()` (lines 281–322)

Private helpers stay with their primary consumer or move to the module that uses them:
- `_add_threshold_line()` (lines 847–877) → `validation.py` (used by validation charts)
- `_pearson_r()` (lines 948–970) → `validation.py` (used by correlation heatmap)
- `_empty_figure()` (lines 971–996) → `style.py` or each module imports it

`src/arandu/report/charts/__init__.py` — re-export all chart functions:
```python
from arandu.report.charts.comparison import (
    create_cross_run_comparison,
    create_location_quality_chart,
    create_parallel_coordinates_chart,
)
from arandu.report.charts.distribution import (
    create_bloom_distribution_chart,
    create_location_treemap,
    create_participant_breakdown_chart,
)
from arandu.report.charts.multihop import create_multihop_chart
from arandu.report.charts.quality import (
    create_confidence_distribution_chart,
    create_quality_radar_chart,
    create_transcription_quality_chart,
)
from arandu.report.charts.timeline import (
    create_funnel_chart,
    create_pipeline_overview_chart,
    create_run_timeline_chart,
)
from arandu.report.charts.validation import (
    create_bloom_validation_heatmap,
    create_correlation_heatmap,
    create_validation_scores_chart,
)
```

**Step 3:** Delete old `src/arandu/core/report/charts.py`.

**Step 4:** Move test files.

```bash
git mv tests/core/report/test_api.py tests/report/test_api.py
git mv tests/core/report/test_service.py tests/report/test_service.py
git mv tests/core/report/test_dataset.py tests/report/test_dataset.py
git mv tests/core/report/test_collector.py tests/report/test_collector.py
git mv tests/core/report/test_exporter.py tests/report/test_exporter.py
git mv tests/core/report/test_generator.py tests/report/test_generator.py
git mv tests/core/report/test_style.py tests/report/charts/test_style.py
git mv tests/core/report/test_charts.py tests/report/charts/test_charts.py
```

Note: `test_charts.py` can be split into per-module test files later as a follow-up. For now, keep it as one file with updated imports.

**Step 5:** Update ALL imports.

| Old path | New path |
|----------|----------|
| `arandu.core.report` | `arandu.report` |
| `arandu.core.report.api` | `arandu.report.api` |
| `arandu.core.report.api_schemas` | `arandu.report.schemas` |
| `arandu.core.report.charts` | `arandu.report.charts` |
| `arandu.core.report.collector` | `arandu.report.collector` |
| `arandu.core.report.dataset` | `arandu.report.dataset` |
| `arandu.core.report.exporter` | `arandu.report.exporter` |
| `arandu.core.report.generator` | `arandu.report.generator` |
| `arandu.core.report.service` | `arandu.report.service` |
| `arandu.core.report.style` | `arandu.report.charts.style` |

**Step 6:** Update `src/arandu/report/__init__.py` with re-exports (same symbols as the old `core/report/__init__.py`).

**Step 7:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 8:** Commit.

```bash
git add -A
git commit -m "refactor: move report domain to arandu.report and split charts"
```

---

### Task 10: Split `main.py` into `cli/`

This is the largest single-file split: 2,116 lines → 6 files.

**Files:**
- Create: `src/arandu/cli/app.py` — Typer app, callbacks, `info()`, `main()` callback
- Create: `src/arandu/cli/transcribe.py` — `transcribe`, `drive_transcribe`, `batch_transcribe`, `validate_transcriptions`
- Create: `src/arandu/cli/qa.py` — `generate_cep_qa`
- Create: `src/arandu/cli/kg.py` — `build_kg`
- Create: `src/arandu/cli/report.py` — `report`, `serve_report`
- Create: `src/arandu/cli/manage.py` — `list_runs`, `run_info`, `replicate`, `refresh_auth`, `rebuild_index`, `enrich_metadata`
- Delete: `src/arandu/main.py`
- Modify: `pyproject.toml` — update entry point
- Move: `tests/test_main.py` → `tests/cli/test_app.py` (or split into domain test files)

**Step 1:** Create `src/arandu/cli/app.py`.

This file creates the Typer app and registers commands from submodules:

```python
"""Arandu CLI application."""

from __future__ import annotations

import typer

from arandu import __version__
from arandu.utils.logger import setup_logging

app = typer.Typer(
    name="arandu",
    help="Composable pipelines for ethnographic knowledge elicitation.",
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        import typer
        typer.echo(f"arandu {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=version_callback, is_eager=True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    setup_logging(verbose=verbose)


# Register commands from submodules
from arandu.cli.transcribe import (  # noqa: E402
    transcribe, drive_transcribe, batch_transcribe, validate_transcriptions,
)
from arandu.cli.qa import generate_cep_qa  # noqa: E402
from arandu.cli.kg import build_kg  # noqa: E402
from arandu.cli.report import report, serve_report  # noqa: E402
from arandu.cli.manage import (  # noqa: E402
    list_runs, run_info, replicate, refresh_auth, rebuild_index, enrich_metadata, info,
)
```

Note: The exact registration pattern depends on how Typer commands are currently decorated. If they use `@app.command()`, they need the same `app` instance. Import the commands after creating `app` and use `app.command()` as a decorator in each submodule, OR register them here.

**Alternative pattern** (cleaner): Each submodule defines functions, and `app.py` registers them:

```python
app.command(name="transcribe")(transcribe)
app.command(name="drive-transcribe")(drive_transcribe)
# etc.
```

Or use Typer's sub-app pattern if command grouping is desired.

**Step 2:** Create each CLI submodule by extracting the corresponding command functions from `main.py`, along with their specific imports and any helper functions they use.

Shared helpers (`_ensure_float`, `_create_segments_from_result`, `_safe_int_conversion`) that are used by multiple commands go into `src/arandu/cli/_helpers.py`.

Module-level config instances (`_config = TranscriberConfig()`, `_results_config = ResultsConfig()`) should be created lazily or passed as defaults in each command that needs them.

**Step 3:** Update `pyproject.toml` entry point:

```toml
[project.scripts]
arandu = "arandu.cli.app:app"
```

**Step 4:** Update `pyproject.toml` Ruff per-file-ignores:

```toml
[tool.ruff.lint.per-file-ignores]
"src/arandu/cli/*.py" = ["ANN001", "ANN201"]
"src/arandu/report/api.py" = ["B008", "TC001", "TC003"]
```

**Step 5:** Move and update tests.

```bash
git mv tests/test_main.py tests/cli/test_app.py
```

Update test imports to use new CLI module paths.

**Step 6:** Run tests and lint.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 7:** Commit.

```bash
git add -A
git commit -m "refactor: split main.py into cli/ submodules"
```

---

### Task 11: Clean up — remove empty `core/` directory and update docs

**Files:**
- Delete: `src/arandu/core/` (should be empty after all moves)
- Delete: `tests/core/` (should be empty after all moves)
- Modify: `AGENTS.md` — update file path references
- Modify: `pyproject.toml` — verify all paths are correct
- Modify: `docs/plans/2026-03-04-codebase-reorganization-design.md` — mark as completed

**Step 1:** Verify `core/` directories are empty, then remove.

```bash
# Check nothing remains
find src/arandu/core/ -name "*.py" ! -name "__init__.py"
find tests/core/ -name "*.py" ! -name "__init__.py"
# If empty, remove
rm -rf src/arandu/core/
rm -rf tests/core/
```

**Step 2:** Run the full test suite one final time.

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run pytest --tb=short -q
```

**Step 3:** Update `AGENTS.md` references if any file paths are mentioned.

**Step 4:** Update memory file with new project structure.

**Step 5:** Commit.

```bash
git add -A
git commit -m "refactor: remove empty core/ directory and update documentation"
```

---

### Task 12: Final verification

**Step 1:** Run full lint and test suite.

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest -v
```

**Step 2:** Verify the CLI entry point works.

```bash
uv run arandu --version
uv run arandu --help
```

**Step 3:** Verify import structure is clean — no remaining `arandu.core` imports.

```bash
grep -rn "arandu\.core" src/ tests/ | grep -v __pycache__
```

Expected: No matches.

**Step 4:** Verify no circular imports.

```bash
python -c "import arandu.cli.app"
```

Expected: No errors.

**Step 5:** Commit any final fixes.

```bash
git add -A
git commit -m "chore: final verification of codebase reorganization"
```

---

## Summary of Commits

| # | Commit message | Scope |
|---|---------------|-------|
| 1 | `chore: create package scaffolding for codebase reorganization` | Empty directories |
| 2 | `refactor: move shared infrastructure to arandu.shared` | 6 modules + tests |
| 3 | `refactor: split schemas.py into shared/, qa/, and kg/ domain schemas` | 1 file → 3 files |
| 4 | `refactor: split config.py into domain-specific config modules` | 1 file → 4 files |
| 5 | `refactor: move transcription pipeline to arandu.transcription` | 4 modules + tests |
| 6 | `refactor: move metadata domain to arandu.metadata` | 3 modules + tests |
| 7 | `refactor: move QA/CEP/Judge pipeline to arandu.qa` | 8 modules + tests |
| 8 | `refactor: move KG pipeline to arandu.kg` | 5 modules + tests |
| 9 | `refactor: move report domain to arandu.report and split charts` | 10 modules + tests |
| 10 | `refactor: split main.py into cli/ submodules` | 1 file → 6 files |
| 11 | `refactor: remove empty core/ directory and update documentation` | Cleanup |
| 12 | `chore: final verification of codebase reorganization` | Verification |
