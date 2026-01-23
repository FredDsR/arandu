# File Structure Documentation

Complete overview of files to be created and modified in the Knowledge Graph Construction Pipeline implementation.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Existing Files to Modify](#existing-files-to-modify)
3. [New Files to Create](#new-files-to-create)
4. [Generated Files and Directories](#generated-files-and-directories)

---

## Directory Structure

```
etno-kgc-preprocessing/
├── .env                              # Environment configuration (git-ignored)
├── .env.example                      # [MODIFY] Configuration template
├── .gitignore                        # [MODIFY] Add new directories
├── README.md                         # [MODIFY] Update with new features
├── pyproject.toml                    # [MODIFY] Add new dependencies
├── docker-compose.yml                # [MODIFY] Add new service profiles
├── Dockerfile                        # (no changes needed)
├── Dockerfile.rocm                   # (no changes needed)
│
├── src/
│   └── gtranscriber/
│       ├── __init__.py              # (no changes needed)
│       ├── main.py                  # [MODIFY] Add new CLI commands
│       ├── config.py                # [MODIFY] Add new settings
│       ├── schemas.py               # [MODIFY] Add new data models
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── engine.py           # (existing, no changes)
│       │   ├── batch.py            # (existing, no changes)
│       │   ├── drive.py            # (existing, no changes)
│       │   ├── checkpoint.py       # (existing, no changes)
│       │   ├── hardware.py         # (existing, no changes)
│       │   ├── media.py            # (existing, no changes)
│       │   ├── io.py               # (existing, no changes)
│       │   │
│       │   ├── llm_client.py       # [NEW] Unified LLM client
│       │   ├── qa_generator.py     # [NEW] QA generation engine
│       │   ├── qa_batch.py         # [NEW] QA batch processing
│       │   ├── kg_builder.py       # [NEW] AutoSchemaKG wrapper
│       │   ├── kg_batch.py         # [NEW] KG batch processing
│       │   ├── metrics.py          # [NEW] Evaluation metrics
│       │   └── evaluator.py        # [NEW] Evaluation orchestration
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logger.py           # (existing, no changes)
│           ├── ui.py               # (existing, no changes)
│           └── console.py          # (existing, no changes)
│
├── scripts/
│   └── slurm/
│       ├── job_common.sh           # (existing, reused)
│       ├── draco.slurm             # (existing, no changes)
│       ├── grace.slurm             # (existing, no changes)
│       ├── blaise.slurm            # (existing, no changes)
│       ├── turing.slurm            # (existing, no changes)
│       ├── sirius.slurm            # (existing, no changes)
│       ├── tupi.slurm              # (existing, no changes)
│       │
│       ├── run_qa_generation.slurm       # [NEW] QA generation job
│       ├── run_kg_construction.slurm     # [NEW] KG construction job
│       └── run_evaluation.slurm          # [NEW] Evaluation job
│
├── tests/                           # [NEW] Test directory
│   ├── __init__.py                 # [NEW]
│   ├── test_llm_client.py          # [NEW] LLM client tests
│   ├── test_qa_generator.py        # [NEW] QA generator tests
│   ├── test_kg_builder.py          # [NEW] KG builder tests
│   ├── test_metrics.py             # [NEW] Metrics tests
│   ├── test_evaluator.py           # [NEW] Evaluator tests
│   │
│   ├── fixtures/                   # [NEW] Test fixtures
│   │   ├── sample_transcription.json
│   │   ├── sample_qa_record.json
│   │   └── sample_graph.graphml
│   │
│   └── integration/                # [NEW] Integration tests
│       ├── test_qa_pipeline.sh
│       ├── test_kg_pipeline.sh
│       └── test_evaluation_pipeline.sh
│
├── docs/                            # [NEW] Documentation directory
│   ├── IMPLEMENTATION_PLAN.md      # [CREATED] Main implementation plan
│   │
│   └── implementation/              # [NEW] Detailed documentation
│       ├── DATA_SCHEMAS.md         # [CREATED] Schema specifications
│       ├── CONFIGURATION.md        # [CREATED] Configuration reference
│       ├── CLI_REFERENCE.md        # [CREATED] CLI commands reference
│       ├── API_DOCUMENTATION.md    # [NEW] API reference
│       ├── FILE_STRUCTURE.md       # [THIS FILE]
│       ├── DEPENDENCIES.md         # [NEW] Dependency documentation
│       ├── TROUBLESHOOTING.md      # [NEW] Common issues
│       └── PERFORMANCE_TUNING.md   # [NEW] Optimization guide
│
├── input/                           # (existing)
│   ├── catalog.csv
│   └── minimal-catalog.csv
│
├── results/                         # (existing - transcription outputs)
├── qa_dataset/                      # [GENERATED] QA generation outputs
├── knowledge_graphs/                # [GENERATED] KG construction outputs
├── evaluation/                      # [GENERATED] Evaluation reports
├── cache/                           # (existing - HuggingFace cache)
└── credentials/                     # (existing - Google OAuth)
```

---

## Existing Files to Modify

### 1. `src/gtranscriber/config.py`

**Current**: ~100 lines
**After**: ~200 lines

**Changes**:
```python
# Add QA Generation settings
qa_provider: str = Field(default="ollama", ...)
qa_model_id: str = Field(default="llama3.1:8b", ...)
qa_ollama_url: str = Field(default="http://localhost:11434", ...)
openai_api_key: str | None = Field(default=None, ...)
anthropic_api_key: str | None = Field(default=None, ...)
questions_per_document: int = Field(default=10, ...)
qa_strategies: list[str] = Field(default=["factual", "conceptual"], ...)
qa_temperature: float = Field(default=0.7, ...)
qa_output_dir: Path = Field(default=Path("qa_dataset"), ...)

# Add KG Construction settings
kg_provider: str = Field(default="ollama", ...)
kg_model_id: str = Field(default="llama3.1:8b", ...)
kg_merge_graphs: bool = Field(default=True, ...)
kg_output_format: str = Field(default="graphml", ...)  # graphml is default (NetworkX-compatible)
kg_schema_mode: str = Field(default="dynamic", ...)
kg_output_dir: Path = Field(default=Path("knowledge_graphs"), ...)

# Add Evaluation settings
evaluation_metrics: list[str] = Field(default=["qa", "entity", "relation", "semantic"], ...)
embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", ...)
evaluation_output_dir: Path = Field(default=Path("evaluation"), ...)
```

**Location**: Lines 23-24 (after existing settings)

### 2. `src/gtranscriber/schemas.py`

**Current**: ~90 lines
**After**: ~400 lines

**Changes**:
```python
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

# Add QA schemas (Pydantic models with validation)
class QAPair(BaseModel): ...
class QARecord(BaseModel): ...

# Add KG metadata (Pydantic model for JSON serialization)
class KGMetadata(BaseModel): ...

# Add Evaluation schemas (Pydantic models with computed fields)
class GraphConnectivity(BaseModel): ...
class EntityCoverageResult(BaseModel): ...
class RelationMetricsResult(BaseModel): ...
class SemanticQualityResult(BaseModel): ...
class EvaluationReport(BaseModel): ...
```

**Note**: All schemas use Pydantic v2 models for:
- Built-in validation with `Field()` constraints
- Automatic JSON serialization via `model_dump_json()`
- JSON deserialization via `model_validate_json()`
- Computed properties via `@computed_field`

Knowledge graphs use AutoSchemaKG's native GraphML output with NetworkX.
No custom KGNode/KGEdge/KGRecord classes needed.

**Location**: Lines 60-88 (after existing schemas)

### 3. `src/gtranscriber/main.py`

**Current**: ~690 lines
**After**: ~1000 lines

**Changes**:
```python
# Add import for new modules
from gtranscriber.core.qa_generator import QAGenerator
from gtranscriber.core.qa_batch import run_batch_qa_generation, QABatchConfig
from gtranscriber.core.kg_builder import KGBuilder
from gtranscriber.core.kg_batch import run_batch_kg_construction, KGBatchConfig
from gtranscriber.core.evaluator import KnowledgeEvaluator, EvaluationConfig

# Add new commands
@app.command()
def generate_qa(...): ...

@app.command()
def build_kg(...): ...

@app.command()
def evaluate(...): ...
```

**Location**: After existing commands (around line 690)

### 4. `docker-compose.yml`

**Current**: ~160 lines
**After**: ~280 lines

**Changes**:
```yaml
services:
  # ... existing services ...

  gtranscriber-qa:
    extends:
      service: gtranscriber
    # ... QA service configuration ...
    profiles:
      - qa

  gtranscriber-kg:
    extends:
      service: gtranscriber
    # ... KG service configuration ...
    profiles:
      - kg

  gtranscriber-eval:
    extends:
      service: gtranscriber
    # ... Evaluation service configuration ...
    profiles:
      - evaluate
```

**Location**: After line 160 (end of file)

### 5. `pyproject.toml`

**Current**: Dependencies section
**After**: Extended dependencies

**Changes**:
```toml
[project]
dependencies = [
    # ... existing dependencies ...

    # New dependencies for P2
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "httpx>=0.27.0",
    "atlas-rag>=0.0.5",
    "networkx>=3.1",
    "scikit-learn>=1.3.0",
    "sentence-transformers>=2.2.0",
    "nltk>=3.8.0",
    "sacrebleu>=2.3.0",
]
```

### 6. `.gitignore`

**Changes**:
```gitignore
# ... existing entries ...

# New directories
qa_dataset/
knowledge_graphs/
evaluation/
graphrag_index/

# Test outputs
tests/outputs/

# Checkpoints
*checkpoint.json
```

### 7. `README.md`

**Changes**:
- Add documentation for new commands
- Update feature list
- Add examples for QA generation, KG construction, evaluation
- Update architecture diagram

**Sections to Add**:
```markdown
## New Features (v2.0)

### Synthetic QA Dataset Generation
Generate question-answer pairs from transcriptions...

### Knowledge Graph Construction
Build knowledge graphs using AutoSchemaKG...

### Knowledge Elicitation Evaluation
Measure quality across four dimensions...
```

### 8. `.env.example`

**Changes**:
```bash
# ... existing settings ...

# QA Generation Settings
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10

# API Keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# KG Construction Settings
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_MERGE_GRAPHS=true

# Evaluation Settings
GTRANSCRIBER_EVALUATION_METRICS=qa,entity,relation,semantic
```

---

## New Files to Create

### Core Modules

#### 1. `src/gtranscriber/core/llm_client.py`

**Purpose**: Unified LLM client supporting OpenAI, Anthropic, and Ollama

**Estimated Size**: ~300 lines

**Key Classes**:
- `LLMProvider(Enum)` - Provider types
- `LLMClient` - Main client class
- `OpenAIClient` - OpenAI implementation
- `AnthropicClient` - Anthropic implementation
- `OllamaClient` - Ollama implementation

**Dependencies**: `openai`, `anthropic`, `httpx`, `tenacity`

#### 2. `src/gtranscriber/core/qa_generator.py`

**Purpose**: QA pair generation with multiple strategies

**Estimated Size**: ~400 lines

**Key Classes**:
- `QAStrategy(Enum)` - Strategy types
- `QAGenerator` - Main generator class
- `PromptTemplates` - Prompt templates for each strategy

**Dependencies**: `llm_client`, `schemas`

#### 3. `src/gtranscriber/core/qa_batch.py`

**Purpose**: Batch QA generation with checkpointing

**Estimated Size**: ~250 lines

**Key Classes**:
- `QABatchConfig(BaseModel)` - Configuration (Pydantic model)
- `run_batch_qa_generation()` - Main entry point

**Dependencies**: `qa_generator`, `batch` (for patterns), `checkpoint`

#### 4. `src/gtranscriber/core/kg_builder.py`

**Purpose**: AutoSchemaKG integration wrapper

**Estimated Size**: ~350 lines

**Key Classes**:
- `KGBuilder` - Main builder class wrapping AutoSchemaKG
- `SchemaManager` - Schema handling

**Dependencies**: `atlas_rag`, `networkx`, `schemas`

#### 5. `src/gtranscriber/core/kg_batch.py`

**Purpose**: Batch KG construction

**Estimated Size**: ~300 lines

**Key Classes**:
- `KGBatchConfig(BaseModel)` - Configuration (Pydantic model)
- `run_batch_kg_construction()` - Main entry point

**Dependencies**: `kg_builder`, `checkpoint`

#### 6. `src/gtranscriber/core/metrics.py`

**Purpose**: Evaluation metric implementations

**Estimated Size**: ~500 lines

**Key Classes**:
- `QAMetrics` - QA-based metrics
- `EntityMetrics` - Entity coverage metrics
- `RelationMetrics` - Relation density metrics
- `SemanticQualityMetrics` - Semantic quality metrics

**Dependencies**: `scikit-learn`, `sacrebleu`, `sentence-transformers`, `networkx`

#### 7. `src/gtranscriber/core/evaluator.py`

**Purpose**: Evaluation orchestration

**Estimated Size**: ~350 lines

**Key Classes**:
- `EvaluationConfig(BaseModel)` - Configuration (Pydantic model)
- `KnowledgeEvaluator` - Main evaluator class

**Dependencies**: `metrics`, `schemas`

### SLURM Scripts

#### 1. `scripts/slurm/run_qa_generation.slurm`

**Purpose**: SLURM job script for QA generation

**Estimated Size**: ~80 lines

**Content**:
```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Configuration and execution
```

#### 2. `scripts/slurm/run_kg_construction.slurm`

**Purpose**: SLURM job script for KG construction

**Estimated Size**: ~90 lines

#### 3. `scripts/slurm/run_evaluation.slurm`

**Purpose**: SLURM job script for evaluation

**Estimated Size**: ~70 lines

### Tests

#### Unit Tests

1. `tests/test_llm_client.py` (~200 lines)
2. `tests/test_qa_generator.py` (~250 lines)
3. `tests/test_kg_builder.py` (~200 lines)
4. `tests/test_metrics.py` (~300 lines)
5. `tests/test_evaluator.py` (~150 lines)

#### Integration Tests

1. `tests/integration/test_qa_pipeline.sh` (~50 lines)
2. `tests/integration/test_kg_pipeline.sh` (~60 lines)
3. `tests/integration/test_evaluation_pipeline.sh` (~40 lines)

#### Fixtures

1. `tests/fixtures/sample_transcription.json`
2. `tests/fixtures/sample_qa_record.json`
3. `tests/fixtures/sample_graph.graphml`

### Documentation

1. `docs/implementation/API_DOCUMENTATION.md` (~1000 lines)
2. `docs/implementation/DEPENDENCIES.md` (~300 lines)
3. `docs/implementation/TROUBLESHOOTING.md` (~500 lines)
4. `docs/implementation/PERFORMANCE_TUNING.md` (~400 lines)

---

## Generated Files and Directories

These files and directories are created during runtime:

### QA Dataset Directory

```
qa_dataset/
├── qa_<gdrive_id_1>.json       # QARecord for document 1
├── qa_<gdrive_id_2>.json       # QARecord for document 2
├── ...
└── qa_checkpoint.json          # Checkpoint for resumption
```

**File Naming**: `qa_<source_gdrive_id>.json`

**Format**: JSON (QARecord schema)

### Knowledge Graphs Directory

```
knowledge_graphs/
├── corpus_graph.graphml              # Main graph (NetworkX-compatible)
├── corpus_graph_metadata.json        # Provenance metadata
├── individual/                       # Per-document graphs (optional)
│   ├── <gdrive_id_1>.graphml
│   ├── <gdrive_id_1>_metadata.json
│   └── ...
└── checkpoints/
    └── kg_checkpoint.json            # Processing checkpoint
```

**File Naming**:
- Graph: `<graph_id>.graphml` (AutoSchemaKG native output)
- Metadata: `<graph_id>_metadata.json` (provenance sidecar)

**Format**: GraphML (loaded with `nx.read_graphml()`) + JSON metadata

### Evaluation Directory

```
evaluation/
├── evaluation_report_<timestamp>.json
├── qa_metrics_<timestamp>.json
├── entity_metrics_<timestamp>.json
└── relation_metrics_<timestamp>.json
```

**File Naming**: `<metric_type>_<ISO8601_timestamp>.json`

**Format**: JSON (EvaluationReport schema)

### Cache Directories

```
cache/
├── huggingface/               # (existing) HuggingFace models
├── sentence_transformers/     # Sentence transformer models
└── nltk_data/                # NLTK data
```

### Temporary Directories

```
/tmp/gtranscriber/
├── downloads/                 # Temporary downloaded files
├── processing/                # Processing workspace
└── exports/                   # Temporary exports
```

---

## File Size Estimates

### Source Code

| File | Estimated Lines | Complexity |
|------|----------------|------------|
| `llm_client.py` | 300 | Medium |
| `qa_generator.py` | 400 | High |
| `qa_batch.py` | 250 | Medium |
| `kg_builder.py` | 350 | High |
| `kg_batch.py` | 300 | Medium |
| `metrics.py` | 500 | High |
| `evaluator.py` | 350 | Medium |
| **Total New Code** | **2,450** | |

### Tests

| Category | Estimated Lines |
|----------|----------------|
| Unit Tests | 1,100 |
| Integration Tests | 150 |
| Fixtures | 500 |
| **Total Tests** | **1,750** |

### Documentation

| Document | Estimated Lines |
|----------|----------------|
| Implementation Plan | 1,200 |
| Data Schemas | 800 |
| Configuration | 600 |
| CLI Reference | 700 |
| API Documentation | 1,000 |
| Other Docs | 1,200 |
| **Total Docs** | **5,500** |

### Overall Project

| Category | Lines of Code |
|----------|---------------|
| Existing Code | ~4,000 |
| New Code | ~2,450 |
| Tests | ~1,750 |
| **Total Code** | **~8,200** |

---

## Version Control

### Branches

Recommended branching strategy:

```
main
├── feature/phase1-foundation
├── feature/phase2-qa-generation
├── feature/phase3-kg-construction
├── feature/phase4-evaluation
└── feature/phase5-research
```

### Commits

Suggested commit structure:

```
Phase 1:
- feat: add unified LLM client
- feat: extend configuration with QA/KG settings
- feat: add QA and KG data schemas
- chore: update dependencies
- docs: add configuration reference

Phase 2:
- feat: implement QA generator with strategies
- feat: add QA batch processing
- feat: add generate-qa CLI command
- test: add QA generator tests
- docs: add QA generation examples

... (etc.)
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
