# Implementation Phase Status

Current project implementation status and completed work.

## Status Overview

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Transcription Pipeline |
| **Phase 2** | ✅ Complete | CEP QA Generation Pipeline |
| **Phase 3** | ⚠️ Code Complete | Knowledge Graph Construction (no successful end-to-end run) |
| **Phase 4** | 🔲 Not Started | Multi-Stage Evaluation — judge refactor + RAG ([#79](https://github.com/FredDsR/etno-kgc-preprocessing/issues/79), [#80](https://github.com/FredDsR/etno-kgc-preprocessing/issues/80)) |
| **Phase 5** | 🔲 Not Started | Writing, Human Evaluation & Publication |

---

## Phase 1: Transcription Pipeline ✅

**Completed**: 2025-12

### What Was Implemented

1. **Batch Transcription Command** (`batch-transcribe`)
   - Parallel processing with configurable workers
   - Checkpoint and resume capability
   - Google Drive integration

2. **Components**
   - `transcription/engine.py` — Whisper ASR wrapper with quantization and device auto-detection
   - `transcription/batch.py` — Batch processing orchestrator with ProcessPoolExecutor
   - `transcription/media.py` — Media duration extraction with FFprobe
   - `transcription/validator.py` — Heuristic quality validation (script match, repetition, segment, content density)
   - `shared/checkpoint.py` — Checkpoint system for resumption
   - `shared/drive.py` — Google Drive integration

3. **Features**
   - Multiple model support (any Hugging Face Whisper model)
   - 8-bit quantization for reduced VRAM
   - GPU/CPU/ROCm support
   - Structured JSON output with metadata

### Run History

| Run | Files | Status |
|-----|-------|--------|
| `20260124_084730_087491_slurm_tupi` | 353/354 (99.7%) | Complete |

### Usage

```bash
arandu batch-transcribe input/catalog.csv --workers 4 --quantize
```

---

## Phase 2: CEP QA Generation Pipeline ✅

**Completed**: 2026-01-29 (basic), 2026-02-11 (CEP-only refactor)

### What Was Implemented

1. **CEP QA Generator** (`qa/cep/`)
   - Bloom's Taxonomy cognitive scaffolding (remember, understand, analyze, evaluate)
   - Configurable Bloom level distribution
   - Scaffolding context (lower-level QA pairs inform higher levels)
   - Reasoning trace generation for multi-hop questions
   - Context chunking for long transcriptions

2. **Judge Pipeline** (`qa/judge/` — will move to `shared/judge/` in Phase B)
   - Protocol-based `JudgeCriterion` interface
   - Composable `JudgePipeline` with weighted criteria (to be renamed `JudgeStep`; see [ROADMAP.md](ROADMAP.md) Phase B)
   - `JudgeRegistry` for criterion factory and caching
   - Criteria: faithfulness (30%), Bloom calibration (25%), informativeness (25%), self-containedness (20%)

3. **Batch Orchestrator** (`qa/batch.py`)
   - Parallel processing with ProcessPoolExecutor
   - Global worker pattern for connection pooling
   - Checkpoint integration

4. **CLI Command** (`generate-cep-qa`)
   - LLM provider selection (Ollama, OpenAI, custom)
   - `--validate` flag will be removed — judge becomes a separate command (see [ROADMAP.md](ROADMAP.md) Phase B)

### Run History

| Run | Records | Status |
|-----|---------|--------|
| `test-cep-01` | 241/309 (78%) | Complete |

### Known Issues

- Weighted average threshold (0.6) masks per-criterion failures — individual thresholds needed
- Self-containedness prompt misses place references without names
- 3 bad transcription records with repetition loops (see TO-DO.md)

### Usage

```bash
arandu generate-cep-qa results/ -o qa_dataset/ --workers 4 --questions 12
```

---

## Phase 3: Knowledge Graph Construction ⚠️

**Code completed**: 2026-02 | **No successful end-to-end run** | Tracked in [#78](https://github.com/FredDsR/etno-kgc-preprocessing/issues/78)

### What Was Implemented

1. **Atlas Backend** (`kg/atlas_backend.py`)
   - AutoSchemaKG integration via atlas-rag package
   - Protocol-based `KGConstructor` interface with factory pattern
   - Portuguese extraction prompts
   - Triple extraction → CSV conversion → concept generation → GraphML export

2. **Batch Orchestrator** (`kg/batch.py`)
   - Parallel processing with resume support
   - Per-document + corpus-level graph merging

3. **CLI Command** (`build-kg`)

### Run History

| Run | Status | Issue |
|-----|--------|-------|
| `test-kg-01` | Failed (0/309) | JSON parse error on startup |
| `test-kg-02` | Stuck mid-pipeline | Triple extraction complete (309 docs). Concept generation reached 6,256/13,383 nodes before SLURM timeout (24h). No GraphML output. |

### Blocking Issues

- **#77**: Concept generation not resumable — SLURM kill loses all progress. Detailed design exists for resume wrapper.
- **Language bug**: `language='pt'` not passed to `generate_concept_csv_temp()`, so concept generation runs with English prompts on Portuguese text. `test-kg-02` concept output shows English conceptualizations.

### What's Needed

1. Implement #77 (resumable concept generation + language fix)
2. Complete a successful run producing a GraphML file
3. Manually inspect graph quality (Portuguese entities/relations)

---

## Phase 4: Multi-Stage Evaluation 🔲

**Status**: Not started — design agreed (2026-03-23), see [ROADMAP.md](ROADMAP.md)

### Redesigned Architecture

The evaluation pipeline has been redesigned from the original plan (which used EM/F1/BLEU metrics). The new design uses the **same composable judge pipeline** across all domains.

**Core idea**: evaluation = retrieve + judge. No traditional QA metrics — the judge module scores retriever answers using the same criteria architecture as QA validation.

### Planned Components

1. **Judge module refactor** — move from `qa/judge/` to `shared/judge/`
   - `JudgeCriterion` (protocol — heuristic or LLM-based)
   - `JudgeStep` (runs N criteria with individual thresholds, renamed from current `JudgePipeline`)
   - `JudgePipeline` (multi-stage: chains steps with filtering between them)

2. **Retriever module** — protocol-based with pluggable implementations
   - BM25 baseline (sparse retrieval over transcriptions)
   - GraphRAG (retrieval over constructed KG)

3. **CLI commands** (each atomic, no model co-loading):
   - `arandu judge transcription` — heuristic filter → LLM judge
   - `arandu judge qa` — multi-criterion LLM judge → human-comparable judge
   - `arandu retrieve` — run retrievers against QA questions
   - `arandu judge answers` — judge retriever answer quality

### Key Design Decisions

- **Judge is cross-domain** — same module validates transcriptions, QA pairs, and retriever answers
- **Heuristics are criteria** — implement `JudgeCriterion` protocol, no special treatment
- **Individual thresholds** — per-criterion minimums replace weighted average
- **Separate commands** — generator generates, judge judges (remove `--validate` flag from `generate-cep-qa`)
- **No EM/F1/BLEU** — judge-based evaluation throughout

---

## Phase 5: Writing, Human Evaluation & Publication 🔲

**Status**: Not started — see [ROADMAP.md](ROADMAP.md) Phase D for detailed chapter mapping

### Dissertation (Monography)

| Chapter | Source Material | Can Start |
|---------|---------------|-----------|
| 1. Introdução | SLR evidence gap, `related-works.md` §10-11, CARE Principles | Now |
| 2. Referencial Teórico | `related-works.md` (45+ refs, 11 sections) → consolidate into narrative | Now |
| 3. Metodologia | `methodology.md` (62K chars, near-complete draft) — §6 needs rewrite for judge-based eval | Now (except eval section) |
| 4. Resultados e Discussão | Transcription + CEP data exist; KG blocked on Phase 3 run; RAG blocked on Phase 4 | Partially |
| 5. Conclusão | — | Last |

### Human Evaluation

Design annotation protocol → recruit specialists → evaluation sessions → Cohen's kappa (LLM vs human). Blocked on Phase 4 (human-comparable judge criterion).

### Articles

- Consolidate 2025/2 article with new experiments
- Bloom taxonomy for cognitive-calibrated QA generation (standalone topic)
- Conference submissions

---

## Current Architecture

```
src/arandu/
├── cli/              – app.py, transcribe.py, qa.py, kg.py, report.py, manage.py
├── transcription/    – engine, batch, validator, media, config, schemas
├── qa/               – cep/, judge/, batch, config, schemas
├── kg/               – atlas_backend, batch, config, schemas, prompts/
├── report/           – api, service, collector, dataset, generator, exporter, charts/
├── metadata/         – enricher, schemas
├── shared/           – config, schemas, llm_client, checkpoint, hardware, io, drive
└── utils/            – console, logger
```

Entry point: `arandu.cli.app:app` (pyproject.toml)

---

## Open Issues (non-frontend)

| # | Title | Type | Priority |
|---|-------|------|----------|
| [#78](https://github.com/FredDsR/etno-kgc-preprocessing/issues/78) | Phase A: Unblock KG | enhancement | Critical |
| [#79](https://github.com/FredDsR/etno-kgc-preprocessing/issues/79) | Phase B: Judge refactor | enhancement | High |
| [#80](https://github.com/FredDsR/etno-kgc-preprocessing/issues/80) | Phase C: RAG evaluation | enhancement | High |
| [#77](https://github.com/FredDsR/etno-kgc-preprocessing/issues/77) | Resumable concept generation + language bug fix | enhancement | Critical (blocks #78) |
| [#35](https://github.com/FredDsR/etno-kgc-preprocessing/issues/35) | Extract `generate_structured()` to LLMClient | refactor | Part of #79 |
| ~~[#75](https://github.com/FredDsR/etno-kgc-preprocessing/issues/75)~~ | ~~Concept gen resume~~ | — | Closed (superseded by #77) |

---

**Last Updated**: 2026-03-23
