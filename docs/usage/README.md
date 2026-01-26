# G-Transcriber Extended Pipelines - Usage Guide

This guide covers the extended pipelines that build upon transcription results.

## Pipelines Overview

G-Transcriber provides two independent pipelines for post-transcription processing:

| Pipeline | Purpose | Output |
|----------|---------|--------|
| **QA Pipeline** | Generate synthetic QA pairs for retrieval evaluation | `qa_dataset/` |
| **KG Pipeline** | Build knowledge graphs for semantic analysis | `knowledge_graphs/` |

Each pipeline can be run independently and has its own evaluation metrics.

---

## QA Pipeline

Generate synthetic question-answer pairs from transcriptions for training and evaluating retrieval systems.

### Components

1. **QA Generation** - Generate QA pairs using LLMs
2. **QA Evaluation** - Measure QA quality (Exact Match, F1, BLEU)

### Quick Start

```bash
# Generate QA pairs
docker compose --profile qa up

# Evaluate QA quality
GTRANSCRIBER_EVALUATION_METRICS=qa docker compose --profile evaluate up
```

### Documentation

- [QA Generation Guide](QA_GENERATION.md) - Detailed QA generation documentation
- [Evaluation Guide](EVALUATION.md#qa-metrics) - QA evaluation metrics

### Output

```
qa_dataset/
├── qa_<gdrive_id_1>.json
├── qa_<gdrive_id_2>.json
└── ...
```

---

## KG Pipeline

Build knowledge graphs from transcriptions using AutoSchemaKG for entity and relation extraction.

### Components

1. **KG Construction** - Extract entities and relations, build graphs
2. **Graph Evaluation** - Measure graph quality (entity coverage, connectivity, semantic coherence)

### Quick Start

```bash
# Build knowledge graphs
docker compose --profile kg up

# Evaluate graph quality
GTRANSCRIBER_EVALUATION_METRICS=entity,relation,semantic docker compose --profile evaluate up
```

### Documentation

- [KG Construction Guide](KG_CONSTRUCTION.md) - Detailed KG construction documentation
- [Evaluation Guide](EVALUATION.md#entity-metrics) - Graph evaluation metrics

### Output

```
knowledge_graphs/
├── corpus_graph.graphml           # Merged corpus graph
├── corpus_graph_metadata.json     # Provenance metadata
└── individual/                    # Per-document graphs (optional)
```

---

## Prerequisites

### Required Data

Both pipelines require transcription results from the transcription pipeline:

```
results/
├── <gdrive_id_1>.json    # EnrichedRecord from transcription
├── <gdrive_id_2>.json
└── ...
```

### Software Requirements

- Docker with Compose v2
- (Optional) NVIDIA GPU with drivers for Ollama acceleration
- (Optional) SLURM for HPC cluster execution

### LLM Provider

Both QA and KG pipelines require an LLM provider:

| Provider | Setup | Cost |
|----------|-------|------|
| **Ollama** (default) | Docker sidecar included | Free (local) |
| **OpenAI** | Set `OPENAI_API_KEY` | Pay-per-use |
| **Custom** | Set `GTRANSCRIBER_LLM_BASE_URL` | Varies |

---

## Configuration

### Environment Variables

```bash
# QA Pipeline
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=10

# KG Pipeline
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_LANGUAGE=pt

# Evaluation
GTRANSCRIBER_EVALUATION_METRICS=qa,entity,relation,semantic

# API Keys (for OpenAI provider)
OPENAI_API_KEY=sk-...
```

See [Configuration Reference](CONFIGURATION.md) for complete list.

---

## Docker Compose Profiles

| Profile | Services | Pipeline |
|---------|----------|----------|
| `qa` | ollama, gtranscriber-qa | QA Pipeline |
| `kg` | ollama, gtranscriber-kg | KG Pipeline |
| `evaluate` | gtranscriber-eval | Both (configurable) |
| `ollama` | ollama | Standalone LLM service |

---

## SLURM Execution

```bash
# QA Pipeline
sbatch scripts/slurm/run_qa_generation.slurm

# KG Pipeline
sbatch scripts/slurm/run_kg_construction.slurm

# Evaluation
sbatch scripts/slurm/run_evaluation.slurm
```

---

## Pipeline Flow

```
┌─────────────────┐
│  Transcription  │
│     Results     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────┐
│   QA   │  │   KG   │
│Pipeline│  │Pipeline│
└────┬───┘  └────┬───┘
     │           │
     ▼           ▼
┌────────┐  ┌────────────┐
│   QA   │  │  Knowledge │
│Dataset │  │   Graphs   │
└────┬───┘  └────┬───────┘
     │           │
     ▼           ▼
┌────────┐  ┌────────────┐
│   QA   │  │   Graph    │
│  Eval  │  │    Eval    │
└────────┘  └────────────┘
```

---

## Detailed Guides

| Guide | Description |
|-------|-------------|
| [QA Generation](QA_GENERATION.md) | Generate synthetic QA datasets |
| [KG Construction](KG_CONSTRUCTION.md) | Build knowledge graphs with AutoSchemaKG |
| [Evaluation](EVALUATION.md) | Quality metrics for both pipelines |
| [Configuration](CONFIGURATION.md) | Environment variables reference |

---

**Document Version**: 1.1
**Last Updated**: 2026-01-26
