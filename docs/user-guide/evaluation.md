# Evaluation Guide

> **⚠️ STATUS: PLANNED - NOT YET IMPLEMENTED**
>
> Evaluation pipeline for assessing QA and KG quality is planned for a future release. The configuration schemas (`EvaluationConfig`, `EvaluationReport`) exist in `src/arandu/config.py` and `src/arandu/schemas.py`, but there are currently:
> - **No CLI commands** for evaluation (no `evaluate` command)
> - **No pipeline modules** implementing the evaluation logic
> - **No dependencies** added yet (`scikit-learn`, `sentence-transformers`, `nltk`, `sacrebleu` are not in `pyproject.toml`)
>
> This documentation describes the **planned functionality** and serves as a specification for future implementation.

---

Evaluate the quality of QA datasets and knowledge graphs using targeted metrics for each pipeline.

## Overview

The evaluation system provides metrics tailored to each pipeline:

| Pipeline | Metrics | Focus |
|----------|---------|-------|
| **QA Pipeline** | `qa` | Answer quality, retrieval readiness |
| **KG Pipeline** | `entity`, `relation`, `semantic` | Graph quality, knowledge coverage |

You can run evaluations independently for each pipeline or combine them.

## Prerequisites

- Docker with Compose v2
- (Optional) Transcription results in `results/`

**For QA Evaluation:**
- QA dataset in `qa_dataset/` directory

**For KG Evaluation:**
- Knowledge graph in `knowledge_graphs/`

**Note**: Evaluation does not require an LLM provider - it uses local computation only.

---

# QA Pipeline Evaluation

Evaluate the quality of generated question-answer pairs.

## QA Metrics

| Metric | Range | Description | Good Value |
|--------|-------|-------------|------------|
| **Exact Match** | 0.0-1.0 | Percentage of exact answer matches | > 0.6 |
| **F1 Score** | 0.0-1.0 | Token-level F1 between predicted and gold | > 0.7 |
| **BLEU Score** | 0-100 | N-gram overlap score | > 60 |

## Running QA Evaluation

### Using Docker Compose

```bash
# Evaluate QA quality only
ARANDU_EVALUATION_METRICS=qa docker compose --profile evaluate up
```

### Using SLURM

```bash
EVAL_METRICS=qa sbatch scripts/slurm/run_evaluation.slurm
```

## QA Output

```
evaluation/
├── report.json                    # Main evaluation report
└── qa_metrics_<timestamp>.json    # Detailed QA metrics
```

## QA Report Example

```json
{
  "dataset_name": "etno_qa_evaluation",
  "evaluation_timestamp": "2026-01-26T18:30:00Z",
  "total_documents": 187,
  "total_qa_pairs": 2244,
  "qa_exact_match": 0.68,
  "qa_f1_score": 0.79,
  "qa_bleu_score": 72.3,
  "overall_score": 0.74
}
```

## QA Troubleshooting

### No QA Metrics

```bash
# Verify QA dataset exists
ls -la qa_dataset/*.json
```

### Low QA Quality

| Issue | Indicator | Solution |
|-------|-----------|----------|
| Low F1 | F1 < 0.6 | Use better LLM model |
| Poor match | EM < 0.5 | Lower temperature (0.3-0.5) |
| Inconsistent | High variance | Increase questions per document |

---

# KG Pipeline Evaluation

Evaluate the quality of knowledge graphs through entity, relation, and semantic metrics.

## Entity Metrics

| Metric | Range | Description | Good Value |
|--------|-------|-------------|------------|
| **Total Entities** | 0+ | Count of all entity mentions | - |
| **Unique Entities** | 0+ | Count of distinct entities | - |
| **Entity Density** | 0+ | Entities per 100 tokens | 10-20 |
| **Entity Diversity** | 0.0-1.0 | unique/total ratio | 0.3-0.6 |

## Relation Metrics

| Metric | Range | Description | Good Value |
|--------|-------|-------------|------------|
| **Total Relations** | 0+ | Count of all relations | - |
| **Unique Relations** | 0+ | Count of relation types | - |
| **Relation Density** | 0+ | Relations per entity | 1.5-3.0 |
| **Average Degree** | 0+ | Mean node connections | > 2.0 |
| **Connected Components** | 1+ | Number of graph components | 1-5 |
| **Graph Density** | 0.0-1.0 | Edge/possible edge ratio | 0.001-0.01 |

## Semantic Metrics

| Metric | Range | Description | Good Value |
|--------|-------|-------------|------------|
| **Coherence Score** | 0.0-1.0 | Semantic consistency | > 0.7 |
| **Information Density** | 0+ | (Entities+Relations)/text_length | > 0.03 |
| **Knowledge Coverage** | 0.0-1.0 | Entities covered by QA | > 0.5 |

## Running KG Evaluation

### Using Docker Compose

```bash
# Evaluate graph quality only
ARANDU_EVALUATION_METRICS=entity,relation docker compose --profile evaluate up

# Include semantic metrics (requires embedding computation)
ARANDU_EVALUATION_METRICS=entity,relation,semantic docker compose --profile evaluate up
```

### Using SLURM

```bash
EVAL_METRICS=entity,relation,semantic sbatch scripts/slurm/run_evaluation.slurm
```

## KG Output

```
evaluation/
├── report.json                        # Main evaluation report
├── entity_metrics_<timestamp>.json    # Entity coverage details
└── relation_metrics_<timestamp>.json  # Relation and connectivity details
```

## KG Report Example

```json
{
  "dataset_name": "etno_kg_evaluation",
  "evaluation_timestamp": "2026-01-26T18:30:00Z",
  "total_documents": 187,
  "entity_coverage": {
    "total_entities": 4823,
    "unique_entities": 1523,
    "entity_density": 12.4,
    "entity_diversity": 0.316,
    "entity_type_distribution": {
      "PERSON": 342,
      "LOCATION": 189,
      "EVENT": 423,
      "CONCEPT": 569
    }
  },
  "relation_metrics": {
    "total_relations": 3847,
    "unique_relations": 1204,
    "relation_density": 2.53,
    "relation_diversity": 0.313,
    "graph_connectivity": {
      "average_degree": 5.05,
      "connected_components": 3,
      "largest_component_size": 1421,
      "density": 0.0033
    }
  },
  "semantic_quality": {
    "coherence_score": 0.78,
    "information_density": 0.042,
    "knowledge_coverage": 0.64
  },
  "overall_score": 0.71
}
```

## KG Troubleshooting

### No Graph Metrics

```bash
# Verify knowledge graph exists
ls -la knowledge_graphs/corpus_graph.graphml
```

### Warning Signs

| Issue | Indicator | Solution |
|-------|-----------|----------|
| Too few entities | Density < 5 | Use larger model, check prompts |
| Low diversity | Diversity < 0.2 | Reduce repetition in extraction |
| Disconnected graph | Components > 10 | Improve relation extraction |
| Low coherence | Score < 0.5 | Check language settings |

### Memory Issues with Semantic Metrics

```bash
# Use smaller embedding model
export ARANDU_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Or run entity/relation metrics first (fast)
ARANDU_EVALUATION_METRICS=entity,relation docker compose --profile evaluate up
```

---

# Full Evaluation

Run both QA and KG evaluation together.

## Quick Start

```bash
# All metrics
docker compose --profile evaluate up

# Or via SLURM
sbatch scripts/slurm/run_evaluation.slurm
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_EVALUATION_METRICS` | `qa,entity,relation,semantic` | Metrics to compute |
| `ARANDU_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for semantic embeddings |
| `ARANDU_QA_DIR` | `./qa_dataset` | QA dataset directory |
| `ARANDU_KG_DIR` | `./knowledge_graphs` | Knowledge graph directory |
| `ARANDU_EVAL_DIR` | `./evaluation` | Output directory |

### Metric Categories

| Category | Metrics | Pipeline |
|----------|---------|----------|
| `qa` | Exact Match, F1, BLEU | QA Pipeline |
| `entity` | Coverage, Density, Diversity | KG Pipeline |
| `relation` | Density, Connectivity | KG Pipeline |
| `semantic` | Coherence, Info Density, Coverage | KG Pipeline |

## Overall Score

The overall score is computed as a weighted average:

```
overall_score = (
    0.3 * qa_f1_score +
    0.2 * entity_diversity +
    0.2 * (relation_density / 3.0) +  # normalized
    0.3 * coherence_score
)
```

When running evaluation for only one pipeline, the score reflects available metrics.

## Interpreting Results

### Good Results

| Metric | Good Value |
|--------|------------|
| `overall_score` | > 0.70 |
| `qa_f1_score` | > 0.75 |
| `entity_diversity` | 0.3-0.6 |
| `relation_density` | 1.5-3.0 |
| `coherence_score` | > 0.75 |

---

# Programmatic Usage

## Loading Reports

```python
from arandu.schemas import EvaluationReport

# Load evaluation report
report = EvaluationReport.load("evaluation/report.json")

print(f"Overall Score: {report.overall_score:.3f}")
print(f"Total Documents: {report.total_documents}")

# QA metrics (if available)
if report.qa_f1_score:
    print(f"QA F1: {report.qa_f1_score:.3f}")

# KG metrics (if available)
if report.entity_coverage:
    print(f"Entity Diversity: {report.entity_coverage.entity_diversity:.3f}")

if report.relation_metrics:
    print(f"Relation Density: {report.relation_metrics.relation_density:.3f}")
```

## Creating Custom Reports

```python
from arandu.schemas import (
    EvaluationReport,
    EntityCoverageResult,
    RelationMetricsResult,
    SemanticQualityResult
)

# QA-only report
qa_report = EvaluationReport(
    dataset_name="qa_evaluation",
    total_documents=50,
    total_qa_pairs=500,
    qa_f1_score=0.75,
    qa_exact_match=0.68,
    qa_bleu_score=71.5
)

# KG-only report
kg_report = EvaluationReport(
    dataset_name="kg_evaluation",
    total_documents=50,
    entity_coverage=EntityCoverageResult(
        total_entities=1000,
        unique_entities=300,
        entity_density=15.0,
        entity_type_distribution={"PERSON": 100, "LOCATION": 200}
    ),
    relation_metrics=RelationMetricsResult(
        total_relations=800,
        unique_relations=250,
        relation_density=2.67
    )
)

# Save reports
qa_report.save("evaluation/qa_report.json")
kg_report.save("evaluation/kg_report.json")
```

---

# Monitoring Progress

## Docker Logs

```bash
# Watch evaluation logs
docker compose --profile evaluate logs -f arandu-eval
```

## SLURM Logs

```bash
# Monitor job output
tail -f logs/arandu-eval_<jobid>.out
```

---

# Best Practices

## For QA Evaluation

1. Generate enough QA pairs (10+ per document)
2. Use consistent LLM settings for generation
3. Compare across different model configurations

## For KG Evaluation

1. Verify graph file exists before evaluation
2. Run entity/relation metrics first (fast)
3. Use multilingual embedding models for non-English content
   - `paraphrase-multilingual-MiniLM-L12-v2` for Portuguese

## General

1. Save reports from different configurations
2. Track improvements over iterations
3. Use recommendations to guide improvements

---

**See also**: [QA Generation](qa-generation.md) | [KG Construction](kg-construction.md) | [Configuration](configuration.md)
