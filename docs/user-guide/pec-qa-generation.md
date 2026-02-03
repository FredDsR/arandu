# PEC QA Generation Guide

Generate cognitively-calibrated question-answer pairs using the **Pipeline de Elicitação Cognitiva** (Cognitive Elicitation Pipeline) with Bloom's Taxonomy scaffolding and LLM-as-a-Judge validation.

## Overview

The PEC pipeline extends the standard QA generation with three specialized modules:

1. **Module I: Bloom Scaffolding** - Generates questions distributed across Bloom's taxonomy levels
2. **Module II: Reasoning & Grounding** - Enriches higher-level questions with reasoning traces
3. **Module III: LLM-as-a-Judge** - Validates QA pairs for faithfulness, Bloom calibration, and informativeness

### Key Features

- **Cognitive Scaffolding**: Questions progress from basic recall to advanced analysis
- **Bloom Taxonomy Levels**: remember, understand, apply, analyze, evaluate, create
- **Reasoning Traces**: Explicit logical connections for complex questions
- **Multi-hop Detection**: Identifies questions requiring distant information connection
- **Tacit Knowledge Extraction**: Surfaces implicit domain knowledge
- **Quality Validation**: LLM-based scoring for faithfulness and informativeness

## Prerequisites

- Transcription results in `results/` directory
- Docker with Compose v2
- LLM provider (Ollama recommended)

## Quick Start

### Using Docker Compose

```bash
# Start PEC QA generation with Ollama sidecar
docker compose --profile pec up
```

### Using SLURM

```bash
# Grace partition (NVIDIA L40S)
sbatch scripts/slurm/pec/grace.slurm

# Tupi partition (NVIDIA RTX 4090)
sbatch scripts/slurm/pec/tupi.slurm

# Sirius partition (AMD, CPU mode)
sbatch scripts/slurm/pec/sirius.slurm
```

### Using CLI

```bash
# Basic usage
gtranscriber generate-pec-qa results/ --output-dir pec_dataset/

# With validation enabled
gtranscriber generate-pec-qa results/ --validate --output-dir pec_dataset/

# Export to JSONL format
gtranscriber generate-pec-qa results/ --jsonl --output-dir pec_dataset/
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_QA_MODEL_ID` | `llama3.1:8b` | Model for QA generation |
| `GTRANSCRIBER_QA_OLLAMA_URL` | `http://ollama:11434` | Ollama API URL |
| `GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `GTRANSCRIBER_QA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `GTRANSCRIBER_QA_WORKERS` | `2` | Parallel workers |

### PEC-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GTRANSCRIBER_PEC_ENABLE_VALIDATION` | `false` | Enable LLM-as-a-Judge validation |
| `GTRANSCRIBER_PEC_VALIDATOR_PROVIDER` | `ollama` | Validator LLM provider |
| `GTRANSCRIBER_PEC_VALIDATOR_MODEL_ID` | `llama3.1:8b` | Validator model |
| `GTRANSCRIBER_PEC_LANGUAGE` | `pt` | Prompt language (`pt` or `en`) |

### Bloom Distribution

Default distribution allocates questions across cognitive levels:

| Level | Default Weight | Description |
|-------|---------------|-------------|
| `remember` | 20% | Recall explicit facts |
| `understand` | 30% | Explain and interpret concepts |
| `analyze` | 30% | Identify relationships and patterns |
| `evaluate` | 20% | Make judgments and justify decisions |

Customize via CLI:
```bash
gtranscriber generate-pec-qa results/ \
  --bloom-dist "remember=0.1,understand=0.2,analyze=0.4,evaluate=0.3"
```

### Example .env Configuration

```bash
# QA Generation Settings
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=12
GTRANSCRIBER_QA_TEMPERATURE=0.7
GTRANSCRIBER_QA_WORKERS=4

# PEC-Specific Settings
GTRANSCRIBER_PEC_ENABLE_VALIDATION=true
GTRANSCRIBER_PEC_VALIDATOR_MODEL_ID=llama3.1:8b
GTRANSCRIBER_PEC_LANGUAGE=pt

# Directories
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_PEC_DIR=./pec_dataset
```

## Usage Examples

### Basic PEC Generation

```bash
# Default configuration (Portuguese prompts, no validation)
docker compose --profile pec up
```

### With LLM-as-a-Judge Validation

```bash
# Enable validation to filter low-quality pairs
GTRANSCRIBER_PEC_ENABLE_VALIDATION=true docker compose --profile pec up
```

### GPU-Accelerated Ollama

```bash
# Use GPU profile for faster inference
docker compose --profile pec-gpu up
```

### English Language Prompts

```bash
# Generate QA pairs with English prompts
GTRANSCRIBER_PEC_LANGUAGE=en docker compose --profile pec up
```

### Export to JSONL

```bash
# Generate and export to JSONL format for training
gtranscriber generate-pec-qa results/ --jsonl --output-dir pec_dataset/
```

## Output Format

PEC records are saved as JSON files in `pec_dataset/`:

```
pec_dataset/
├── <gdrive_id_1>_pec_qa.json
├── <gdrive_id_2>_pec_qa.json
├── pec_checkpoint.json          # For resumption
└── pec_qa_export.jsonl          # Optional JSONL export
```

### QARecordPEC Schema

```json
{
  "source_gdrive_id": "1abc123xyz",
  "source_filename": "interview_2023.mp3",
  "transcription_text": "O pescador contou que quando o rio sobe...",
  "qa_pairs": [
    {
      "question": "O que o pescador faz quando o rio sobe?",
      "answer": "Guarda o barco para evitar perda",
      "context": "Se o rio sobe rápido, guardo o barco para evitar perda",
      "question_type": "factual",
      "confidence": 0.92,
      "bloom_level": "remember",
      "reasoning_trace": null,
      "is_multi_hop": false,
      "hop_count": null,
      "tacit_inference": null
    },
    {
      "question": "Por que o pescador guarda o barco quando o rio sobe?",
      "answer": "Para evitar perda do equipamento devido ao risco de enchente",
      "context": "Se o rio sobe rápido, guardo o barco para evitar perda",
      "question_type": "conceptual",
      "confidence": 0.88,
      "bloom_level": "analyze",
      "reasoning_trace": "Fato: rio sobe → Ação: guardar barco → Razão: evitar perda",
      "is_multi_hop": false,
      "hop_count": null,
      "tacit_inference": "Subida rápida do rio indica risco iminente de enchente"
    }
  ],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "generation_timestamp": "2026-02-03T10:30:00Z",
  "total_pairs": 12,
  "validated_pairs": 10,
  "bloom_distribution": {
    "remember": 3,
    "understand": 4,
    "analyze": 3,
    "evaluate": 2
  },
  "validation_summary": {
    "avg_faithfulness": 0.85,
    "avg_bloom_calibration": 0.78,
    "avg_informativeness": 0.72,
    "avg_overall": 0.79
  },
  "pec_version": "1.0"
}
```

### JSONL Export Format

Each line contains one QA pair:

```json
{"question": "O que aconteceu?", "answer": "...", "context": "...", "bloom_level": "remember", "confidence": 0.92}
{"question": "Por que isso aconteceu?", "answer": "...", "context": "...", "bloom_level": "analyze", "reasoning_trace": "..."}
```

## Bloom Taxonomy Levels

### remember
- **Description**: Recall explicit facts from the text
- **Question Starters**: O que, Quem, Quando, Onde, Qual
- **Example**: "O que o pescador faz quando o rio sobe?"

### understand
- **Description**: Explain, interpret, or summarize concepts
- **Question Starters**: Explique, Descreva, Por que (basic), Como funciona
- **Example**: "Explique o processo de preparação do barco."

### apply
- **Description**: Use knowledge in new situations
- **Question Starters**: Como você aplicaria, O que aconteceria se
- **Example**: "Como o pescador aplicaria este conhecimento em outra situação?"

### analyze
- **Description**: Identify relationships, patterns, and connections
- **Question Starters**: Por que, Quais são as causas, Compare, Relacione
- **Example**: "Por que a subida do rio representa um risco para o equipamento?"

### evaluate
- **Description**: Make judgments and justify decisions
- **Question Starters**: Avalie, Justifique, Qual a importância, É adequado
- **Example**: "A decisão de guardar o barco foi acertada? Justifique."

### create
- **Description**: Propose solutions or create something new
- **Question Starters**: Proponha, Como melhorar, Elabore, Sugira
- **Example**: "Proponha uma solução alternativa para proteger o barco."

## Validation Criteria

When validation is enabled, each QA pair is scored on three criteria:

### Faithfulness (40% weight)
Is the answer grounded in the provided context?

| Score | Description |
|-------|-------------|
| 1.0 | Answer completely grounded in text |
| 0.8 | Answer well grounded with minimal inferences |
| 0.6 | Answer mostly grounded with some non-trivial inferences |
| 0.4 | Answer partially grounded with significant inferences |
| 0.2 | Answer weakly grounded |
| 0.0 | Answer not grounded, hallucinated, or contradictory |

### Bloom Calibration (30% weight)
Does the question match the proposed cognitive level?

| Score | Description |
|-------|-------------|
| 1.0 | Perfectly calibrated - requires exactly the declared level |
| 0.8 | Well calibrated - requires predominantly the declared level |
| 0.6 | Reasonably calibrated with some overlap |
| 0.4 | Undercalibrated - requires lower cognitive level |
| 0.0 | Completely miscalibrated |

### Informativeness (30% weight)
Does the answer reveal non-obvious or tacit knowledge?

| Score | Description |
|-------|-------------|
| 1.0 | Reveals significant tacit knowledge or practical know-how |
| 0.8 | Reveals useful and non-obvious knowledge |
| 0.6 | Reveals moderately useful contextual information |
| 0.4 | Common but well-articulated information |
| 0.0 | Trivial or obvious information |

The overall score is a weighted average. QA pairs below the threshold (default 0.6) are marked as invalid.

## Programmatic Usage

```python
from gtranscriber.schemas import QARecordPEC, QAPairPEC
from gtranscriber.config import PECConfig, QAConfig, get_pec_config

# Load existing PEC record
record = QARecordPEC.load("pec_dataset/1abc123xyz_pec_qa.json")

# Access QA pairs with Bloom levels
for qa in record.qa_pairs:
    print(f"Level: {qa.bloom_level}")
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
    if qa.reasoning_trace:
        print(f"Reasoning: {qa.reasoning_trace}")
    print()

# Filter by Bloom level
analyze_pairs = [qa for qa in record.qa_pairs if qa.bloom_level == "analyze"]

# Check Bloom distribution
print(f"Distribution: {record.bloom_distribution}")

# Export to JSONL string
jsonl_content = record.to_jsonl()

# Export to JSONL file
record.to_jsonl("output.jsonl")

# Get PEC configuration
pec_config = get_pec_config()
print(f"Validation enabled: {pec_config.enable_validation}")
print(f"Bloom levels: {pec_config.bloom_levels}")
```

## Monitoring Progress

### Docker Logs

```bash
# Watch PEC generation logs
docker compose --profile pec logs -f gtranscriber-pec

# Check Ollama status
docker compose --profile pec logs ollama
```

### SLURM Logs

```bash
# Monitor job output
tail -f logs/pec_<partition>_<jobid>.out

# Check job status
squeue -u $USER
```

## Resumption

The pipeline automatically checkpoints progress. To resume an interrupted job:

```bash
# Simply restart - checkpoint is detected automatically
docker compose --profile pec up
```

The checkpoint file (`pec_dataset/pec_checkpoint.json`) tracks:
- Completed documents
- Failed documents (for retry)
- Processing statistics

## Best Practices

### Model Selection
- **llama3.1:8b**: Balanced speed/quality for most use cases
- **llama3.1:70b**: Higher quality for final production runs
- **qwen2.5:14b**: Good alternative with strong PT-BR support

### Validation Strategy
1. Run initial generation without validation for speed
2. Enable validation for final dataset quality check
3. Use validation threshold of 0.6-0.7 to balance coverage and quality

### Bloom Distribution
- Start with default distribution
- Increase `analyze`/`evaluate` for research datasets
- Increase `remember`/`understand` for educational datasets

### Language Selection
- Use `pt` (Portuguese) for PT-BR content (default)
- Use `en` (English) for English content
- Prompts are optimized for each language

## Troubleshooting

### No QA Pairs Generated

```bash
# Check if transcription results exist
ls -la results/*.json

# Verify Ollama is running
docker compose --profile pec exec ollama ollama list

# Check for minimum context length
# Text must be at least 100 characters
```

### Low Bloom Calibration Scores

- Ensure model has good reasoning capabilities
- Try larger model: `GTRANSCRIBER_QA_MODEL_ID=llama3.1:70b`
- Lower temperature for more consistent output: `GTRANSCRIBER_QA_TEMPERATURE=0.5`

### Validation Rejecting Most Pairs

- Lower threshold: default is 0.6, try 0.5
- Check validator model capability
- Review prompt language matches content language

### Ollama Connection Issues

```bash
# Restart Ollama service
docker compose --profile pec restart ollama

# Check Ollama health
docker compose --profile pec exec ollama curl http://localhost:11434/api/tags

# Pull model if missing
docker compose --profile pec exec ollama ollama pull llama3.1:8b
```

---

**See also**: [QA Generation](qa-generation.md) | [Configuration](configuration.md) | [SLURM Deployment](../deployment/slurm.md)
