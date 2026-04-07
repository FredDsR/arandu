---
title: CEP QA Generation
description: Generate cognitively-calibrated QA pairs using the Cognitive Elicitation Pipeline with Bloom's Taxonomy scaffolding.
---

Generate cognitively-calibrated question-answer pairs using the **Cognitive Elicitation Pipeline (CEP)** with Bloom's Taxonomy scaffolding and LLM-as-a-Judge validation.

## Overview

The CEP pipeline extends the standard QA generation with three specialized modules:

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
# Start CEP QA generation with Ollama sidecar
docker compose --profile cep up
```

### Using SLURM

```bash
# Grace partition (NVIDIA L40S)
sbatch scripts/slurm/cep/grace.slurm

# Tupi partition (NVIDIA RTX 4090)
sbatch scripts/slurm/cep/tupi.slurm

# Sirius partition (AMD, CPU mode)
sbatch scripts/slurm/cep/sirius.slurm
```

### Using CLI

```bash
# Basic usage (validation enabled by default)
arandu generate-cep-qa results/ --output-dir cep_dataset/

# Disable validation for faster processing
arandu generate-cep-qa results/ --no-validate --output-dir cep_dataset/

# Export to JSONL format
arandu generate-cep-qa results/ --jsonl --output-dir cep_dataset/

# With pipeline ID for tracking
arandu generate-cep-qa results/ --id etno-project-001
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `ARANDU_QA_MODEL_ID` | `qwen3:14b` | Model for QA generation |
| `ARANDU_QA_OLLAMA_URL` | `http://ollama:11434/v1` | Ollama API URL |
| `ARANDU_QA_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `ARANDU_QA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `ARANDU_QA_WORKERS` | `2` | Parallel workers |

### CEP-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_CEP_ENABLE_VALIDATION` | `true` | Enable LLM-as-a-Judge validation |
| `ARANDU_CEP_VALIDATOR_PROVIDER` | `ollama` | Validator LLM provider |
| `ARANDU_CEP_VALIDATOR_MODEL_ID` | `qwen3:14b` | Validator model |
| `ARANDU_CEP_LANGUAGE` | `pt` | Prompt language (`pt` or `en`) |
| `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT` | `true` | Pass prior QA pairs to higher Bloom levels |
| `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS` | `10` | Max prior QA pairs to include as context |

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
arandu generate-cep-qa results/ \
  --bloom-dist "remember:0.1,understand:0.2,analyze:0.4,evaluate:0.3"
```

### Example .env Configuration

```bash
# QA Generation Settings
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_QUESTIONS_PER_DOCUMENT=12
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_WORKERS=4

# CEP-Specific Settings
ARANDU_CEP_ENABLE_VALIDATION=true
ARANDU_CEP_VALIDATOR_MODEL_ID=qwen3:14b
ARANDU_CEP_LANGUAGE=pt

# Directories
ARANDU_RESULTS_DIR=./results
ARANDU_CEP_DIR=./cep_dataset
```

## Usage Examples

### Basic CEP Generation

```bash
# Default configuration (Portuguese prompts, validation enabled)
docker compose --profile cep up
```

### Without LLM-as-a-Judge Validation

```bash
# Disable validation for faster processing
ARANDU_CEP_ENABLE_VALIDATION=false docker compose --profile cep up
```

### GPU-Accelerated Ollama

```bash
# Use GPU profile for faster inference
docker compose --profile cep-gpu up
```

### English Language Prompts

```bash
# Generate QA pairs with English prompts
ARANDU_CEP_LANGUAGE=en docker compose --profile cep up
```

### Export to JSONL

```bash
# Generate and export to JSONL format for training
arandu generate-cep-qa results/ --jsonl --output-dir cep_dataset/
```

## Output Format

CEP records are saved as JSON files in `cep_dataset/`:

```
cep_dataset/
├── <file_id_1>_cep_qa.json
├── <file_id_1>_cep_qa.jsonl   # JSONL export for this record
├── <file_id_2>_cep_qa.json
├── <file_id_2>_cep_qa.jsonl   # JSONL export for this record
└── cep_checkpoint.json          # For resumption
```

### QARecordCEP Schema

```json
{
  "source_file_id": "1abc123xyz",
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
      "tacit_inference": "Subida rápida do rio indica risco iminente de enchente",
      "generation_prompt": "Gere uma pergunta de análise que identifique relações de causa e efeito..."
    }
  ],
  "model_id": "qwen3:14b",
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
    "avg_overall_score": 0.79,
    "validation_pass_rate": 0.83
  },
  "cep_version": "1.0"
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
from arandu.schemas import QARecordCEP, QAPairCEP
from arandu.config import CEPConfig, QAConfig, get_cep_config

# Load existing CEP record
record = QARecordCEP.load("cep_dataset/1abc123xyz_cep_qa.json")

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

# Get CEP configuration
cep_config = get_cep_config()
print(f"Validation enabled: {cep_config.enable_validation}")
print(f"Bloom levels: {cep_config.bloom_levels}")
```

## Monitoring Progress

### Docker Logs

```bash
# Watch CEP generation logs
docker compose --profile cep logs -f arandu-cep

# Check Ollama status
docker compose --profile cep logs ollama
```

### SLURM Logs

```bash
# Monitor CEP job output
tail -f logs/cep_<partition>_<jobid>.out

# Check job status
squeue -u $USER
```

## Resumption

The pipeline automatically checkpoints progress. To resume an interrupted job:

```bash
# Simply restart - checkpoint is detected automatically
docker compose --profile cep up
```

The checkpoint file (`cep_dataset/cep_checkpoint.json`) tracks:
- Completed documents
- Failed documents (for retry)
- Processing statistics

## Best Practices

### Model Selection
- **qwen3:14b**: Default. Best balance of pt-BR support, JSON reliability, and inference speed (24GB GPU)
- **qwen3:8b**: Lighter alternative (~5GB VRAM) for development or constrained GPUs
- **llama3.3:70b**: Highest quality for final production runs (48GB+ GPU)

### Validation Strategy
1. Keep validation enabled (default) for quality assurance
2. Disable validation (`--no-validate`) for faster iteration during development
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
docker compose --profile cep exec ollama ollama list

# Check for minimum context length
# Text must be at least 100 characters
```

### Low Bloom Calibration Scores

- Ensure model has good reasoning capabilities
- Try larger model: `ARANDU_QA_MODEL_ID=llama3.3:70b`
- Lower temperature for more consistent output: `ARANDU_QA_TEMPERATURE=0.5`

### Validation Rejecting Most Pairs

- Lower threshold: default is 0.6, try 0.5
- Check validator model capability
- Review prompt language matches content language

### Ollama Connection Issues

```bash
# Restart Ollama service
docker compose --profile cep restart ollama

# Check Ollama health
docker compose --profile cep exec ollama curl http://localhost:11434/api/tags

# Pull model if missing
docker compose --profile cep exec ollama ollama pull qwen3:14b
```

---

**See also**: [QA Generation](/guides/qa-generation/) | [Configuration](/configuration/) | [Docker Deployment](/deployment/docker/)
