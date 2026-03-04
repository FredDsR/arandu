# QA Generation Guide

Generate cognitively-scaffolded question-answer pairs from transcription results using the CEP (Cognitive Elicitation Pipeline).

## Overview

The CEP QA generation pipeline creates extractive QA pairs from transcribed text using Bloom's Taxonomy cognitive scaffolding with optional LLM-as-a-Judge validation. Each QA pair includes:

- **Question**: Generated question calibrated to a Bloom cognitive level
- **Answer**: Extractive answer from the source text
- **Context**: Source text segment
- **Bloom Level**: Cognitive level (remember, understand, analyze, evaluate, etc.)
- **Confidence**: Generation confidence score
- **Reasoning Trace**: Logical connection chain (for higher-level questions)
- **Timestamps**: Optional time references from transcription

## Prerequisites

- Transcription results in `results/` directory
- Docker with Compose v2
- LLM provider (Ollama recommended)

## Quick Start

### Using Docker Compose

```bash
# Start CEP QA generation with Ollama sidecar
docker compose --profile qa up
```

### Using SLURM

```bash
sbatch scripts/slurm/run_qa_generation.slurm
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `ARANDU_QA_MODEL_ID` | `qwen3:14b` | Model for QA generation |
| `ARANDU_QA_OLLAMA_URL` | `http://localhost:11434/v1` | Ollama API URL |
| `ARANDU_QA_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `ARANDU_QA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `ARANDU_WORKERS` | `2` | Parallel workers |

### CEP-Specific Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_CEP_ENABLE_VALIDATION` | `true` | Enable LLM-as-a-Judge validation |
| `ARANDU_CEP_BLOOM_LEVELS` | `remember,understand,analyze,evaluate` | Bloom levels to generate |
| `ARANDU_CEP_VALIDATION_THRESHOLD` | `0.6` | Minimum score to pass validation |
| `ARANDU_CEP_VALIDATOR_MODEL_ID` | `qwen3:14b` | Model for validation |

### Example .env Configuration

```bash
# QA Generation Settings
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_QUESTIONS_PER_DOCUMENT=15
ARANDU_QA_TEMPERATURE=0.7
ARANDU_WORKERS=4

# CEP Settings
ARANDU_CEP_ENABLE_VALIDATION=true
ARANDU_CEP_BLOOM_LEVELS=remember,understand,analyze,evaluate

# Directories
ARANDU_RESULTS_DIR=./results
ARANDU_QA_DIR=./qa_dataset
```

## Usage Examples

### Basic Usage

```bash
# Default configuration
docker compose --profile qa up
```

### Custom Model

```bash
# Use different model
ARANDU_QA_MODEL_ID=qwen2.5:14b docker compose --profile qa up
```

### More Questions per Document

```bash
# Generate 20 QA pairs per document
ARANDU_QA_QUESTIONS_PER_DOCUMENT=20 docker compose --profile qa up
```

### Using OpenAI

```bash
# Use OpenAI instead of Ollama
export ARANDU_QA_PROVIDER=openai
export ARANDU_QA_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...
docker compose --profile qa up
```

### SLURM with Custom Settings

```bash
# Submit with custom model and workers
QA_MODEL=llama3.1:70b WORKERS=8 QUESTIONS_PER_DOCUMENT=20 \
  sbatch scripts/slurm/run_qa_generation.slurm
```

## Output Format

CEP QA records are saved as JSON files in the versioned results directory:

```
results/<pipeline_id>/cep/outputs/
├── <file_id_1>_cep_qa.json
├── <file_id_2>_cep_qa.json
└── cep_checkpoint.json      # For resumption
```

### QARecordCEP Schema

```json
{
  "source_file_id": "1abc123xyz",
  "source_filename": "interview_2023.mp3",
  "transcription_text": "The flooding was caused by...",
  "qa_pairs": [
    {
      "question": "What caused the flooding in the region?",
      "answer": "Heavy rainfall combined with poor drainage",
      "context": "The flooding was caused by heavy rainfall...",
      "question_type": "factual",
      "confidence": 0.92,
      "bloom_level": "remember",
      "reasoning_trace": "Direct recall from text",
      "generation_prompt": "Generate a factual question that tests recall...",
      "start_time": 45.3,
      "end_time": 52.1
    }
  ],
  "model_id": "qwen3:14b",
  "provider": "ollama",
  "generation_timestamp": "2026-01-26T10:30:00Z",
  "total_pairs": 12,
  "bloom_distribution": {
    "remember": 3,
    "understand": 4,
    "analyze": 3,
    "evaluate": 2
  },
  "validated_pairs": 10,
  "validation_summary": {
    "avg_faithfulness": 0.85,
    "avg_bloom_calibration": 0.78,
    "avg_informativeness": 0.72,
    "avg_overall_score": 0.79,
    "validation_pass_rate": 0.83
  }
}
```

## Programmatic Usage

```python
from arandu.schemas import QARecordCEP, QAPairCEP

# Load existing CEP QA record
record = QARecordCEP.load("results/pipeline_id/cep/outputs/1abc123xyz_cep_qa.json")

# Access QA pairs
for qa in record.qa_pairs:
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
    print(f"Bloom: {qa.bloom_level}, Confidence: {qa.confidence}")
    print()

# Export to JSONL for KGQA training
record.to_jsonl("output.jsonl")
```

## Monitoring Progress

### Docker Logs

```bash
# Watch QA generation logs
docker compose --profile qa logs -f arandu-qa

# Check Ollama status
docker compose --profile qa logs ollama
```

### SLURM Logs

```bash
# Monitor job output
tail -f logs/arandu-qa_<jobid>.out

# Check job status
squeue -u $USER
```

## Resumption

The pipeline automatically checkpoints progress. To resume an interrupted job:

```bash
# Simply restart - checkpoint is detected automatically
docker compose --profile qa up
```

The checkpoint file tracks:
- Completed documents
- Failed documents (for retry)
- Processing statistics

## Best Practices

1. **Model Selection**
   - Use `qwen3:14b` for balanced speed/quality
   - Use `llama3.1:70b` for higher quality (slower)
   - Use `llama3.2:3b` for faster processing

2. **Temperature**
   - Lower (0.3-0.5): More consistent, less creative
   - Default (0.7): Balanced
   - Higher (0.9-1.0): More varied questions

3. **Workers**
   - Match to available CPU cores / 2
   - Reduce if experiencing OOM errors

4. **Validation**
   - Enable for production datasets (default)
   - Disable for quick iteration (`ARANDU_CEP_ENABLE_VALIDATION=false`)

## Troubleshooting

### No QA Pairs Generated

```bash
# Check if transcription results exist
ls -la results/*.json

# Verify Ollama is running
docker compose --profile qa exec ollama ollama list
```

### Low Quality Questions

- Increase model size: `ARANDU_QA_MODEL_ID=llama3.1:70b`
- Lower temperature: `ARANDU_QA_TEMPERATURE=0.5`
- Ensure transcription quality is good

### Ollama Connection Refused

```bash
# Restart Ollama service
docker compose --profile qa restart ollama

# Check Ollama health
docker compose --profile qa exec ollama curl http://localhost:11434/v1/api/tags
```

---

**See also**: [KG Construction](kg-construction.md) | [Evaluation](evaluation.md) | [Configuration](configuration.md)
