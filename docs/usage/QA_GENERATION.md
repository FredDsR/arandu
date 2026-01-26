# QA Generation Guide

Generate synthetic question-answer pairs from transcription results using LLM-based extraction.

## Overview

The QA generation pipeline creates extractive QA pairs from transcribed text using multiple question generation strategies. Each QA pair includes:

- **Question**: Generated question about the content
- **Answer**: Extractive answer from the source text
- **Context**: Source text segment
- **Question Type**: Strategy used (factual, conceptual, temporal, entity)
- **Confidence**: Generation confidence score
- **Timestamps**: Optional time references from transcription

## Prerequisites

- Transcription results in `results/` directory
- Docker with Compose v2
- LLM provider (Ollama recommended)

## Quick Start

### Using Docker Compose

```bash
# Start QA generation with Ollama sidecar
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
| `GTRANSCRIBER_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_QA_MODEL_ID` | `llama3.1:8b` | Model for QA generation |
| `GTRANSCRIBER_QA_OLLAMA_URL` | `http://ollama:11434` | Ollama API URL |
| `GTRANSCRIBER_QUESTIONS_PER_DOCUMENT` | `10` | QA pairs per document |
| `GTRANSCRIBER_QA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `GTRANSCRIBER_QA_STRATEGIES` | `factual,conceptual` | Question strategies |
| `GTRANSCRIBER_WORKERS` | `2` | Parallel workers |

### Question Strategies

| Strategy | Description | Example Questions |
|----------|-------------|-------------------|
| `factual` | Who, what, when, where | "What caused the flooding?" |
| `conceptual` | Why, how explanations | "Why did the community evacuate?" |
| `temporal` | Time-based questions | "When did the event occur?" |
| `entity` | Entity-focused questions | "Who was the mayor during the crisis?" |

### Example .env Configuration

```bash
# QA Generation Settings
GTRANSCRIBER_QA_PROVIDER=ollama
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=15
GTRANSCRIBER_QA_STRATEGIES=factual,conceptual,temporal
GTRANSCRIBER_QA_TEMPERATURE=0.7
GTRANSCRIBER_WORKERS=4

# Directories
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_QA_DIR=./qa_dataset
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
GTRANSCRIBER_QA_MODEL_ID=qwen2.5:14b docker compose --profile qa up
```

### More Questions per Document

```bash
# Generate 20 QA pairs per document
GTRANSCRIBER_QUESTIONS_PER_DOCUMENT=20 docker compose --profile qa up
```

### Using OpenAI

```bash
# Use OpenAI instead of Ollama
export GTRANSCRIBER_QA_PROVIDER=openai
export GTRANSCRIBER_QA_MODEL_ID=gpt-4o-mini
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

QA records are saved as JSON files in `qa_dataset/`:

```
qa_dataset/
├── qa_<gdrive_id_1>.json
├── qa_<gdrive_id_2>.json
└── qa_checkpoint.json      # For resumption
```

### QARecord Schema

```json
{
  "source_gdrive_id": "1abc123xyz",
  "source_filename": "interview_2023.mp3",
  "transcription_text": "The flooding was caused by...",
  "qa_pairs": [
    {
      "question": "What caused the flooding in the region?",
      "answer": "Heavy rainfall combined with poor drainage",
      "context": "The flooding was caused by heavy rainfall combined with poor drainage infrastructure...",
      "question_type": "factual",
      "confidence": 0.92,
      "start_time": 45.3,
      "end_time": 52.1
    }
  ],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "generation_timestamp": "2026-01-26T10:30:00Z",
  "total_pairs": 12
}
```

## Programmatic Usage

```python
from gtranscriber.schemas import QARecord, QAPair

# Load existing QA record
record = QARecord.load("qa_dataset/qa_1abc123xyz.json")

# Access QA pairs
for qa in record.qa_pairs:
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
    print(f"Type: {qa.question_type}, Confidence: {qa.confidence}")
    print()

# Create new QA pair
new_qa = QAPair(
    question="What was the impact?",
    answer="widespread damage",
    context="The flooding caused widespread damage to homes.",
    question_type="factual",
    confidence=0.85
)
```

## Monitoring Progress

### Docker Logs

```bash
# Watch QA generation logs
docker compose --profile qa logs -f gtranscriber-qa

# Check Ollama status
docker compose --profile qa logs ollama
```

### SLURM Logs

```bash
# Monitor job output
tail -f logs/gtranscriber-qa_<jobid>.out

# Check job status
squeue -u $USER
```

## Resumption

The pipeline automatically checkpoints progress. To resume an interrupted job:

```bash
# Simply restart - checkpoint is detected automatically
docker compose --profile qa up
```

The checkpoint file (`qa_dataset/qa_checkpoint.json`) tracks:
- Completed documents
- Failed documents (for retry)
- Processing statistics

## Best Practices

1. **Model Selection**
   - Use `llama3.1:8b` for balanced speed/quality
   - Use `llama3.1:70b` for higher quality (slower)
   - Use `llama3.2:3b` for faster processing

2. **Question Strategies**
   - Start with `factual,conceptual` (default)
   - Add `temporal` for time-sensitive content
   - Add `entity` for content with many named entities

3. **Temperature**
   - Lower (0.3-0.5): More consistent, less creative
   - Default (0.7): Balanced
   - Higher (0.9-1.0): More varied questions

4. **Workers**
   - Match to available CPU cores / 2
   - Reduce if experiencing OOM errors

## Troubleshooting

### No QA Pairs Generated

```bash
# Check if transcription results exist
ls -la results/*.json

# Verify Ollama is running
docker compose --profile qa exec ollama ollama list
```

### Low Quality Questions

- Increase model size: `GTRANSCRIBER_QA_MODEL_ID=llama3.1:70b`
- Lower temperature: `GTRANSCRIBER_QA_TEMPERATURE=0.5`
- Ensure transcription quality is good

### Ollama Connection Refused

```bash
# Restart Ollama service
docker compose --profile qa restart ollama

# Check Ollama health
docker compose --profile qa exec ollama curl http://localhost:11434/api/tags
```

---

**See also**: [KG Construction](KG_CONSTRUCTION.md) | [Evaluation](EVALUATION.md) | [Configuration](CONFIGURATION.md)
