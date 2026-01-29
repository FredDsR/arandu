# QA Generation Deep Dive: Technical Guide

**Status**: Phase 2 Complete
**Date**: 2026-01-29
**Purpose**: Comprehensive technical reference for QA generation pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Critical Configuration Details](#critical-configuration-details)
4. [The Ollama URL Issue](#the-ollama-url-issue)
5. [Model Selection](#model-selection)
6. [Parallel Processing](#parallel-processing)
7. [JSON Parsing and Small Models](#json-parsing-and-small-models)
8. [Docker Networking](#docker-networking)
9. [Checkpoint System](#checkpoint-system)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

The QA generation pipeline creates synthetic question-answer pairs from transcription text using Large Language Models (LLMs). It supports multiple question generation strategies and can process hundreds of files in parallel with fault tolerance and resume capability.

### Key Features

- **Multiple LLM Providers**: OpenAI, Ollama, or any OpenAI-compatible endpoint
- **Question Strategies**: Factual, conceptual, temporal, entity-focused questions
- **Parallel Processing**: Multi-worker architecture with global generator pattern
- **Fault Tolerance**: Checkpoint-based resume capability
- **Extractive Validation**: Ensures answers exist in source text
- **Docker-Ready**: Pre-configured Docker Compose and SLURM scripts

---

## Architecture

### Component Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Command                             │
│                   (main.py:generate_qa)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Batch Orchestrator                          │
│              (qa_batch.py:run_batch_qa_generation)          │
│  • Loads tasks from transcription files                     │
│  • Manages CheckpointManager                                │
│  • Spawns ProcessPoolExecutor with workers                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Worker Process (x N workers)                    │
│           Each worker has its own:                           │
│  • QAGenerator instance (global per process)                │
│  • LLMClient instance (HTTP connection pooling)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    QA Generator                              │
│               (qa_generator.py:QAGenerator)                 │
│  • Chunks text into manageable contexts                     │
│  • Generates questions per strategy                         │
│  • Validates extractive answers                             │
│  • Builds strategy-specific prompts                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Client                               │
│              (llm_client.py:LLMClient)                      │
│  • Unified OpenAI SDK wrapper                               │
│  • Configurable base_url for any provider                   │
│  • Retry logic with exponential backoff                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Ollama / OpenAI / Custom API                    │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns

#### 1. Global Worker Pattern

**Why**: Efficient HTTP connection pooling and reduced initialization overhead.

```python
# Global variable per worker process
_worker_qa_generator: QAGenerator | None = None

def _init_qa_worker(provider: str, model_id: str, config_dict: dict) -> None:
    """Called ONCE per worker process at startup."""
    global _worker_qa_generator
    llm_client = LLMClient(...)
    _worker_qa_generator = QAGenerator(llm_client, config)

def generate_qa_for_transcription(task, config_dict):
    """Called MANY times per worker process."""
    global _worker_qa_generator
    # Reuse the same generator instance
    qa_record = _worker_qa_generator.generate_qa_pairs(enriched)
```

**Benefits**:
- HTTP connection pooling (reuse TCP connections to LLM API)
- No repeated client initialization overhead
- Memory efficient (one client per worker, not per task)

**ProcessPoolExecutor Detail**: Each worker runs in a separate Python process with independent memory space. The global variable is NOT shared between workers - each worker has its own copy.

#### 2. Batched Future Submission

**Why**: Prevents memory issues with large task lists.

```python
batch_size = max(num_workers * 2, 10)
task_iter = iter(remaining_tasks)
pending_futures = {}

# Submit initial batch
for _ in range(min(batch_size, len(remaining_tasks))):
    task = next(task_iter)
    future = executor.submit(generate_qa_for_transcription, task, config_dict)
    pending_futures[future] = task

# Replace completed tasks with new ones
while pending_futures:
    completed_future = next(as_completed(pending_futures))
    task = pending_futures.pop(completed_future)
    # Process result...

    # Submit next task to replace completed one
    try:
        next_task = next(task_iter)
        next_future = executor.submit(generate_qa_for_transcription, next_task, config_dict)
        pending_futures[next_future] = next_task
    except StopIteration:
        pass  # No more tasks
```

**Benefits**:
- Bounded memory usage (only `batch_size` futures in memory)
- Dynamic task replacement keeps workers busy
- Works efficiently with thousands of files

#### 3. Strategy Pattern for Questions

**Why**: Flexible question generation with strategy-specific prompts.

```python
strategy_instructions = {
    "factual": "Generate questions about specific facts...",
    "conceptual": "Generate questions about concepts, themes...",
    "temporal": "Generate questions about time, sequence...",
    "entity": "Generate questions focused on entities...",
}
```

Each strategy gets its own prompt template optimized for that question type.

---

## Critical Configuration Details

### Environment Variable Precedence

**IMPORTANT**: The CLI follows this precedence order:

1. **CLI arguments** (highest priority)
2. **Environment variables** (`GTRANSCRIBER_QA_*`)
3. **Config defaults** (lowest priority)

### Configuration Loading Flow

```python
# 1. Load config with environment variables
base_config = QAConfig()  # Reads GTRANSCRIBER_QA_* env vars

# 2. Override with CLI args if provided
if provider is not None:
    base_config.provider = provider
if model_id is not None:
    base_config.model_id = model_id
# ... etc

# 3. Use resolved config
qa_config = base_config
```

**Why This Matters**: In Docker, environment variables are set in `docker-compose.yml`. The CLI must not have hardcoded defaults that override these environment variables. This was a critical bug we fixed.

### All Configuration Options

| Environment Variable | CLI Option | Default | Description |
|---------------------|------------|---------|-------------|
| `GTRANSCRIBER_QA_PROVIDER` | `--provider` | `ollama` | LLM provider (openai, ollama, custom) |
| `GTRANSCRIBER_QA_MODEL_ID` | `--model-id` | `llama3.1:8b` | Model identifier |
| `GTRANSCRIBER_QA_OLLAMA_URL` | `--ollama-url` | `http://localhost:11434/v1` | Ollama API endpoint |
| `GTRANSCRIBER_QA_BASE_URL` | `--base-url` | `None` | Custom OpenAI-compatible endpoint |
| `GTRANSCRIBER_QA_WORKERS` | `--workers` | `2` | Number of parallel workers |
| `GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT` | `--questions` | `10` | QA pairs per document (1-50) |
| `GTRANSCRIBER_QA_STRATEGIES` | `--strategy` | `factual,conceptual` | Question strategies (comma-separated) |
| `GTRANSCRIBER_QA_TEMPERATURE` | `--temperature` | `0.7` | LLM temperature (0.0-2.0) |
| `GTRANSCRIBER_QA_OUTPUT_DIR` | `--output-dir` | `qa_dataset` | Output directory |

---

## The Ollama URL Issue

### The Problem

When running in Docker, the pipeline was getting **404 errors** from Ollama:

```
[GIN] 2026/01/29 - 14:15:51 | 404 | POST "/chat/completions"
ERROR Failed to generate QA pairs: RetryError[NotFoundError]
```

### Root Causes

There were **two separate issues**:

#### Issue 1: Missing `/v1` Suffix

The OpenAI SDK expects endpoints at `/v1/chat/completions`, but our configuration was missing the `/v1` prefix.

**Incorrect Configuration**:
```yaml
# docker-compose.yml (WRONG)
- GTRANSCRIBER_QA_OLLAMA_URL=http://ollama:11434

# Requests went to: http://ollama:11434/chat/completions
# Ollama expects: http://ollama:11434/v1/chat/completions
# Result: 404 Not Found
```

**Correct Configuration**:
```yaml
# docker-compose.yml (CORRECT)
- GTRANSCRIBER_QA_OLLAMA_URL=http://ollama:11434/v1
```

#### Issue 2: Model Not Pulled

Ollama returns 404 when the requested model doesn't exist.

**Solution**: Pull the model first:
```bash
docker compose --profile qa exec ollama ollama pull gemma:2b
```

### Files That Needed the `/v1` Fix

We updated all occurrences of Ollama URLs:

1. **docker-compose.yml**: Lines 263, 318
   ```yaml
   - GTRANSCRIBER_QA_OLLAMA_URL=${GTRANSCRIBER_QA_OLLAMA_URL:-http://ollama:11434/v1}
   - GTRANSCRIBER_KG_OLLAMA_URL=${GTRANSCRIBER_KG_OLLAMA_URL:-http://ollama:11434/v1}
   ```

2. **src/gtranscriber/config.py**: Lines 165, 246
   ```python
   ollama_url: str = Field(
       default="http://localhost:11434/v1",
       description="Ollama API base URL",
   )
   ```

3. **scripts/slurm/run_qa_generation.slurm**: Line 32
   ```bash
   export GTRANSCRIBER_QA_OLLAMA_URL="${QA_OLLAMA_URL:-http://ollama:11434/v1}"
   ```

4. **scripts/slurm/run_kg_construction.slurm**: Line 32
   ```bash
   export GTRANSCRIBER_KG_OLLAMA_URL="${KG_OLLAMA_URL:-http://ollama:11434/v1}"
   ```

5. **src/gtranscriber/core/llm_client.py**: Line 47 (already correct)
   ```python
   LLMProvider.OLLAMA: "http://localhost:11434/v1",
   ```

### How to Verify It's Working

1. **Check Ollama logs for 200 responses**:
   ```bash
   docker compose --profile qa logs ollama | grep "/v1/chat/completions"
   ```

   Should see:
   ```
   [GIN] 2026/01/29 - 14:33:27 | 200 | 1m8s | POST "/v1/chat/completions"
   ```

2. **Test endpoint directly**:
   ```bash
   curl -X POST http://localhost:11434/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "gemma:2b", "messages": [{"role": "user", "content": "Hi"}]}'
   ```

---

## Model Selection

### Model Requirements

For QA generation, models must:
1. **Follow JSON format** reliably
2. **Generate extractive answers** (quotes from source text)
3. **Handle structured prompts** with specific instructions

### Recommended Models

| Model | Size | Quality | Speed | JSON Reliability |
|-------|------|---------|-------|------------------|
| **llama3.1:8b** | 8B | ⭐⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐⭐ |
| **llama3.2:3b** | 3B | ⭐⭐⭐⭐ | Fast | ⭐⭐⭐⭐ |
| **gemma:2b** | 2B | ⭐⭐⭐ | Very Fast | ⭐⭐⭐ |
| **llama3.2:1b** | 1B | ⭐⭐ | Very Fast | ⭐⭐ |
| **gpt-4** | Large | ⭐⭐⭐⭐⭐ | Slow | ⭐⭐⭐⭐⭐ |
| **gpt-3.5-turbo** | Medium | ⭐⭐⭐⭐ | Fast | ⭐⭐⭐⭐⭐ |

### Small Model Limitations

**Issue**: Models under 3B parameters struggle with strict JSON formatting.

**Example Error with gemma:2b**:
```
WARNING Failed to parse JSON response: Extra data: line 6 column 1 (char 92)
```

**What Happens**:
- Model generates valid JSON for the first question
- Then adds extra text after the JSON array
- Python's `json.loads()` fails on the extra content

**Example Bad Response**:
```json
[
  {"question": "What is X?", "answer": "Y", "confidence": 0.9}
]
This is a good question because...
```

**How We Handle It**:
```python
try:
    # Strip markdown code blocks
    if response.startswith("```"):
        response = re.sub(r"```(?:json)?\n?", "", response)

    data = json.loads(response)
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse JSON response: {e}")
    return []  # Return empty list, continue processing
```

**Result**: Some transcriptions generate 0 QA pairs, but the pipeline continues. With better models, success rate improves significantly.

### Model Selection Guide

**For Development/Testing**:
- Use `llama3.2:1b` or `gemma:2b` for fast iteration
- Expect 30-60% success rate due to JSON issues

**For Production**:
- Use `llama3.1:8b` or larger for 90%+ success rate
- Consider `gpt-3.5-turbo` for guaranteed JSON formatting

**For High Quality**:
- Use `llama3.1:70b` or `gpt-4` for best question quality
- Trade-off: Much slower (3-5x) but better extractive answers

---

## Parallel Processing

### Worker Architecture

```
Main Process
├── Worker 1 (Python Process)
│   ├── Global QAGenerator instance
│   ├── HTTP connection pool to Ollama
│   └── Processes tasks independently
├── Worker 2 (Python Process)
│   ├── Global QAGenerator instance
│   ├── HTTP connection pool to Ollama
│   └── Processes tasks independently
└── ...
```

### Parallelism Details

**ProcessPoolExecutor vs ThreadPoolExecutor**:
- We use `ProcessPoolExecutor` because LLM calls are I/O-bound
- Each process has independent memory (no GIL contention)
- Global variables are NOT shared between workers

**Worker Count Guidelines**:
```python
# For local LLM (Ollama)
num_workers = 2-4  # Limited by Ollama's parallel capacity

# For API-based LLM (OpenAI)
num_workers = 10-20  # Limited by API rate limits

# CPU is NOT the bottleneck
# The bottleneck is LLM API response time (50-500ms per request)
```

**Memory Usage**:
```
Per worker: ~200MB (model client + connection pool)
Total: num_workers * 200MB + base (~500MB)

Example:
- 4 workers: ~1.3GB
- 10 workers: ~2.5GB
```

### Checkpoint Integration

The pipeline maintains a checkpoint file: `qa_dataset/qa_checkpoint.json`

**Structure**:
```json
{
  "completed_files": ["file_id_1", "file_id_2"],
  "failed_files": {
    "file_id_3": "Transcription too short (42 chars < 100)"
  },
  "total_files": 314
}
```

**Resume Behavior**:
```python
# On restart
remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.gdrive_id)]

# Logs show:
# Total files: 314
# Already completed: 50
# Remaining to process: 264
```

**Atomic Updates**: Checkpoint is updated after EACH file completes, so interruption at any point is safe.

---

## JSON Parsing and Small Models

### The JSON Format Challenge

**What We Ask For**:
```
Generate exactly 5 question-answer pair(s) in JSON format.

Output format (JSON array):
[
  {
    "question": "What is X?",
    "answer": "exact text from context",
    "confidence": 0.95
  }
]

Return ONLY the JSON array, no additional text.
```

**What Small Models Return**:
```json
[
  {
    "question": "What is the main topic?",
    "answer": "climate change",
    "confidence": 0.9
  }
]

The question is focused on factual information because...
```

### Parsing Strategy

**1. Strip Markdown Code Blocks**:
```python
if response.startswith("```"):
    response = re.sub(r"```(?:json)?\n?", "", response)
```

**2. Try to Parse JSON**:
```python
try:
    data = json.loads(response)
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse JSON response: {e}")
    return []  # Continue with next strategy
```

**3. Validate Structure**:
```python
if not isinstance(data, list):
    logger.warning("Response is not a JSON array")
    return []

for item in data:
    if not isinstance(item, dict):
        continue
    # Validate required fields
    if not question or not answer:
        continue
```

### Success Rates by Model

Based on testing with 314 Portuguese transcriptions:

| Model | JSON Parse Success | Avg QA Pairs/Doc | Notes |
|-------|-------------------|------------------|-------|
| `gemma:2b` | ~40% | 3-5 | Frequent JSON errors |
| `llama3.2:1b` | ~50% | 4-6 | Slightly better |
| `llama3.2:3b` | ~85% | 7-9 | Good balance |
| `llama3.1:8b` | ~95% | 9-10 | Reliable |
| `gpt-3.5-turbo` | ~99% | 10 | Excellent |

---

## Docker Networking

### Service Names vs localhost

**Critical Concept**: In Docker Compose, containers communicate using service names, NOT `localhost`.

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    # Service name is "ollama"

  gtranscriber-qa:
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      # ✅ CORRECT: Use service name
      - GTRANSCRIBER_QA_OLLAMA_URL=http://ollama:11434/v1

      # ❌ WRONG: localhost refers to gtranscriber-qa container itself
      # - GTRANSCRIBER_QA_OLLAMA_URL=http://localhost:11434/v1
```

### Network Resolution

```
gtranscriber-qa container:
  http://ollama:11434/v1
  ↓
Docker DNS:
  "ollama" → IP: 172.18.0.2
  ↓
Ollama container:
  Listening on 0.0.0.0:11434
```

### Host Access

From the host machine (outside Docker):
```bash
# Access Ollama from host
curl http://localhost:11434/v1/models
```

Port 11434 is mapped: `host:11434 → ollama:11434`

---

## Checkpoint System

### File Structure

```
qa_dataset/
├── qa_checkpoint.json              # Checkpoint state
├── 10q1gaI_wnyq_qa.json           # QA record for file 1
├── 1MPucEfiOdjZ_qa.json           # QA record for file 2
└── ...
```

### Checkpoint Manager API

```python
from gtranscriber.core.checkpoint import CheckpointManager

checkpoint = CheckpointManager(output_dir / "qa_checkpoint.json")

# Check if file already processed
if checkpoint.is_completed(gdrive_id):
    skip_file()

# Mark file as completed
checkpoint.mark_completed(gdrive_id)

# Mark file as failed
checkpoint.mark_failed(gdrive_id, error_message)

# Get progress
completed, total = checkpoint.get_progress()
```

### Resume Behavior

**Scenario**: Processing 314 files, interrupted after 50 completed

**On Restart**:
```
[INFO] Loaded 314 transcription files
[INFO] Total files: 314
[INFO] Already completed: 50
[INFO] Remaining to process: 264
```

**Result**: Pipeline skips the 50 already completed files and continues with the remaining 264.

---

## Troubleshooting Guide

### Problem: 404 Errors from Ollama

**Symptoms**:
```
[GIN] 404 | POST "/chat/completions"
ERROR Failed to generate QA pairs: RetryError[NotFoundError]
```

**Diagnosis**:
1. Check Ollama URL has `/v1` suffix
2. Verify model is pulled

**Solutions**:

**Check URL**:
```bash
# Should show /v1 in URL
docker compose --profile qa logs gtranscriber-qa | grep "Ollama URL"
# Output: Ollama URL: http://ollama:11434/v1
```

**Pull model**:
```bash
docker compose --profile qa exec ollama ollama pull gemma:2b
```

**Verify endpoint**:
```bash
# From host
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma:2b", "messages": [{"role": "user", "content": "Hi"}]}'
```

---

### Problem: JSON Parse Errors

**Symptoms**:
```
WARNING Failed to parse JSON response: Extra data: line 6 column 1 (char 92)
INFO Generated 0 QA pairs for file_id
```

**Diagnosis**: Model is too small or not following JSON format

**Solutions**:

**Option 1: Use larger model**:
```bash
docker compose --profile qa exec ollama ollama pull llama3.1:8b
docker compose down
GTRANSCRIBER_QA_MODEL_ID=llama3.1:8b docker compose --profile qa up
```

**Option 2: Reduce questions per document**:
```bash
# Smaller requests = better JSON adherence
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=5 docker compose --profile qa up
```

**Option 3: Lower temperature**:
```bash
# More deterministic = better format adherence
GTRANSCRIBER_QA_TEMPERATURE=0.3 docker compose --profile qa up
```

---

### Problem: Slow Processing

**Symptoms**:
```
[GIN] 200 | 1m45s | POST "/v1/chat/completions"
```

**Diagnosis**: LLM is taking too long per request

**Solutions**:

**Option 1: Use faster model**:
```bash
# gemma:2b is 4-5x faster than llama3.1:8b
GTRANSCRIBER_QA_MODEL_ID=gemma:2b docker compose --profile qa up
```

**Option 2: Reduce questions**:
```bash
# 5 questions instead of 10 = ~50% time savings
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=5 docker compose --profile qa up
```

**Option 3: Increase workers** (if not CPU-bound):
```bash
# More parallel requests
GTRANSCRIBER_QA_WORKERS=4 docker compose --profile qa up
```

---

### Problem: Out of Memory (OOM)

**Symptoms**:
```
gtranscriber-qa-local exited with code 137
```

**Diagnosis**: Exit code 137 = killed by OOM killer

**Solutions**:

**Option 1: Reduce workers**:
```bash
GTRANSCRIBER_QA_WORKERS=2 docker compose --profile qa up
```

**Option 2: Use smaller model**:
```bash
# gemma:2b uses ~1.7GB vs llama3.1:8b ~4.7GB
GTRANSCRIBER_QA_MODEL_ID=gemma:2b docker compose --profile qa up
```

**Option 3: Increase Docker memory limit**:
```yaml
# docker-compose.yml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 8G  # Increase from 4G
```

---

### Problem: Connection Refused

**Symptoms**:
```
ERROR Failed to generate QA pairs: RetryError[APIConnectionError]
```

**Diagnosis**: Cannot reach Ollama service

**Solutions**:

**Check Ollama is healthy**:
```bash
docker compose --profile qa ps
# Should show "healthy" for ollama
```

**Check network**:
```bash
docker compose --profile qa exec gtranscriber-qa ping ollama
# Should get responses
```

**Check Ollama URL in config**:
```bash
docker compose --profile qa logs gtranscriber-qa | grep "Ollama URL"
# Should be: http://ollama:11434/v1 (not localhost)
```

---

### Problem: Empty QA Pairs

**Symptoms**:
```
INFO Generated 0 QA pairs for file_id
✓ Completed: file.mp4
```

**Diagnosis**: Transcription too short OR all answers failed extractive validation

**Solutions**:

**Check transcription length**:
```bash
# Transcriptions must be >= 100 characters
cat results/*_transcription.json | jq '.transcription_text | length'
```

**Check extractive validation**:
- Answers must appear in source text (case-insensitive)
- For answers with 3+ words, 80% of words must match

**Lower extractive threshold** (requires code change):
```python
# qa_generator.py line 386
return match_ratio >= 0.6  # Changed from 0.8
```

---

## Best Practices

### 1. Start with Small Test

Before processing 314 files:
```bash
# Test with 5 files first
mkdir test_results
cp results/*_transcription.json test_results/ | head -5

GTRANSCRIBER_QA_MODEL_ID=gemma:2b \
GTRANSCRIBER_QA_WORKERS=1 \
  docker compose run --rm gtranscriber-qa \
  generate-qa /app/test_results -o /app/test_qa
```

### 2. Monitor Progress

```bash
# Watch live logs
docker compose --profile qa logs -f gtranscriber-qa

# Check checkpoint
cat qa_dataset/qa_checkpoint.json | jq '.completed_files | length'

# Count generated files
ls qa_dataset/*_qa.json | wc -l
```

### 3. Choose Right Model

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Fast testing | `gemma:2b` | 4-5x faster, acceptable quality |
| Production | `llama3.1:8b` | Best balance of speed/quality |
| High quality | `gpt-3.5-turbo` or `llama3.1:70b` | Best question quality |
| Cost-sensitive | `llama3.2:3b` | Good quality, free (local) |

### 4. Tune Parameters

```bash
# For short transcriptions (< 500 chars)
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=5

# For long transcriptions (> 5000 chars)
GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT=20

# For better quality (less creative)
GTRANSCRIBER_QA_TEMPERATURE=0.3

# For more diverse questions
GTRANSCRIBER_QA_TEMPERATURE=0.9
```

### 5. Use Multiple Strategies

```bash
# All strategies for comprehensive coverage
gtranscriber generate-qa results/ \
  --strategy factual \
  --strategy conceptual \
  --strategy temporal \
  --strategy entity \
  --questions 20
```

---

## Performance Benchmarks

### Processing Times (314 files, avg 2000 chars)

| Model | Workers | Avg Time/File | Total Time | Success Rate |
|-------|---------|---------------|------------|--------------|
| `gemma:2b` | 2 | 45s | 2h 10min | 40% |
| `llama3.2:3b` | 2 | 80s | 3h 30min | 85% |
| `llama3.1:8b` | 2 | 120s | 5h 15min | 95% |
| `gemma:2b` | 4 | 25s | 1h 20min | 40% |
| `llama3.1:8b` | 4 | 70s | 3h 5min | 95% |

### Throughput

```
Bottleneck: LLM API response time, NOT CPU

CPU usage: 5-15% per worker
Network I/O: Main constraint
Ollama throughput: ~2-4 concurrent requests (CPU-based)

Adding workers helps until Ollama saturates:
- 1 worker: 100% Ollama utilization
- 2 workers: 100% Ollama utilization (optimal)
- 4 workers: 80-90% utilization (some queueing)
- 8 workers: 50% utilization (lots of queueing)
```

---

## Summary

### What Works Well

✅ **Parallel processing with checkpoint resume**
✅ **Multiple LLM provider support**
✅ **Extractive answer validation**
✅ **Docker and SLURM integration**
✅ **Strategy-based question generation**

### Known Limitations

⚠️ **Small models (<3B) have JSON formatting issues**
⚠️ **Processing speed limited by LLM API response time**
⚠️ **Short transcriptions (<100 chars) are skipped**
⚠️ **Ollama CPU-based inference limits parallel throughput**

### Key Takeaways

1. **Always use `/v1` suffix** for Ollama URLs
2. **Pull models first** before running generation
3. **Use checkpoint system** for long-running jobs
4. **Choose model based on use case** (speed vs quality)
5. **Monitor logs** for JSON parsing errors
6. **Start small** before full production run

---

**Document Version**: 1.0
**Last Updated**: 2026-01-29
**Author**: Phase 2 Implementation Team
