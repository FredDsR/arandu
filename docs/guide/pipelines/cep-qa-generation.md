# CEP QA Generation Guide

Generate cognitively-calibrated question-answer pairs using the **Cognitive Elicitation Pipeline (CEP)** with Bloom's Taxonomy scaffolding. With the scaffolding gate on (default), `generate-cep-qa` judges each pair with LLM-as-a-Judge while generating; the [`judge-qa`](../cli-reference.md#judge-qa) command covers pairs without a clean verdict (or everything with `--rejudge`).

## Overview

The CEP pipeline generates QA pairs with Bloom scaffolding, then the scaffolding gate judges each pair inside `generate-cep-qa` (default on), and `judge-qa` covers pairs without a clean verdict:

1. **Module I: Bloom Scaffolding** - Generates questions distributed across Bloom's taxonomy levels
2. **Module II: Generation Rationale** - Each pair records why it was built from its context, in the same generation call (an inspection aid, never a judge input)
3. **Module III: LLM-as-a-Judge (scaffolding gate during generation, plus the `judge-qa` step for anything left unjudged)** - Validates QA pairs for faithfulness, Bloom calibration, informativeness, and self-containedness

### Key Features

- **Cognitive Scaffolding**: Questions progress from basic recall to advanced analysis
- **Bloom Taxonomy Levels**: remember, understand, apply, analyze, evaluate, create
- **Generation Rationale**: Each pair records why it was built from its context (an inspection aid, never a judge input)
- **Quality Validation** (scaffolding gate in `generate-cep-qa`, plus `judge-qa` for pairs without a clean verdict or with `--rejudge`): LLM-based scoring for faithfulness, Bloom calibration, informativeness, and self-containedness

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
# Generate CEP QA pairs (the gate needs a validator model; use --no-scaffolding-gate to skip it)
arandu generate-cep-qa results/ --output-dir cep_dataset/ --validator-model qwen3:14b

# Validate the generated pairs with LLM-as-a-Judge (covers pairs without a clean verdict; `--rejudge` redoes all)
arandu judge-qa cep_dataset/ --model qwen3:14b

# Export to JSONL format
arandu generate-cep-qa results/ --jsonl --output-dir cep_dataset/ --validator-model qwen3:14b

# With pipeline ID for tracking
arandu generate-cep-qa results/ --id etno-project-001 --validator-model qwen3:14b
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_QA_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `ARANDU_QA_MODEL_ID` | `qwen3:14b` | Model for QA generation |
| `ARANDU_QA_OLLAMA_URL` | `http://ollama:11434/v1` | Ollama API URL |
| `ARANDU_QA_TEMPERATURE` | `0.7` | LLM temperature (0.0-2.0) |
| `ARANDU_QA_WORKERS` | `2` | Parallel workers |

### CEP-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_CEP_LANGUAGE` | `pt` | Prompt language (`pt` or `en`) |
| `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT` | `true` | Pass prior QA pairs to higher Bloom levels |
| `ARANDU_CEP_ENABLE_SCAFFOLDING_GATE` | `true` | Judge each pair before it can scaffold later pairs; needs `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS` | `10` | Max prior QA pairs to include as context |

### Judge Variables (used by the scaffolding gate and `judge-qa`)

The validator client is configured with the `ARANDU_JUDGE_` prefix (shared with `judge-transcription`), both for the scaffolding gate inside `generate-cep-qa` and for `judge-qa`. There is no `ARANDU_CEP_ENABLE_VALIDATION` switch. In resume mode, a file whose pairs were gated by a different validator model is refused (exit 1) instead of mixing models; run `judge-qa` with the gate's validator model or pass `--rejudge`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_JUDGE_VALIDATOR_MODEL` | (none) | Judge model. Required by the scaffolding gate (default on, via this var or `--validator-model`) and by `judge-qa` (via this var or `--model`). The SLURM cep and judge jobs default it to `qwen3:14b` |
| `ARANDU_JUDGE_VALIDATOR_PROVIDER` | inferred | `openai`, `ollama`, or `custom` |
| `ARANDU_JUDGE_VALIDATOR_BASE_URL` | inferred | Validator base URL (falls back to `ARANDU_LLM_BASE_URL`) |

### Scaffolding Gate

With the gate active, each generated pair is judged by the same judge `judge-qa` uses, right after generation and before the next Bloom level is generated. The rules:

- The gate has no effect unless `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT` is on, and it is active only when both switches are on (`--no-scaffolding-gate` turns it off for one run).
- Only approved pairs scaffold later levels. A pair without a verdict (the validator raised) or with a judge error counts as not approved.
- The admission filter runs before the `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS` cap, so the cap is applied to approved pairs only.
- Rejected pairs stay in the output with their verdict and are not regenerated, so the Bloom distribution is preserved.
- The verdict is written at generation time; `judge-qa` resume skips pairs with a clean verdict and re-judges those with a judge error.
- A file whose pairs got no clean verdict at all counts as a failed file. If only some pairs lack one, a warning is logged and the file is kept.
- Each record carries `scaffolding_gate_enabled`, and the run records the gate settings in `run_metadata.json` (see [Schemas](../../development/schemas.md)).
- Resuming a pipeline id generated under the other gate regime is refused. Use `--rebuild` or a new `--id`; ids generated before the gate existed count as gate off.

### Bloom Distribution

The distribution is the absolute number of QA pairs to generate at each
cognitive level, per chunk. The per-chunk ladder size is their sum.

| Level | Default Count | Description |
|-------|---------------|-------------|
| `remember` | 3 | Recall explicit facts (factual base / scaffolding ground) |
| `understand` | 1 | Explain and interpret concepts |
| `analyze` | 1 | Identify relationships and patterns |
| `evaluate` | 1 | Make judgments and justify decisions |

Customize via CLI (integer counts, not weights):
```bash
arandu generate-cep-qa results/ \
  --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1" \
  --validator-model qwen3:14b
```

### Example .env Configuration

```bash
# QA Generation Settings
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_WORKERS=4
ARANDU_QA_OUTPUT_DIR=cep_dataset

# CEP-Specific Settings
ARANDU_CEP_LANGUAGE=pt

# Judge Settings (used by the scaffolding gate and judge-qa)
ARANDU_CEP_ENABLE_SCAFFOLDING_GATE=true
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama

# Directories
ARANDU_RESULTS_DIR=./results
```

## Usage Examples

### Basic CEP Generation

```bash
# Default configuration (Portuguese prompts); pairs are judged by the scaffolding gate
docker compose --profile cep up
```

### Validating Generated Pairs

```bash
# With the gate on, verdicts are already written at generation time. Run judge-qa
# to resume pairs without a clean verdict (it skips the gated ones):
arandu judge-qa cep_dataset/ --model qwen3:14b

# Sample a subset while iterating
arandu judge-qa cep_dataset/ --files 2 --pairs 3
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
arandu generate-cep-qa results/ --jsonl --output-dir cep_dataset/ --validator-model qwen3:14b
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
      "rationale": "O trecho afirma diretamente o que o pescador faz",
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
      "rationale": "O trecho liga a subida do rio à decisão de guardar o barco",
      "reasoning_trace": null,
      "is_multi_hop": false,
      "hop_count": null,
      "tacit_inference": null,
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
{"question": "Por que isso aconteceu?", "answer": "...", "context": "...", "bloom_level": "analyze", "rationale": "..."}
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

Verdicts are written at generation time by the scaffolding gate, or later by [`judge-qa`](../cli-reference.md#judge-qa) command. Each criterion is an independent gate: the pair passes only if **every** criterion evaluated for it clears its own `threshold` (`0.625` for every criterion today, set in `prompts/judge/criteria/<criterion>/config.json`). There is no aggregate score and no criterion weights. Non-remember pairs are scored on all four criteria below; `remember` pairs are scored on **faithfulness** and **bloom calibration** only, since factual recall is neither expected to reveal tacit knowledge nor to be fully self-contained (the other two are omitted, not auto-passed).

### Faithfulness
Is the answer grounded in the provided context?

| Score | Description |
|-------|-------------|
| 1.0 | Answer completely grounded in text |
| 0.8 | Answer well grounded with minimal inferences |
| 0.6 | Answer mostly grounded with some non-trivial inferences |
| 0.4 | Answer partially grounded with significant inferences |
| 0.2 | Answer weakly grounded |
| 0.0 | Answer not grounded, hallucinated, or contradictory |

### Bloom Calibration
Does the question match the proposed cognitive level?

| Score | Description |
|-------|-------------|
| 1.0 | Perfectly calibrated - requires exactly the declared level |
| 0.8 | Well calibrated - requires predominantly the declared level |
| 0.6 | Reasonably calibrated with some overlap |
| 0.4 | Undercalibrated - requires lower cognitive level |
| 0.0 | Completely miscalibrated |

### Informativeness
Does the answer reveal non-obvious or tacit knowledge?

| Score | Description |
|-------|-------------|
| 1.0 | Reveals significant tacit knowledge or practical know-how |
| 0.8 | Reveals useful and non-obvious knowledge |
| 0.6 | Reveals moderately useful contextual information |
| 0.4 | Common but well-articulated information |
| 0.0 | Trivial or obvious information |

### Self-Containedness
Can the question be understood and answered without external context beyond the provided text?

| Score | Description |
|-------|-------------|
| 1.0 | Fully self-contained; no outside context needed |
| 0.6 | Mostly self-contained with minor ambiguity |
| 0.0 | Depends on context not present in the question/answer |

A QA pair is marked invalid (`is_valid = false`) as soon as **one** of the criteria evaluated for it scores below its own gate. A high score on the others does not compensate for it.

## Programmatic Usage

```python
from arandu.qa.schemas import QARecordCEP, QAPairCEP
from arandu.qa.config import CEPConfig, QAConfig, get_cep_config

# Load existing CEP record
record = QARecordCEP.load("cep_dataset/1abc123xyz_cep_qa.json")

# Access QA pairs with Bloom levels
for qa in record.qa_pairs:
    print(f"Level: {qa.bloom_level}")
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
    if qa.rationale:
        print(f"Rationale: {qa.rationale}")
    print()

# Filter by Bloom level
analyze_pairs = [qa for qa in record.qa_pairs if qa.bloom_level == "analyze"]

# Check Bloom distribution
print(f"Distribution: {record.bloom_distribution}")

# Export to JSONL string
jsonl_content = record.to_jsonl()

# Export to JSONL file
record.to_jsonl("output.jsonl")

# Inspect per-pair judge verdicts (set by the gate at generation or by judge-qa)
for qa in record.qa_pairs:
    print(f"{qa.bloom_level}: is_valid={qa.is_valid}")

# Get CEP configuration
cep_config = get_cep_config()
print(f"Bloom distribution: {cep_config.bloom_distribution}")
print(f"Pairs per chunk: {cep_config.pairs_per_chunk}")
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
1. Generate with the scaffolding gate on, then run `judge-qa` for pairs without a clean verdict
2. During development, sample with `judge-qa --files N --pairs M` for faster iteration
3. To move the bar, edit the per-criterion `threshold` in `prompts/judge/criteria/<criterion>/config.json`; there is no environment variable for it
4. Re-run with `judge-qa --rejudge` after changing the validator model or a gate

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

- Lower the per-criterion `threshold` in `prompts/judge/criteria/<criterion>/config.json`: default is 0.625, and the next anchor down is 0.5
- Check validator model capability
- Review prompt language matches content language

### Gate Fails With "no validator model is set"

The gate is on by default and needs a validator. Set `ARANDU_JUDGE_VALIDATOR_MODEL` (or pass `--validator-model`), or disable the gate with `ARANDU_CEP_ENABLE_SCAFFOLDING_GATE=false` (`--no-scaffolding-gate`). The command exits with code 1; it also exits 1 with "Invalid scaffolding gate validator" when the validator provider or base URL cannot build a client.

### Files Fail With "no clean verdict"

Every pair of a file came back unjudged or with a judge error, usually because the validator endpoint is down. Check the validator model and `ARANDU_JUDGE_VALIDATOR_BASE_URL`, then re-run.

### Resume Refused After Changing the Gate

The pipeline id holds outputs written with the gate on (or off) and this run requests the other regime. Re-run with `--rebuild` or a new `--id`.

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

**See also**: [Configuration](../configuration.md) | [SLURM Deployment](../../operations/slurm.md)
