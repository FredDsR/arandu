---
title: CLI Reference
description: Command reference for the Arandu CLI.
---

## Command groups

- Transcription: `transcribe`, `drive-transcribe`, `batch-transcribe`
- Judging: `judge-transcription`, `judge-qa`
- QA/CEP generation: `generate-cep-qa`
- Knowledge graph: `build-kg`
- Run management: `replicate`, `refresh-auth`, `list-runs`, `run-info`, `rebuild-index`, `enrich-metadata`, `info`
- Reporting: `report`, `serve-report`

Global flags:

- `--help`
- `--version`

## Transcription commands

### `arandu transcribe FILE_PATH`

Transcribe one local media file.

Key options: `--model-id/-m`, `--output/-o`, `--quantize/-q`, `--cpu`, `--language/-l`.

### `arandu drive-transcribe FILE_ID`

Transcribe one Google Drive file and upload the result.

Key options: `--model-id/-m`, `--credentials/-c`, `--token/-t`, `--quantize/-q`, `--cpu`,
`--language/-l`.

### `arandu batch-transcribe CATALOG_FILE`

Batch transcription from a catalog CSV.

Key options: `--output-dir/-o`, `--model-id/-m`, `--credentials/-c`, `--token/-t`,
`--workers/-w`, `--checkpoint`, `--quantize/-q`, `--cpu`, `--language/-l`, `--id`.

## Judging commands

Judging runs as a separate step after generation/transcription.

### `arandu judge-transcription INPUT_DIR`

Writes per-record verdicts to each `*_transcription.json` under `validation`
(`JudgePipelineResult`), and `is_valid` is derived from `validation.passed`.

Options:

| Option | Type | Default | Description |
|---|---|---|---|
| `--language/-l` | `str` | `pt` | Expected language for judging |
| `--validator-model` | `str \| None` | `ARANDU_JUDGE_VALIDATOR_MODEL` | Enables LLM criteria when set |
| `--validator-provider` | `str \| None` | inferred | Provider (`openai`, `ollama`, `custom`) |
| `--validator-base-url` | `str \| None` | from env | Base URL for custom/OpenAI-compatible endpoints |
| `--validator-temperature` | `float \| None` | `ARANDU_JUDGE_TEMPERATURE` | LLM sampling temperature |
| `--validator-max-tokens` | `int \| None` | `ARANDU_JUDGE_MAX_TOKENS` | Max response tokens |
| `--rejudge/--resume` | flag | `--resume` | Re-run all records or skip already judged ones |

Examples:

```bash
# Heuristic-only
arandu judge-transcription results/

# Heuristics + LLM criteria
arandu judge-transcription results/ --validator-model qwen3:14b

# Force fresh run
arandu judge-transcription results/ --validator-model qwen3:14b --rejudge
```

### `arandu judge-qa INPUT_DIR`

Judges CEP QA pairs and persists verdicts on each pair in `validation`
(`JudgePipelineResult`).

Options:

| Option | Type | Default | Description |
|---|---|---|---|
| `--model/-m` | `str \| None` | `ARANDU_JUDGE_VALIDATOR_MODEL` | Judge model (required via flag/env) |
| `--provider` | `str \| None` | inferred | Provider (`openai`, `ollama`, `custom`) |
| `--base-url` | `str \| None` | env fallback | Custom/OpenAI-compatible URL |
| `--language/-l` | `str` | `pt` | Prompt language |
| `--files` | `int \| None` | all | Max QA files to sample |
| `--pairs` | `int \| None` | all | Max pairs per file |
| `--rejudge/--resume` | flag | `--resume` | Re-run all sampled pairs or skip judged pairs |

Examples:

```bash
arandu judge-qa cep_dataset/
arandu judge-qa cep_dataset/ --provider ollama --model qwen3:14b
arandu judge-qa cep_dataset/ --files 2 --pairs 3
```

## QA / CEP generation

### `arandu generate-cep-qa INPUT_DIR`

Generates CEP QA pairs from transcription outputs.

Key options: `--output-dir/-o`, `--provider`, `--model-id/-m`, `--workers/-w`,
`--temperature`, `--ollama-url`, `--base-url`, `--language/-l`,
`--bloom-dist`, `--jsonl/--no-jsonl`, `--id`.

Example:

```bash
arandu generate-cep-qa results/ --output-dir cep_dataset/
```

Then run QA judging:

```bash
arandu judge-qa cep_dataset/
```

## Knowledge graph

### `arandu build-kg INPUT_DIR`

Builds graph artifacts from transcription outputs.

Key options: `--output-dir/-o`, `--provider`, `--model-id/-m`, `--backend`, `--language/-l`,
`--temperature`, `--ollama-url`, `--base-url`, `--backend-option`, `--no-concepts`, `--id`.

## Run management and reporting

### Authentication and metadata

- `arandu refresh-auth [--credentials/-c] [--token/-t]`
- `arandu enrich-metadata INPUT_DIR OUTPUT_DIR [--pipeline-id/--id]`

### Run indexing and inspection

- `arandu info`
- `arandu list-runs [--pipeline/-p] [--results-dir/-r]`
- `arandu run-info RUN_ID [--pipeline/-p] [--results-dir/-r]`
- `arandu rebuild-index [--results-dir/-r]`
- `arandu replicate SOURCE_PIPELINE_ID [--id] [--results-dir]`

### Reports

- `arandu report [--run-id/--id] [--output/-o] [--no-png] [--results-dir]`
- `arandu serve-report RESULTS_DIR [--port/-p] [--host] [--no-browser]`

## Common workflow

```bash
# 1) Transcribe
arandu batch-transcribe input/catalog.csv --workers 4 --id project-001

# 2) Judge transcriptions
arandu judge-transcription results/ --validator-model qwen3:14b

# 3) Generate CEP QA
arandu generate-cep-qa results/ --id project-001

# 4) Judge QA
arandu judge-qa qa_dataset/ --model qwen3:14b

# 5) Build KG
arandu build-kg results/ --id project-001
```
