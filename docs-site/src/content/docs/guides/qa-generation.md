---
title: QA Generation
description: Generate CEP QA pairs from transcription outputs and evaluate them with judge-qa.
---

Use `generate-cep-qa` to create Bloom-scaffolded QA pairs from transcription JSON files.
Judging is a separate step with `judge-qa`.

## Quick start

```bash
# 1) Generate CEP QA pairs
arandu generate-cep-qa results/ --output-dir qa_dataset/

# 2) Judge generated pairs
arandu judge-qa qa_dataset/ --model qwen3:14b
```

## Generation options

Common options for `generate-cep-qa`:

- `--output-dir/-o`
- `--provider`
- `--model-id/-m`
- `--workers/-w`
- `--questions`
- `--temperature`
- `--ollama-url`
- `--base-url`
- `--language/-l`
- `--bloom-dist`
- `--jsonl/--no-jsonl`
- `--id`

## Judge options

Common options for `judge-qa`:

- `--model/-m` (required via flag or `ARANDU_JUDGE_VALIDATOR_MODEL`)
- `--provider`
- `--base-url`
- `--language/-l`
- `--files`
- `--pairs`
- `--rejudge/--resume`

## Environment variables

Generation (`ARANDU_QA_*` + `ARANDU_CEP_*`):

- `ARANDU_QA_PROVIDER`, `ARANDU_QA_MODEL_ID`, `ARANDU_QA_OLLAMA_URL`, `ARANDU_QA_BASE_URL`
- `ARANDU_QA_QUESTIONS_PER_DOCUMENT`, `ARANDU_QA_TEMPERATURE`, `ARANDU_QA_MAX_TOKENS`
- `ARANDU_QA_OUTPUT_DIR`, `ARANDU_QA_LANGUAGE`, `ARANDU_QA_WORKERS`
- `ARANDU_CEP_BLOOM_LEVELS`, `ARANDU_CEP_BLOOM_DISTRIBUTION`
- `ARANDU_CEP_ENABLE_REASONING_TRACES`, `ARANDU_CEP_MAX_HOP_COUNT`
- `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT`, `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS`
- `ARANDU_CEP_REASONING_MAX_TOKENS`, `ARANDU_CEP_LANGUAGE`
- `ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT`

Judge (`ARANDU_JUDGE_*`):

- `ARANDU_JUDGE_VALIDATOR_MODEL`
- `ARANDU_JUDGE_VALIDATOR_PROVIDER`
- `ARANDU_JUDGE_VALIDATOR_BASE_URL`
- `ARANDU_JUDGE_TEMPERATURE`
- `ARANDU_JUDGE_MAX_TOKENS`
- `ARANDU_JUDGE_LANGUAGE`

## Output

Generated files are written as `*_cep_qa.json`.
Each QA pair may later receive a `validation` payload (`JudgePipelineResult`) after `judge-qa` runs.

---

See also: [CEP QA Generation](/guides/cep-qa-generation/) · [CLI Reference](/reference/cli/)
