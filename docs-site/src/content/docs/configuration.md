---
title: Configuration Reference
description: Source-of-truth environment variables and settings classes used by Arandu.
---

This page mirrors the active `BaseSettings` classes in `src/arandu/**/config.py` and
`src/arandu/shared/config.py`.

## Loading order

Settings precedence is:

1. CLI options
2. Environment variables
3. `.env`
4. Class defaults

## Settings classes

Current settings classes:

- `TranscriberConfig` (`ARANDU_`)
- `QAConfig` (`ARANDU_QA_`)
- `CEPConfig` (`ARANDU_CEP_`)
- `JudgeConfig` (`ARANDU_JUDGE_`)
- `KGConfig` (`ARANDU_KG_`)
- `EvaluationConfig` (`ARANDU_EVAL_`)
- `ResultsConfig` (`ARANDU_RESULTS_`)
- `LLMConfig` (aliases: `OPENAI_API_KEY`, `ARANDU_LLM_BASE_URL`)

## `TranscriberConfig` (`ARANDU_`)

| Field | Env var | Default |
|---|---|---|
| `model_id` | `ARANDU_MODEL_ID` | `openai/whisper-large-v3` |
| `language` | `ARANDU_LANGUAGE` | `None` |
| `return_timestamps` | `ARANDU_RETURN_TIMESTAMPS` | `true` |
| `chunk_length_s` | `ARANDU_CHUNK_LENGTH_S` | `30` |
| `stride_length_s` | `ARANDU_STRIDE_LENGTH_S` | `5` |
| `force_cpu` | `ARANDU_FORCE_CPU` | `false` |
| `quantize` | `ARANDU_QUANTIZE` | `false` |
| `quantize_bits` | `ARANDU_QUANTIZE_BITS` | `8` |
| `credentials` | `ARANDU_CREDENTIALS` | `credentials.json` |
| `token` | `ARANDU_TOKEN` | `token.json` |
| `scopes` | `ARANDU_SCOPES` | `["https://www.googleapis.com/auth/drive"]` |
| `workers` | `ARANDU_WORKERS` | `1` |
| `catalog_file` | `ARANDU_CATALOG_FILE` | `catalog.csv` |
| `input_dir` | `ARANDU_INPUT_DIR` | `./input` |
| `results_dir` | `ARANDU_RESULTS_DIR` | `./results` |
| `credentials_dir` | `ARANDU_CREDENTIALS_DIR` | `./` |
| `hf_cache_dir` | `ARANDU_HF_CACHE_DIR` | `./cache/huggingface` |
| `temp_dir` | `ARANDU_TEMP_DIR` | platform temp dir + `/arandu` |
| `max_retries` | `ARANDU_MAX_RETRIES` | `3` |
| `retry_delay` | `ARANDU_RETRY_DELAY` | `1.0` |

## `QAConfig` (`ARANDU_QA_`)

| Field | Env var | Default |
|---|---|---|
| `provider` | `ARANDU_QA_PROVIDER` | `ollama` |
| `model_id` | `ARANDU_QA_MODEL_ID` | `qwen3:14b` |
| `ollama_url` | `ARANDU_QA_OLLAMA_URL` | `http://localhost:11434/v1` |
| `base_url` | `ARANDU_QA_BASE_URL` | `None` |
| `questions_per_document` | `ARANDU_QA_QUESTIONS_PER_DOCUMENT` | `10` |
| `temperature` | `ARANDU_QA_TEMPERATURE` | `0.7` |
| `max_tokens` | `ARANDU_QA_MAX_TOKENS` | `2048` |
| `output_dir` | `ARANDU_QA_OUTPUT_DIR` | `qa_dataset` |
| `language` | `ARANDU_QA_LANGUAGE` | `pt` |
| `workers` | `ARANDU_QA_WORKERS` | `2` |

## `CEPConfig` (`ARANDU_CEP_`)

| Field | Env var | Default |
|---|---|---|
| `enable_reasoning_traces` | `ARANDU_CEP_ENABLE_REASONING_TRACES` | `true` |
| `bloom_levels` | `ARANDU_CEP_BLOOM_LEVELS` | `remember,understand,analyze,evaluate` |
| `bloom_distribution` | `ARANDU_CEP_BLOOM_DISTRIBUTION` | `{"remember":0.2,"understand":0.3,"analyze":0.3,"evaluate":0.2}` |
| `enable_scaffolding_context` | `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT` | `true` |
| `max_scaffolding_pairs` | `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS` | `10` |
| `max_hop_count` | `ARANDU_CEP_MAX_HOP_COUNT` | `3` |
| `reasoning_max_tokens` | `ARANDU_CEP_REASONING_MAX_TOKENS` | `2048` |
| `validation_threshold` | `ARANDU_CEP_VALIDATION_THRESHOLD` | `0.6` |
| `faithfulness_weight` | `ARANDU_CEP_FAITHFULNESS_WEIGHT` | `0.30` |
| `bloom_calibration_weight` | `ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT` | `0.25` |
| `informativeness_weight` | `ARANDU_CEP_INFORMATIVENESS_WEIGHT` | `0.25` |
| `self_containedness_weight` | `ARANDU_CEP_SELF_CONTAINEDNESS_WEIGHT` | `0.20` |
| `enable_source_metadata_context` | `ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT` | `true` |
| `language` | `ARANDU_CEP_LANGUAGE` | `pt` |

## `JudgeConfig` (`ARANDU_JUDGE_`)

| Field | Env var | Default |
|---|---|---|
| `language` | `ARANDU_JUDGE_LANGUAGE` | `pt` |
| `temperature` | `ARANDU_JUDGE_TEMPERATURE` | `0.3` |
| `max_tokens` | `ARANDU_JUDGE_MAX_TOKENS` | `2048` |
| `validator_model` | `ARANDU_JUDGE_VALIDATOR_MODEL` | `None` |
| `validator_provider` | `ARANDU_JUDGE_VALIDATOR_PROVIDER` | `None` (inferred) |
| `validator_base_url` | `ARANDU_JUDGE_VALIDATOR_BASE_URL` | `None` |

## `KGConfig` (`ARANDU_KG_`)

| Field | Env var | Default |
|---|---|---|
| `backend` | `ARANDU_KG_BACKEND` | `atlas` |
| `backend_options` | `ARANDU_KG_BACKEND_OPTIONS` | `{}` |
| `provider` | `ARANDU_KG_PROVIDER` | `ollama` |
| `model_id` | `ARANDU_KG_MODEL_ID` | `llama3.1:8b` |
| `ollama_url` | `ARANDU_KG_OLLAMA_URL` | `http://localhost:11434/v1` |
| `base_url` | `ARANDU_KG_BASE_URL` | `None` |
| `temperature` | `ARANDU_KG_TEMPERATURE` | `0.5` |
| `language` | `ARANDU_KG_LANGUAGE` | `pt` |
| `output_dir` | `ARANDU_KG_OUTPUT_DIR` | `knowledge_graphs` |

## `EvaluationConfig` (`ARANDU_EVAL_`)

| Field | Env var | Default |
|---|---|---|
| `metrics` | `ARANDU_EVAL_METRICS` | `qa,entity,relation,semantic` |
| `embedding_model` | `ARANDU_EVAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `output_dir` | `ARANDU_EVAL_OUTPUT_DIR` | `evaluation` |
| `qa_dir` | `ARANDU_EVAL_QA_DIR` | `qa_dataset` |
| `kg_dir` | `ARANDU_EVAL_KG_DIR` | `knowledge_graphs` |
| `results_dir` | `ARANDU_EVAL_RESULTS_DIR` | `results` |

## `ResultsConfig` (`ARANDU_RESULTS_`)

| Field | Env var | Default |
|---|---|---|
| `base_dir` | `ARANDU_RESULTS_BASE_DIR` | `./results` |
| `enable_versioning` | `ARANDU_RESULTS_ENABLE_VERSIONING` | `true` |

## `LLMConfig` aliases

| Field | Env var | Default |
|---|---|---|
| `openai_api_key` | `OPENAI_API_KEY` | `None` |
| `base_url` | `ARANDU_LLM_BASE_URL` | `None` |

## Minimal `.env` example

```bash
OPENAI_API_KEY=
ARANDU_LLM_BASE_URL=

ARANDU_MODEL_ID=openai/whisper-large-v3
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_CEP_LANGUAGE=pt
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_KG_PROVIDER=ollama
ARANDU_RESULTS_BASE_DIR=./results
```
