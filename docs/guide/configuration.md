# Configuration Reference

Complete reference for all configuration settings in the Arandu pipeline.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [TranscriberConfig](#transcriberconfig)
3. [QAConfig](#qaconfig)
4. [CEPConfig](#cepconfig)
5. [JudgeConfig](#judgeconfig)
6. [EmicJudgeSettings](#emicjudgesettings)
7. [KGConfig](#kgconfig)
8. [LLMConfig](#llmconfig)
9. [ResultsConfig](#resultsconfig)
10. [Environment Variables](#environment-variables)
11. [Configuration Examples](#configuration-examples)

---

## Configuration System Overview

The Arandu project uses **Pydantic Settings** for configuration management with hierarchical loading:

1. **Command-line arguments** (highest priority)
2. **Environment variables** with config-specific prefixes
3. **`.env` file** in project root
4. **Default values** in each config class (lowest priority)

**Configuration modules**: there is no flat `config.py`. Each config class lives next to the domain it configures:

| Config Class | Module | Prefix |
|--------------|--------|--------|
| `TranscriberConfig` | `arandu.transcription.config` | `ARANDU_` |
| `QAConfig` | `arandu.qa.config` | `ARANDU_QA_` |
| `CEPConfig` | `arandu.qa.config` | `ARANDU_CEP_` |
| `JudgeConfig` | `arandu.qa.config` | `ARANDU_JUDGE_` |
| `EmicJudgeSettings` | `arandu.shared.emic.settings` | `ARANDU_EMIC_JUDGE_` |
| `KGConfig` | `arandu.kg.config` | `ARANDU_KG_` |
| `LLMConfig` | `arandu.shared.config` | (no prefix; uses the alias `ARANDU_LLM_BASE_URL`) |
| `ResultsConfig` | `arandu.shared.config` | `ARANDU_RESULTS_` |

**Usage**:
```python
from arandu.transcription.config import TranscriberConfig
from arandu.qa.config import QAConfig

transcriber_config = TranscriberConfig()
qa_config = QAConfig()
print(transcriber_config.model_id)  # openai/whisper-large-v3
print(qa_config.provider)  # ollama
```

---

## TranscriberConfig

Configuration settings for the transcription pipeline.

**Module**: `arandu.transcription.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_`

### Model Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_id` | `str` | `"openai/whisper-large-v3"` | Hugging Face model ID for Whisper transcription |
| `language` | `str \| None` | `None` | Language code (e.g., 'pt'). If None, auto-detect |
| `return_timestamps` | `bool` | `True` | Return timestamps for transcription segments |
| `chunk_length_s` | `int` | `30` | Audio chunk length in seconds |
| `stride_length_s` | `int` | `5` | Stride length in seconds between chunks |

### Hardware Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `force_cpu` | `bool` | `False` | Force CPU execution instead of GPU |
| `quantize` | `bool` | `False` | Enable 8-bit quantization to reduce VRAM |
| `quantize_bits` | `int` | `8` | Number of bits for quantization |

### Google Drive Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `credentials` | `str` | `"credentials.json"` | Path to Google OAuth2 credentials file |
| `token` | `str` | `"token.json"` | Path to Google OAuth2 token file |
| `scopes` | `list[str]` | `["https://www.googleapis.com/auth/drive"]` | OAuth2 scopes for Google Drive API |

**Note**: `credentials_file` and `token_file` are backward-compatible property aliases for `credentials` and `token`.

### Batch Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `workers` | `int` | `1` | Number of parallel workers for batch processing |
| `catalog_file` | `str` | `"catalog.csv"` | Name of the catalog CSV file |

### Path Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `input_dir` | `str` | `"./input"` | Directory containing input files |
| `results_dir` | `str` | `"./results"` | Directory for transcription results |
| `credentials_dir` | `str` | `"./"` | Directory containing credentials and token files |
| `hf_cache_dir` | `str` | `"./cache/huggingface"` | Hugging Face cache directory for model storage |

### Processing Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `temp_dir` | `str` | `"<tmp>/arandu"` (platform-specific) | Temporary directory for file processing |
| `max_retries` | `int` | `3` | Maximum number of retry attempts for failed operations |
| `retry_delay` | `float` | `1.0` | Delay in seconds between retry attempts |

**Example Configuration**:
```python
from arandu.transcription.config import TranscriberConfig

config = TranscriberConfig()
# Or with custom settings:
config = TranscriberConfig(
    model_id="openai/whisper-large-v3",
    force_cpu=False,
    workers=4,
)
```

---

## QAConfig

Configuration settings for the QA generation pipeline.

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_QA_`

### LLM Provider Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "openai", "ollama", "custom" |
| `model_id` | `str` | `"qwen3:14b"` | Model ID for QA generation |
| `ollama_url` | `str` | `"http://localhost:11434/v1"` | Ollama API base URL |
| `base_url` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints |
| `api_key` | `SecretStr \| None` | `None` | This client's API key, required unless the provider is ollama (no shared fallback). Recorded masked in run snapshots |
| `extra_body` | `dict \| None` | `None` | Provider-specific fields merged into every request body, set as JSON. See [Cloud providers and thinking models](#cloud-providers-and-thinking-models) |

### Generation Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `temperature` | `float` | `0.7` | Temperature for QA generation LLM (range: 0.0-2.0) |
| `max_tokens` | `int` | `8192` | Max tokens for QA generation LLM. Sized for thinking models (Qwen3, Gemini 2.5) whose reasoning tokens consume the budget before the JSON output |

> **Note**: The per-chunk ladder size is not a QAConfig knob. It is the sum of `CEPConfig.bloom_distribution` (the integer pair counts per Bloom level), so the distribution is the single source of truth.

### Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | `Path` | `Path("qa_dataset")` | Output directory for QA datasets |

### Language and Processing

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language code for QA generation prompts (ISO 639-1: 'en' or 'pt') |
| `workers` | `int` | `2` | Number of parallel workers for QA generation |

**Example Configuration**:
```python
from arandu.qa.config import QAConfig

config = QAConfig(
    provider="ollama",
    model_id="qwen3:14b",
    language="pt",
)
```

---

## CEPConfig

Configuration settings for the CEP (Cognitive Elicitation Pipeline) with Bloom's Taxonomy scaffolding. With the scaffolding gate on (default), `generate-cep-qa` judges each pair while generating; `judge-qa` covers pairs without a clean verdict (or everything with `--rejudge`).

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_CEP_`

### Module Toggles

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_scaffolding_context` | `bool` | `True` | Pass previously generated QA pairs as context to higher Bloom levels |
| `enable_scaffolding_gate` | `bool` | `True` | Judge each pair before it can scaffold later pairs (needs a validator model). No effect without `enable_scaffolding_context`; rules in [Scaffolding gate](pipelines/cep-qa-generation.md#scaffolding-gate) |
| `enable_source_metadata_context` | `bool` | `True` | Include extracted source metadata (participant name, location, date) in CEP prompt context |

### Module I - Bloom Scaffolding Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `bloom_distribution` | `dict[str, int]` | `{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}` | Absolute number of QA pairs to generate at each Bloom level, per chunk (single source of truth). Counts must be non-negative; the total must be between 1 and `MAX_BLOOM_PAIRS_PER_CHUNK` (50). Locked at 3/1/1/1 for the thesis run. |
| `max_scaffolding_pairs` | `int` | `10` | Max prior QA pairs to include as scaffolding context (min: 1, max: 50); applies after the gate filter |

**Valid Bloom Levels**: `remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`

**Derived property**: `pairs_per_chunk` returns the sum of `bloom_distribution` values (the per-chunk ladder size). There is no `bloom_levels` field; the keys of `bloom_distribution` are the levels to generate.

### Judge Scoring Settings

`CEPConfig` has none. `judge-qa` decides with the four-criterion LLM-as-a-Judge, and a pair passes only when **every** criterion passes its own gate (`JudgeStepResult.passed` is an `all()` over the criterion scores). Each gate is the `threshold` in `prompts/judge/criteria/<criterion>/config.json`. There is no aggregate score, no criterion weights, and no environment variable that moves the cut.

The validator *client* settings (model, provider, base URL) live in [`JudgeConfig`](#judgeconfig).

### Language Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language for CEP prompts (ISO 639-1: 'pt' or 'en') |

**Example Configuration**:
```python
from arandu.qa.config import CEPConfig

config = CEPConfig(
    bloom_distribution={"remember": 3, "understand": 2, "analyze": 1},
    enable_scaffolding_context=True,
)
print(config.pairs_per_chunk)  # 6
```

---

## JudgeConfig

Configuration for the composable judge pipeline. Supplies the validator LLM client settings shared by the `judge-transcription` LLM filter stage and the `judge-qa` command. Replaces the removed `TranscriptionQualityConfig` (transcription validation is now a filter pipeline, not a weighted-score config; see the [Transcription Validation guide](pipelines/transcription-validation.md)).

**Module**: `arandu.qa.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_JUDGE_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language for judge criterion prompts (ISO 639-1: 'pt' or 'en') |
| `temperature` | `float` | `0.1` | Temperature for judge LLM (low for consistent evaluation, range: 0.0-1.0) |
| `max_tokens` | `int` | `8192` | Max tokens for judge responses (min: 128, max: 8192) |
| `validator_model` | `str \| None` | `None` | Model ID enabling the LLM stage (e.g. `qwen3:14b`, `gemini-2.5-flash`). When unset, `judge-transcription` runs heuristic-only and skips the LLM stage. Also feeds the CEP scaffolding gate, where it is required while the gate is active |
| `validator_provider` | `str \| None` | `None` | Provider: "openai", "ollama", or "custom". Inferred from `ARANDU_LLM_BASE_URL` (custom when set, else ollama) when unspecified |
| `validator_base_url` | `str \| None` | `None` | Base URL for the validator provider. Inherits `ARANDU_LLM_BASE_URL` only when the resolved provider is "custom" |
| `validator_api_key` | `SecretStr \| None` | `None` | This client's API key, required unless the provider is ollama (no shared fallback). Recorded masked in run snapshots |
| `validator_extra_body` | `dict \| None` | `None` | Provider-specific fields merged into every request body, set as JSON. See [Cloud providers and thinking models](#cloud-providers-and-thinking-models). Recorded in the `judge-qa` run snapshot |

**Example Configuration**:
```python
from arandu.qa.config import JudgeConfig

config = JudgeConfig()
# Loaded from ARANDU_JUDGE_* env vars; e.g.
# ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
# ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
```

---

## EmicJudgeSettings

Configuration for `arandu emic-judge`, the Phase D ordinal emic-validity judge
(spec §5). A thin subclass of the shared `LLMSettings`, so it inherits the
canonical connection + sampling fields and overrides only the two defaults the
emic judgment deliberately changes.

**Module**: `arandu.shared.emic.settings` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_EMIC_JUDGE_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "ollama", "openai", or "custom" |
| `model_id` | `str` | `"qwen3:14b"` | Model identifier. **Methodological parameter**: see the note below |
| `base_url` | `str \| None` | `None` | Base URL override; required when `provider == "custom"` |
| `api_key` | `SecretStr \| None` | `None` | This client's API key, required unless the provider is ollama (no shared fallback). Recorded masked in run snapshots |
| `extra_body` | `dict \| None` | `None` | Provider-specific fields merged into every request body, set as JSON. See [Cloud providers and thinking models](#cloud-providers-and-thinking-models). Every `LLMSettings` stage (answerer, judge-answers, non-answerable) has the same field |
| `temperature` | `float` | `0.1` | Lower than the other judges: the emic judgment is structural, not creative (spec §4.2 principle 8) |
| `max_tokens` | `int` | `8192` | Response budget. qwen3 burns thinking tokens against it |
| `language` | `"pt"` | `"pt"` | Narrowed to `pt`, the only `emic_validity` prompt template that ships today |
| `workers` | `int` | `1` | Client-side concurrent LLM requests, honoured via `map_concurrent` |

**The model is not a budget knob.** The scores this stage produces are the
study's measurement of emic validity; the human annotation round (spec §6)
reports agreement with them rather than replacing them. Changing
`ARANDU_EMIC_JUDGE_MODEL_ID` changes the instrument under test, so a run feeding
the agreement study must pin the model the dissertation describes.

**Scope.** `--scope all` (default) scores every CEP pair and records its
`judge-qa` verdict on each score, which is what allows emic validity to be
cross-tabulated against approval. `--scope approved` scores only
canonically-approved pairs. The stratified human-comparison sample keeps its
frame on the approved corpus either way.

**Concurrency.** `workers` must be paired with server-side slots
(`OLLAMA_NUM_PARALLEL`) and the per-slot context VRAM budget;
`scripts/slurm/emic/tupi.slurm` derives the slots from the worker count so the
two cannot drift. Note that a rate-limited call becomes a null score rather than
being backed off and retried (`JudgeCriterion.evaluate` converts failures into
error scores before the batch-level throttle can see them), so raise `workers`
carefully against a metered provider.

**Example Configuration**:
```bash
# Loaded from ARANDU_EMIC_JUDGE_* env vars; e.g.
ARANDU_EMIC_JUDGE_MODEL_ID=qwen3:14b
ARANDU_EMIC_JUDGE_PROVIDER=ollama
ARANDU_EMIC_JUDGE_WORKERS=4
```

---

## KGConfig

Configuration settings for the knowledge graph construction pipeline.

**Module**: `arandu.kg.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_KG_`

### Backend Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `str` | `"atlas"` | KGC backend: `"atlas"` (AutoSchemaKG). Must match pattern `^(atlas)$` |
| `backend_options` | `dict` | `{}` | Backend-specific options passed through to the constructor (e.g., `chunk_size`, `batch_size_triple`, `max_workers`) |

### LLM Provider Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | `str` | `"ollama"` | LLM provider: "openai", "ollama", "custom" |
| `model_id` | `str` | `"llama3.1:8b"` | Model ID for KG construction |
| `ollama_url` | `str` | `"http://localhost:11434/v1"` | Ollama API base URL for KG construction |
| `base_url` | `str \| None` | `None` | Custom base URL for OpenAI-compatible endpoints |
| `api_key` | `SecretStr \| None` | `None` | This client's API key, required unless the provider is ollama (no shared fallback). Recorded masked in run snapshots |
| `extra_body` | `dict \| None` | `None` | Extra request fields, as JSON. atlas-rag forwards only `reasoning_effort`, so any other key is rejected at load time. `ARANDU_ATLAS_RAG_EXTRA_BODY` follows the same rule for the retrieve arm |

### LLM Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `temperature` | `float` | `0.5` | Temperature for KG construction LLM (lower = more consistent, range: 0.0-2.0) |

### Language and Prompts

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `language` | `str` | `"pt"` | Language code for extraction prompts (ISO 639-1): `"pt"`, `"en"` |

Prompts are stored in `prompts/kg/atlas/` using language-keyed JSON files. The atlas backend loads the appropriate language at runtime.

### Output Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | `Path` | `Path("knowledge_graphs")` | Output directory for knowledge graphs |

**Example Configuration**:
```python
from arandu.kg.config import KGConfig

config = KGConfig(
    backend="atlas",
    provider="ollama",
    model_id="qwen3:14b",
    language="pt",
    temperature=0.5,
    backend_options={"chunk_size": 4096, "max_workers": 4},
)
```

**Atlas Backend Options** (passed via `backend_options`):

| Option | Default | Description |
|--------|---------|-------------|
| `batch_size_triple` | `3` | Batch size for triple extraction |
| `batch_size_concept` | `16` | Batch size for concept generation |
| `chunk_size` | `8192` | Characters per text chunk |
| `max_new_tokens` | `2048` | Max tokens for LLM generation |
| `include_concept` | `true` | Whether to run concept generation |
| `max_workers` | `3` | Thread pool size for API calls |

---

## LLMConfig

Shared LLM settings across pipelines. It holds no API key: each LLM client reads its own
`ARANDU_<STAGE>_API_KEY` (see [Cloud providers and thinking models](#cloud-providers-and-thinking-models)).

**Module**: `arandu.shared.config` &nbsp;|&nbsp; **Environment Prefix**: None (uses field aliases)

| Setting | Type | Default | Alias | Description |
|---------|------|---------|-------|-------------|
| `base_url` | `str \| None` | `None` | `ARANDU_LLM_BASE_URL` | Custom base URL for OpenAI-compatible endpoints |

**Example Configuration**:
```python
from arandu.shared.config import LLMConfig

config = LLMConfig()
# Loaded from the ARANDU_LLM_BASE_URL env var
```

**Environment Variables**:
```bash
export ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

---

## ResultsConfig

Configuration for versioned results management.

**Module**: `arandu.shared.config` &nbsp;|&nbsp; **Environment Prefix**: `ARANDU_RESULTS_`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_dir` | `Path` | `Path("./results")` | Base directory for versioned results |
| `enable_versioning` | `bool` | `True` | Enable versioned result directories |

**Example Configuration**:
```python
from pathlib import Path

from arandu.shared.config import ResultsConfig

config = ResultsConfig(
    base_dir=Path("/data/results"),
    enable_versioning=True,
)
```

---

## Environment Variables

Configuration settings are loaded from environment variables with config-specific prefixes.

### Prefix Reference

| Config Class | Prefix | Example |
|--------------|--------|---------|
| `TranscriberConfig` | `ARANDU_` | `ARANDU_MODEL_ID` |
| `QAConfig` | `ARANDU_QA_` | `ARANDU_QA_PROVIDER` |
| `CEPConfig` | `ARANDU_CEP_` | `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS` |
| `JudgeConfig` | `ARANDU_JUDGE_` | `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `KGConfig` | `ARANDU_KG_` | `ARANDU_KG_PROVIDER` |
| `LLMConfig` | (No prefix) | `ARANDU_LLM_BASE_URL` |
| `ResultsConfig` | `ARANDU_RESULTS_` | `ARANDU_RESULTS_BASE_DIR` |

### Format

```bash
<PREFIX><SETTING_NAME>=<value>
```

### Examples by Config Class

**TranscriberConfig** (`ARANDU_`):
```bash
export ARANDU_MODEL_ID=openai/whisper-large-v3
export ARANDU_FORCE_CPU=false
export ARANDU_WORKERS=4
export ARANDU_RETRY_DELAY=1.0
```

**QAConfig** (`ARANDU_QA_`):
```bash
export ARANDU_QA_PROVIDER=openai
export ARANDU_QA_MODEL_ID=gpt-4o-mini
export ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
export ARANDU_QA_LANGUAGE=pt
```

**CEPConfig** (`ARANDU_CEP_`):
```bash
export ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=15
export ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT=true
# Bloom distribution is a dict; set it as JSON
export ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'
```

**JudgeConfig** (`ARANDU_JUDGE_`):
```bash
export ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
export ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
export ARANDU_JUDGE_TEMPERATURE=0.1
```

**KGConfig** (`ARANDU_KG_`):
```bash
export ARANDU_KG_PROVIDER=openai
export ARANDU_KG_MODEL_ID=gpt-4o
export ARANDU_KG_LANGUAGE=pt
export ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
```

**LLMConfig** (No prefix, uses aliases):
```bash
export ARANDU_LLM_BASE_URL=https://my-custom-endpoint/v1
```

**ResultsConfig** (`ARANDU_RESULTS_`):
```bash
export ARANDU_RESULTS_BASE_DIR=/data/results
export ARANDU_RESULTS_ENABLE_VERSIONING=true
```

### API Keys

**Sensitive values should be set as environment variables only** (never commit to git).
Each LLM client has its own key and there is no shared fallback (`OPENAI_API_KEY` and
`GEMINI_API_KEY` are not read by any LLM client):

```bash
export ARANDU_QA_API_KEY=sk-...                # CEP generator
export ARANDU_JUDGE_VALIDATOR_API_KEY=sk-...   # judge-transcription, judge-qa, CEP gate
export ARANDU_KG_API_KEY=sk-...                # build-kg
export ARANDU_ANSWERER_API_KEY=sk-...          # answer
export ARANDU_JUDGE_ANSWERS_API_KEY=sk-...     # judge-answers
export ARANDU_NONANSWERABLE_API_KEY=sk-...     # generate-non-answerable
export ARANDU_EMIC_JUDGE_API_KEY=sk-...        # emic-judge
export ARANDU_ATLAS_RAG_API_KEY=sk-...         # retrieve, atlas_rag arm
```

Only clients whose provider is not `ollama` need a key. The Gemini embedder is not an
LLM client and keeps its own setting (`ARANDU_EMBEDDER_API_KEY_ENV`, default
`GEMINI_API_KEY`).

**Note**: The `.env` file should be added to `.gitignore`.

### Cloud providers and thinking models

Any OpenAI-compatible API works with `provider=custom` and its base URL. Each
LLM client has its own key variable, `ARANDU_<STAGE>_API_KEY` (for example
`ARANDU_QA_API_KEY` for the generator and `ARANDU_JUDGE_VALIDATOR_API_KEY` for
the validator), with no shared fallback, so a key only reaches the provider it
was set for. A run can therefore generate with one provider and judge with
another. Embeddings are separate
(`ARANDU_EMBEDDER_*`): DeepSeek and Maritaca have no embedding API, so keep the
embedder on Gemini or `sentence_transformers`.

| Provider | `BASE_URL` | Example `MODEL_ID` | Thinking off (`EXTRA_BODY`) |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-flash`, `deepseek-v4-pro` | `{"reasoning_effort": "none"}` |
| Maritaca | `https://chat.maritaca.ai/api` | `sabia-4`, `sabiazinho-4` | not needed (non-thinking models) |
| Qwen (DashScope intl.) | `https://<workspace>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `qwen3.8-flash` | `{"enable_thinking": false}` |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-2603` | `{"reasoning_effort": "none"}` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` | `{"reasoning_effort": "none"}` |

Thinking models ignore `temperature`, so a judge configured for low-temperature
scoring needs thinking off, and the reasoning tokens are billed as output and
count against `max_tokens`. atlas-rag (`build-kg`, the `atlas_rag` retrieve
arm) forwards only `reasoning_effort`; use a provider that honours it there.
DeepSeek accepts only `json_object` structured output: the client tries
`json_schema` once, gets a 400, and falls back to `json_object` for the rest of
the run. Maritaca documents structured output only for its Responses API, so
check JSON mode on Chat Completions with a short smoke run first. Verify base
URLs and model ids against the provider's docs before a run; they change.

Generation on DeepSeek, judging on Gemini (exported in the shell or the
SLURM job):

```bash
export ARANDU_QA_PROVIDER=custom
export ARANDU_QA_BASE_URL=https://api.deepseek.com
export ARANDU_QA_MODEL_ID=deepseek-flash
export ARANDU_QA_API_KEY=sk-deepseek-...
export ARANDU_QA_EXTRA_BODY='{"reasoning_effort": "none"}'

export ARANDU_JUDGE_VALIDATOR_PROVIDER=custom
export ARANDU_JUDGE_VALIDATOR_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export ARANDU_JUDGE_VALIDATOR_MODEL=gemini-2.5-flash
export ARANDU_JUDGE_VALIDATOR_API_KEY=gemini-key-...

export ARANDU_KG_PROVIDER=custom
export ARANDU_KG_BASE_URL=https://api.deepseek.com
export ARANDU_KG_MODEL_ID=deepseek-flash
export ARANDU_KG_API_KEY=sk-deepseek-...
export ARANDU_KG_EXTRA_BODY='{"reasoning_effort": "none"}'

# Same five variables (PROVIDER, BASE_URL, MODEL_ID, API_KEY, EXTRA_BODY) for
# ARANDU_ANSWERER_, ARANDU_JUDGE_ANSWERS_, ARANDU_NONANSWERABLE_,
# ARANDU_EMIC_JUDGE_ and ARANDU_ATLAS_RAG_.
```

Export the keys rather than relying on `.env`: `QAConfig`, `JudgeConfig` and
`KGConfig` read `.env`, but the `LLMSettings` stages (answer, judge-answers,
non-answerable, emic-judge) read only the process environment. The commands fail before any LLM call when a cloud
client has no key, naming the variable to set. On the cluster the worker
container receives every `ARANDU_*` variable (`scripts/slurm/container_lib.sh`);
`docker-compose.yml` forwards each service's key variables by name. Keys are
`SecretStr` fields, so run snapshots (`run_metadata.json`) record them masked,
and the environment snapshot redacts any variable whose name contains `KEY`.

Price the run first with `uv run python scripts/estimate_pipeline_cost.py --id <run-id>
--providers deepseek-flash,sabia-4` (add `--reasoning-tokens N` when thinking stays on).

---

## Configuration Examples

### Example 1: Local Development with Ollama

**.env file**:
```bash
# Transcription
ARANDU_MODEL_ID=openai/whisper-large-v3
ARANDU_WORKERS=2
ARANDU_RETRY_DELAY=1.0

# QA Generation
ARANDU_QA_PROVIDER=ollama
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
ARANDU_QA_LANGUAGE=pt

# CEP (Cognitive Elicitation Pipeline)
ARANDU_CEP_LANGUAGE=pt

# Judge (validator client for judge-qa / judge-transcription)
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama

# KG Construction
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
ARANDU_KG_LANGUAGE=pt
```

**CLI Usage**:
```bash
# QA generation (uses .env settings; set ARANDU_JUDGE_VALIDATOR_MODEL or pass --validator-model)
arandu generate-cep-qa results/

# Judge pairs without a clean verdict (--rejudge redoes all)
arandu judge-qa qa_dataset/

# Override specific settings
arandu generate-cep-qa results/ --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1" --provider ollama --validator-model qwen3:14b
```

### Example 2: Production with OpenAI API

**.env file**:
```bash
# QA Generation with OpenAI (each client has its own key)
ARANDU_QA_PROVIDER=openai
ARANDU_QA_API_KEY=sk-your-key-here
ARANDU_QA_MODEL_ID=gpt-4o-mini
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_LANGUAGE=en

# Judge with OpenAI
ARANDU_JUDGE_VALIDATOR_PROVIDER=openai
ARANDU_JUDGE_VALIDATOR_API_KEY=sk-your-key-here
ARANDU_JUDGE_VALIDATOR_MODEL=gpt-4o-mini

# KG Construction with OpenAI
ARANDU_KG_PROVIDER=openai
ARANDU_KG_API_KEY=sk-your-key-here
ARANDU_KG_MODEL_ID=gpt-4o
ARANDU_KG_TEMPERATURE=0.5

# Results versioning
ARANDU_RESULTS_BASE_DIR=/data/transcriptions
ARANDU_RESULTS_ENABLE_VERSIONING=true

# Output directories
ARANDU_QA_OUTPUT_DIR=/data/qa_dataset
ARANDU_KG_OUTPUT_DIR=/data/knowledge_graphs
```

### Example 3: Hybrid Approach (OpenAI + Ollama)

**.env file**:
```bash
# QA with OpenAI (higher quality)
ARANDU_QA_PROVIDER=openai
ARANDU_QA_API_KEY=sk-your-key-here
ARANDU_QA_MODEL_ID=gpt-4o

# KG with Ollama (cost-effective)
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1

# Workers
ARANDU_WORKERS=4
```

### Example 4: SLURM Cluster Configuration

**SLURM script** (use the partition script under `scripts/slurm/cep/`):
```bash
#!/bin/bash
#SBATCH --job-name=arandu-cep
#SBATCH --partition=tupi
#SBATCH --cpus-per-task=16

# Set configuration via environment
export ARANDU_QA_PROVIDER=ollama
export ARANDU_QA_MODEL_ID=qwen3:14b
export ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
export ARANDU_QA_WORKERS=8

# Use $SCRATCH for I/O
export ARANDU_RESULTS_BASE_DIR=$SCRATCH/results
export ARANDU_QA_OUTPUT_DIR=$SCRATCH/qa_dataset

# Run via Docker (the cep_common.sh helper wraps this)
docker compose --profile cep up arandu-cep --abort-on-container-exit
```

### Example 5: Docker Compose Environment

**docker-compose.override.yml** (local overrides):
```yaml
services:
  arandu-cep:
    environment:
      - ARANDU_QA_PROVIDER=ollama
      - ARANDU_QA_MODEL_ID=qwen3:14b
      - ARANDU_QA_OLLAMA_URL=http://host.docker.internal:11434/v1
      - ARANDU_QA_WORKERS=4
    volumes:
      - ./results:/app/results:ro
      - ./qa_dataset:/app/qa_dataset:rw

  arandu-kg:
    environment:
      - ARANDU_KG_PROVIDER=ollama
      - ARANDU_KG_MODEL_ID=llama3.1:8b
      - ARANDU_KG_OLLAMA_URL=http://host.docker.internal:11434/v1
    volumes:
      - ./results:/app/results:ro
      - ./knowledge_graphs:/app/knowledge_graphs:rw
```

### Example 6: CEP with Advanced Bloom Scaffolding

**.env file**:
```bash
# CEP Settings
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_ENABLE_SCAFFOLDING_GATE=true
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=15

# Bloom distribution (integer pair counts per level; single source of truth)
ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'

# Judge gates are per criterion and live in
# prompts/judge/criteria/<criterion>/config.json, not in the environment.

# Validator client
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_TEMPERATURE=0.1
```

---

## Configuration Validation

The configuration system includes validation rules enforced by Pydantic.

### Custom Field Validation

```python
# From CEPConfig — counts must be non-negative, reference valid Bloom levels,
# and total between 1 and MAX_BLOOM_PAIRS_PER_CHUNK (50)
@field_validator("bloom_distribution")
@classmethod
def validate_bloom_distribution(cls, v: dict[str, int]) -> dict[str, int]:
    ...  # each count >= 0, valid level, 1 <= sum <= 50
```

### Pattern Validation

```python
# From KGConfig
backend: str = Field(
    default="atlas",
    pattern="^(atlas)$",  # only the atlas backend is supported
)
```

### Model Validation (Cross-Field)

```python
# From AnswererSettings — three independently overridable token budgets
# must still leave room for the retrieved passages
@model_validator(mode="after")
def _check_packing_budget(self) -> Self:
    budget = self.max_context_tokens - self.prompt_overhead_tokens - self.max_tokens
    if budget <= 0:
        raise ValueError(...)
    return self
```

---

## Configuration Loading Order

Pydantic Settings loads configuration in the following priority order (highest to lowest):

1. **Command-line arguments** (highest priority):
   ```bash
   arandu generate-cep-qa results/ --provider ollama --bloom-dist "remember:3,understand:1,analyze:1,evaluate:1" --validator-model qwen3:14b
   ```

2. **Environment variables**:
   ```bash
   export ARANDU_QA_PROVIDER=openai
   export ARANDU_QA_MODEL_ID=gpt-4o-mini
   ```

3. **`.env` file** in project root:
   ```bash
   ARANDU_QA_PROVIDER=ollama
   ARANDU_QA_MODEL_ID=qwen3:14b
   ```

4. **Default values** in each config class (lowest priority):
   ```python
   provider: str = Field(default="ollama")
   model_id: str = Field(default="qwen3:14b")
   ```

**Result**: Command-line arguments override everything else. Environment variables override `.env` file and defaults.

---

## Best Practices

1. **Use `.env` for local development**
   - Easy to manage and version control (with .env.example)
   - Keep `.env` in `.gitignore`
   - Commit `.env.example` with safe defaults

2. **Use environment variables for production**
   - Better for CI/CD and containerized environments
   - Secrets management integration (e.g., Kubernetes Secrets)
   - No risk of committing sensitive data

3. **Use command-line arguments for one-off overrides**
   - Quick testing without changing configuration
   - Scripting and automation
   - Debugging specific settings

4. **Never commit API keys**
   - Always use environment variables or secrets management
   - Add `.env` to `.gitignore`
   - Use `.env.example` for documentation

5. **Document configuration changes**
   - Update `.env.example` when adding new settings
   - Add comments explaining non-obvious settings
   - Document valid values and ranges

6. **Validate configuration early**
   - Let Pydantic validation catch errors at startup
   - Use type hints and constraints
   - Write tests for custom validators

---

## Configuration File Template

**`.env.example`** (committed to git):
```bash
# ============================================================================
# Arandu Configuration Template
# Copy this file to .env and fill in your values
# ============================================================================

# ============================================================================
# TranscriberConfig (ARANDU_)
# ============================================================================

# Model settings
ARANDU_MODEL_ID=openai/whisper-large-v3
ARANDU_LANGUAGE=  # Optional: pt, en, etc. (auto-detect if not set)
ARANDU_CHUNK_LENGTH_S=30
ARANDU_STRIDE_LENGTH_S=5

# Hardware settings
ARANDU_FORCE_CPU=false
ARANDU_QUANTIZE=false

# Google Drive settings
ARANDU_CREDENTIALS=credentials.json
ARANDU_TOKEN=token.json

# Batch processing
ARANDU_WORKERS=2

# Paths
ARANDU_INPUT_DIR=./input
ARANDU_RESULTS_DIR=./results
ARANDU_HF_CACHE_DIR=./cache/huggingface

# Processing
ARANDU_MAX_RETRIES=3
ARANDU_RETRY_DELAY=1.0

# ============================================================================
# QAConfig (ARANDU_QA_)
# ============================================================================

# LLM Provider
ARANDU_QA_PROVIDER=ollama  # openai, ollama, custom
ARANDU_QA_MODEL_ID=qwen3:14b
ARANDU_QA_OLLAMA_URL=http://localhost:11434/v1
# ARANDU_QA_BASE_URL=  # For custom OpenAI-compatible endpoints

# Generation
ARANDU_QA_TEMPERATURE=0.7
ARANDU_QA_MAX_TOKENS=8192

# Output
ARANDU_QA_OUTPUT_DIR=qa_dataset

# Language
ARANDU_QA_LANGUAGE=pt  # pt or en
ARANDU_QA_WORKERS=2

# ============================================================================
# CEPConfig (ARANDU_CEP_)
# ============================================================================

# Module toggles
ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT=true
ARANDU_CEP_ENABLE_SCAFFOLDING_GATE=true
ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT=true

# Bloom scaffolding (integer pair counts per level; single source of truth)
ARANDU_CEP_BLOOM_DISTRIBUTION='{"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}'
ARANDU_CEP_MAX_SCAFFOLDING_PAIRS=10

# Judge gates are per criterion and live in
# prompts/judge/criteria/<criterion>/config.json, not in the environment.

# Language
ARANDU_CEP_LANGUAGE=pt

# ============================================================================
# JudgeConfig (ARANDU_JUDGE_) - validator client for judge-qa / judge-transcription
# ============================================================================

ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama  # openai, ollama, custom
# ARANDU_JUDGE_VALIDATOR_BASE_URL=  # For custom OpenAI-compatible endpoints
ARANDU_JUDGE_TEMPERATURE=0.1
ARANDU_JUDGE_LANGUAGE=pt

# ============================================================================
# KGConfig (ARANDU_KG_)
# ============================================================================

# Backend
ARANDU_KG_BACKEND=atlas

# LLM Provider
ARANDU_KG_PROVIDER=ollama  # openai, ollama, custom
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_OLLAMA_URL=http://localhost:11434/v1
# ARANDU_KG_BASE_URL=  # For custom OpenAI-compatible endpoints

# LLM settings
ARANDU_KG_TEMPERATURE=0.5

# Language and prompts
ARANDU_KG_LANGUAGE=pt

# Output
ARANDU_KG_OUTPUT_DIR=knowledge_graphs

# ============================================================================
# LLMConfig (No prefix - uses aliases)
# ============================================================================

# Custom endpoints
# ARANDU_LLM_BASE_URL=http://localhost:11434/v1

# ============================================================================
# ResultsConfig (ARANDU_RESULTS_)
# ============================================================================

ARANDU_RESULTS_BASE_DIR=./results
ARANDU_RESULTS_ENABLE_VERSIONING=true
```

---

**Document Version**: 3.0  
**Last Updated**: 2026-06-16  
**Changes**: Synced to the per-domain config modules (no flat `config.py`); replaced the removed `TranscriptionQualityConfig` with `JudgeConfig`; CEP now uses `bloom_distribution` (no `bloom_levels`/`enable_validation`) with four judge weights; removed nonexistent KG `merge_graphs`/`output_format`/`schema_mode`/`prompt_path` fields.
