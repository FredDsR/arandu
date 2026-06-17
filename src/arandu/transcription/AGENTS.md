# Transcription Module: Agent Guide

Whisper ASR over local files / Google Drive, plus a two-stage quality judge.
Config: `TranscriberConfig` (`config.py`, env prefix `ARANDU_`). CLI:
`arandu transcribe` / `drive-transcribe` / `batch-transcribe` / `judge-transcription`
(`cli/transcribe.py`). Output: `EnrichedRecord` JSON under
`results/<id>/transcription/outputs/`.

## Module map

| File | Role |
| ---- | ---- |
| `config.py` | `TranscriberConfig`: model, device (force_cpu/quantize), chunk/stride, Drive creds, paths |
| `engine.py` | `WhisperEngine`: lazy `transformers` pipeline, device/dtype selection, quantization |
| `batch.py` | Catalog load, parallel workers, Drive download + audio extract, `CheckpointManager` + `ResultsManager` |
| `judge.py` | `TranscriptionJudge` (two-stage filter pipeline) + `build_validator_client` |
| `media.py` | ffmpeg/ffprobe audio extraction, duration, audio-stream checks |
| `criteria/` | Heuristic criteria: `content_length`, `script_match`, `repetition`, `content_density`, `segment_quality` |

## Patterns to follow

- **Two-stage filter judge.** `heuristic_filter` (always) runs the five
  pure-Python criteria — content-length floor first (short-circuits), then
  script match, repetition, content density, segment quality. `llm_filter`
  (optional, only when a validator client is supplied, and only if heuristics
  pass) runs `language_drift` + `hallucination_loop`. Both are filter gates: a
  fail sets `passed=False` and `rejected_at=<stage>`. There is no aggregate score.
- **Validator client resolution** (`build_validator_client`): explicit
  `base_url` wins; `provider="custom"` requires a base URL; explicit
  `openai`/`ollama` ignore `ARANDU_LLM_BASE_URL`; unset provider infers `custom`
  when `ARANDU_LLM_BASE_URL` is set, else `ollama`. Model/provider/base-url fall
  back to `ARANDU_JUDGE_VALIDATOR_*` (shared with `judge-qa`).
- **Resume**: `batch` checkpoints completed/failed `file_id`s; re-running skips
  completed ones. Versioned output requires `ResultsConfig.enable_versioning`.

## Complex logic worth knowing

- `EnrichedRecord` mixes in `JudgeResultMixin`: judge output lands in
  `validation` (a `JudgePipelineResult`), `is_valid` is computed from
  `validation.passed`. A `@model_validator` migrates the retired
  `transcription_quality` payload to `validation` (or drops the very old
  weighted-score struct), so old records become re-judgeable.
- The heuristic criteria target Whisper failure modes: `script_match` uses
  `unicodedata` (catches CJK/garbage but intentionally *passes* same-script
  language drift, which `language_drift` catches); `segment_quality` flags
  uniform 1-second intervals (a hallucination signature); `content_length_floor`
  was added post-calibration to catch silence-filler loops.
- LLM criteria are text-only: they cannot catch plausible silence-fillers or
  low-SNR invention (those need audio-aware signals, tracked separately).

## Gotchas

- Validation is cached on the record; editing the transcription text does not
  re-judge — rerun `judge-transcription --rejudge`.
- `provider` inference is surprising: unset provider + `ARANDU_LLM_BASE_URL` set
  ⇒ `custom`. Test with the env both set and unset.
- Workers are process-isolated (one `WhisperEngine` each); config mutated after
  worker init is invisible to running workers.

**Deployment surface**: `arandu:latest` (`Dockerfile`); compose services
`arandu` / `arandu-cpu` / `arandu-rocm`. Full map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
