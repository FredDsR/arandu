---
title: Transcription Judging
description: Judge transcription outputs with heuristic and optional LLM criteria.
---

Use `arandu judge-transcription` after transcription to evaluate record quality.
Judging is a separate CLI stage.

## What gets persisted

Each `*_transcription.json` record receives:

- `validation: JudgePipelineResult`
- computed `is_valid` (derived from `validation.passed`)

## Pipeline stages

1. `heuristic_filter`
   - `content_length_floor`
   - `script_match`
   - `repetition`
   - `content_density`
   - `segment_quality`
2. `llm_filter` (only when a validator model is configured)
   - `language_drift`
   - `hallucination_loop`

If a filter stage rejects, downstream non-`always` stages are skipped.

## Usage

```bash
# Heuristic-only
arandu judge-transcription results/

# Heuristics + LLM criteria
arandu judge-transcription results/ --validator-model qwen3:14b

# Resume mode is default; rejudge forces full rerun
arandu judge-transcription results/ --validator-model qwen3:14b --rejudge
```

## Options

- `--language/-l`
- `--validator-model`
- `--validator-provider`
- `--validator-base-url`
- `--validator-temperature`
- `--validator-max-tokens`
- `--rejudge/--resume`

## Related settings

- `ARANDU_JUDGE_LANGUAGE`
- `ARANDU_JUDGE_VALIDATOR_MODEL`
- `ARANDU_JUDGE_VALIDATOR_PROVIDER`
- `ARANDU_JUDGE_VALIDATOR_BASE_URL`
- `ARANDU_JUDGE_TEMPERATURE`
- `ARANDU_JUDGE_MAX_TOKENS`

---

See also: [CLI Reference](/reference/cli/)
