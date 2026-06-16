# Transcription Quality Validation Guide

Detect common Whisper failure modes in transcription output with a two-stage filter pipeline: pure-Python heuristics first, then optional LLM criteria.

## Overview

The transcription judge catches failures that Whisper produces silently:

- **Too-short output** - records too short to carry extractable content
- **Wrong language/script** - CJK characters when expecting Portuguese (strongest heuristic signal)
- **Repeated words/phrases** - e.g., "Obrigada" x30, looping phrase artifacts
- **Abnormal content density** - too few or too many words per minute
- **Suspicious segment patterns** - uniform 1-second intervals, empty segments
- **Latin-script language drift** - sustained English/French content that the script check passes by design (LLM stage)
- **Formulaic hallucinations** - YouTube-style openings/closings, short apology/filler loops (LLM stage)

Validation runs as a standalone CLI command (`judge-transcription`) over a directory of `*_transcription.json` records. There is no inline post-transcription hook: judging is an explicit, separate step.

## How It Works

### Two-Stage Filter Pipeline

The judge is a **filter pipeline**, not a weighted average. There is no aggregate or overall score. Each stage runs a set of criteria, and a record passes a stage only if it clears **every** criterion in that stage. A failing filter stage rejects the record and the pipeline stops (later stages are skipped).

| Stage | Mode | Criteria | Runs when |
|-------|------|----------|-----------|
| `heuristic_filter` | filter | content length floor, script match, repetition, content density, segment quality | always (CPU-only, no model) |
| `llm_filter` | filter | `language_drift`, `hallucination_loop` | only when a validator LLM client is configured |

The length floor runs first inside the heuristic stage, so very short records short-circuit the rest of the pipeline before any further check or LLM call. The LLM stage is skipped automatically whenever the heuristic stage already rejected a record.

Each criterion produces a score in `[0.0, 1.0]` and passes when `score >= threshold`. The pipeline verdict is binary:

- `passed = True` only when no filter stage rejected the record.
- `passed = False` when any filter stage rejected it (`rejected_at` records which stage).

### Verdict Persistence

`judge-transcription` writes the verdict back into each `*_transcription.json` record. The full pipeline result lands in `validation` and `is_valid` is derived from `validation.passed`, so the two cannot drift:

```json
{
  "validation": {
    "passed": false,
    "rejected_at": "heuristic_filter",
    "stage_results": {
      "heuristic_filter": {
        "criterion_scores": {
          "content_length_floor": { "score": 1.0, "threshold": 0.5, "rationale": "length_within_floor" },
          "script_match": { "score": 1.0, "threshold": 0.6, "rationale": "Text uses expected character set." },
          "repetition": { "score": 0.0, "threshold": 0.5, "rationale": "high_word_repetition:obrigada:30" },
          "content_density": { "score": 0.95, "threshold": 0.4, "rationale": "Content density is within normal range." },
          "segment_quality": { "score": 1.0, "threshold": 0.4, "rationale": "Segment patterns are within normal range." }
        }
      }
    }
  },
  "is_valid": false
}
```

- `is_valid: true` - passed every filter stage
- `is_valid: false` - rejected by a filter stage (see `rejected_at`)
- `is_valid: null` - not yet judged (the `validation` field is absent)

No aggregate side-file is produced. Run a downstream analytics script over the directory for cross-record reports.

## Running the Judge

The command operates on a directory of transcription JSON files. Heuristics always run; the LLM stage is opt-in.

### Heuristics Only

```bash
arandu judge-transcription results/
```

With no validator model configured, the command prints a notice and runs heuristic-only mode (the `language_drift` and `hallucination_loop` criteria are skipped). Each `*_transcription.json` file is updated in place.

### With the LLM Stage

```bash
arandu judge-transcription results/ --validator-model qwen3:14b
arandu judge-transcription results/ --validator-model gemini-2.5-flash
```

Supplying a validator model enables the second filter stage. The command fails fast before walking the input directory if the validator is misconfigured or unreachable.

### Custom Language

```bash
arandu judge-transcription results/ --language en
```

Sets the expected language for the script-match heuristic and the LLM prompts (default: `pt`).

### Rejudge vs Resume

```bash
# Default: resume. Skip records that already carry a `validation` payload.
arandu judge-transcription results/

# Force a fresh pass over every record (e.g. after changing the validator model).
arandu judge-transcription results/ --validator-model qwen3:14b --rejudge
```

`--resume` (the default) only judges the unjudged remainder, so a re-submission after a wall-clock limit picks up where it left off. `--rejudge` re-evaluates everything from scratch.

### CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `INPUT_DIR` | Path | (required) | Directory containing `*_transcription.json` files |
| `--language` / `-l` | str | `pt` | Expected transcription language code |
| `--validator-model` | str | (none) | Model ID enabling the LLM filter stage. Falls back to `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `--validator-provider` | str | inferred | `openai`, `ollama`, or `custom`. Falls back to `ARANDU_JUDGE_VALIDATOR_PROVIDER` |
| `--validator-base-url` | str | inferred | Base URL for the validator. Falls back to `ARANDU_JUDGE_VALIDATOR_BASE_URL`, then `ARANDU_LLM_BASE_URL` |
| `--validator-temperature` | float | `0.3` | Sampling temperature for LLM criteria. Falls back to `ARANDU_JUDGE_TEMPERATURE` |
| `--validator-max-tokens` | int | `2048` | Max tokens for LLM criterion responses. Falls back to `ARANDU_JUDGE_MAX_TOKENS` |
| `--rejudge` / `--resume` | flag | `--resume` | `--rejudge` re-evaluates everything; `--resume` skips already-judged records |

**Note**: When `--validator-provider` is unset, the provider is inferred from `ARANDU_LLM_BASE_URL` (`custom` when that variable is set, otherwise `ollama`). An explicit `custom` provider requires a base URL, or the command exits with an error.

### Output

The command prints a tally:

```
Total files: 3
Passed: 2
Failed: 1
Resumed (already judged, skipped): 0
```

## Heuristic Criteria in Detail

All five heuristic criteria run in the `heuristic_filter` stage. Each has its own pass threshold; the record passes the stage only when all five pass.

### 1. Content Length Floor (threshold 0.5)

Binary gate that runs first so very short records short-circuit the pipeline. Scores `1.0` when the stripped text clears both floors, `0.0` otherwise. There is no partial credit.

- `min_chars`: 200 characters (after stripping)
- `min_words`: 30 whitespace-tokenised words

Rejecting tiny records here closes the silence-filler gap that content density alone admits (a short "Thank you." can score above the density threshold while carrying no extractable content).

**Issue tag**: `text_too_short:<chars>_chars_<words>_words`

### 2. Script / Charset Match (threshold 0.6)

Checks whether the text uses the expected character set for the configured language. For Latin-script languages (`pt`, `en`, `es`, `fr`, `de`, `it`), it uses `unicodedata.name()` to classify each alphabetic character, correctly handling Latin Extended-B characters and Portuguese combining diacriticals (`ã`, `ç`, `ê`).

**Scoring**:
- 100% Latin text -> 1.0
- More than 50% CJK characters -> 0.0 (definitive wrong-language signal)
- Non-Latin ratio above `max_non_latin_ratio` (default 0.1) -> 0.2
- No alphabetic content -> 0.5 (neutral)

**Issue tags**: `wrong_script:cjk_detected`, `high_non_latin_ratio`, `no_alphabetic_content`

### 3. Repetition Detection (threshold 0.5)

Detects single-word floods and repeated multi-word phrases.

- **Word repetition**: flagged when the most common word exceeds `max_word_repetition_ratio` (default 0.15) of total words.
- **Phrase repetition**: checks 3-, 4-, and 5-gram patterns; flagged when any phrase appears at least `max_phrase_repetition_count` (default 4) times, with its coverage ratio.

**Scoring**: based on the worst repetition ratio found, `score = max(0.0, 1.0 - worst_ratio)`. A single extreme case (e.g., "Obrigada" x30) therefore scores worse than several mild repetitions. Transcriptions shorter than 5 words receive a neutral 0.7 with a `very_short_transcription` tag.

**Issue tags**: `high_word_repetition:<word>:<count>`, `repeated_phrase:<text>:<count>`, `very_short_transcription`

### 4. Content Density (threshold 0.4)

Checks whether the words-per-minute ratio falls within a reasonable range for spoken language.

**Range**: `min_words_per_minute` (default 30.0) to `max_words_per_minute` (default 300.0).

**Scoring**:
- Within range -> 1.0
- Below minimum -> linearly scaled (`wpm / min`)
- Above maximum -> linearly scaled (`2.0 - wpm / max`)
- Duration unknown (`null`) -> 0.5 (neutral, does not skew results)
- Invalid duration (0 or negative) -> 0.3

**Issue tags**: `low_content_density:<wpm>_wpm`, `high_content_density:<wpm>_wpm`, `duration_unknown:neutral_score`, `invalid_duration`

### 5. Segment Quality (threshold 0.4)

Analyzes Whisper timestamp segments for processing artifacts.

- **Uniform intervals**: flagged when `suspicious_uniform_intervals` (default 5) or more consecutive segments have ~1-second start-time gaps (within `uniform_interval_tolerance` of +/- 0.1s), suggesting a timestamp loop.
- **Empty segments**: flagged when more than `max_empty_segment_ratio` (default 0.2) of segments have no text.

**Scoring**:
- No segments provided -> 0.5 (neutral)
- No anomalies -> 1.0
- Empty-segment ratio exceeded -> minus 0.3
- Suspicious uniform intervals -> minus 0.5

**Issue tags**: `suspicious_uniform_intervals:<count>`, `high_empty_segments:<empty>/<total>`

## LLM Criteria in Detail

The `llm_filter` stage runs only when a validator model is configured. Both criteria are text-only and target failures the heuristics cannot detect. Each is loaded from a prompt file under `prompts/judge/criteria/` and gated by a per-criterion threshold defined in that criterion's `config.json`.

### `language_drift` (threshold 0.8)

Catches sustained Latin-script language drift (e.g., English content in a Portuguese transcription) that `script_match` passes by design, because the alphabet is still Latin. Lexical loans, proper nouns, and short acronyms are tolerated; whole sentences in another language count as drift.

### `hallucination_loop` (threshold 0.7)

Catches formulaic Whisper hallucinations that `repetition` misses because its n-gram threshold is tuned for heavy repetition: YouTube/podcast-style openings and closings, short artificial loops appearing only a handful of times, and formulaic filler. Language is irrelevant for this criterion.

**Limitations**: both LLM criteria are text-only and cannot detect plausible silence-fillers (a single coherent invented sentence with no loop signature), low-SNR invention (phonetically-close but wrong words), or natural-reading name/number substitutions. Those require audio-aware signals (`avg_logprob` / `no_speech_prob` per segment, VAD, multi-model cross-check) and are tracked separately.

## Configuration Reference

LLM-stage settings use the `ARANDU_JUDGE_` environment variable prefix. They serve as fallbacks for the matching CLI options.

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `validator_model` | `str \| None` | `None` | `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `validator_provider` | `str \| None` | inferred | `ARANDU_JUDGE_VALIDATOR_PROVIDER` |
| `validator_base_url` | `str \| None` | inferred | `ARANDU_JUDGE_VALIDATOR_BASE_URL` |
| `temperature` | `float` | `0.3` | `ARANDU_JUDGE_TEMPERATURE` |
| `max_tokens` | `int` | `2048` | `ARANDU_JUDGE_MAX_TOKENS` |
| `language` | `str` | `"pt"` | `ARANDU_JUDGE_LANGUAGE` |

**Note**: The per-criterion thresholds for the heuristics (content length 0.5, script match 0.6, repetition 0.5, content density 0.4, segment quality 0.4) and the LLM criteria (`language_drift` 0.8, `hallucination_loop` 0.7) are defined in code and prompt `config.json` files. They are not exposed as environment variables.

### Example `.env` Configuration

```bash
# Enable the LLM filter stage against a local Ollama model
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
ARANDU_JUDGE_TEMPERATURE=0.3
ARANDU_JUDGE_MAX_TOKENS=2048
```

## Interpreting Results

Because the model is a filter (not a weighted average), a record fails as soon as any single criterion in a stage drops below its threshold. There is no compensation between criteria.

### Common Failure Patterns

| Pattern | Rejecting criterion | Root cause |
|---------|---------------------|------------|
| Under 200 chars / 30 words | `content_length_floor` | Mostly silence, truncated output |
| Japanese text for Portuguese audio | `script_match` | Whisper language-detection failure |
| "Obrigada" x30 | `repetition` | Whisper repetition loop |
| 5 words in 60 seconds | `content_density` | Mostly silence or background noise |
| Uniform 1-second segments | `segment_quality` | Whisper timestamp artifact |
| English narration in a pt transcription | `language_drift` (LLM) | Latin-script drift the script check passes |
| "Welcome back to our channel" filler | `hallucination_loop` (LLM) | Formulaic hallucination |

### Tuning Guidance

- Run heuristics-only first to clear the obvious failures cheaply; the length floor and script match catch the bulk of bad records with no model.
- Enable the LLM stage when you need to catch Latin-script drift and formulaic hallucinations that the heuristics pass. Both add per-record LLM calls, so budget accordingly.
- The criteria thresholds are tuned for Portuguese riverine-community interviews. Pure repetition cases that the old weighted model could not reject are now caught deterministically by the dedicated `repetition` and `content_length_floor` criteria, since each gate fails independently.

## Programmatic Usage

```python
from arandu.transcription.judge import TranscriptionJudge, build_validator_client

# Heuristics only
judge = TranscriptionJudge(language="pt")
result = judge.evaluate_transcription(
    text=record.transcription_text,
    duration_ms=record.duration_milliseconds,
    segments=record.segments or [],
)
print(result.passed, result.rejected_at)

# With the LLM filter stage
client = build_validator_client("qwen3:14b", provider="ollama")
judge = TranscriptionJudge(language="pt", validator_client=client)
result = judge.evaluate_transcription(text=record.transcription_text)
```

`evaluate_transcription` returns a `JudgePipelineResult` with `passed`, `rejected_at`, and per-stage `stage_results`. There is no overall numeric score to read.

---

**See also**: [Transcription](transcription.md) | [Configuration](configuration.md) | [Getting Started](getting-started.md)

**Document Version**: 2.0
**Last Updated**: 2026-06-16
**Status**: Aligned with codebase
