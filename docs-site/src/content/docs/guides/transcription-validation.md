---
title: Transcription Quality Validation
description: Detect common Whisper failure modes using lightweight heuristic checks.
---

Detect common Whisper failure modes in transcription output using lightweight heuristic checks.

## Overview

The transcription quality validator catches failures that Whisper produces silently:

- **Wrong language/script** - Japanese characters when expecting Portuguese (strongest signal)
- **Repeated words/phrases** - e.g., "Obrigada" x30, looping phrase artifacts
- **Suspicious segment patterns** - uniform 1-second intervals, empty segments
- **Abnormal content density** - too few or too many words per minute

Validation runs **inline** after every transcription (CPU-only, negligible overhead) and is also available as a standalone CLI command for retroactive checks.

## How It Works

### Weighted Scoring

Each transcription is evaluated on four dimensions. The overall quality score is a weighted average:

| Check | Weight | What it detects |
|-------|--------|-----------------|
| **Script/charset match** | 0.35 | Text uses wrong alphabet (e.g., CJK when expecting Latin) |
| **Repetition detection** | 0.30 | Single word floods, repeated multi-word phrases |
| **Segment patterns** | 0.20 | Uniform timestamp intervals, empty segments |
| **Content density** | 0.15 | Words per minute outside 30-300 wpm range |

Each dimension produces a score from 0.0 (worst) to 1.0 (best). The weighted average becomes the `overall_score`. If the overall score falls below the configured `quality_threshold` (default: 0.5), the record is marked `is_valid: false`.

> **Note**: Weights must sum to 1.0. A `@model_validator` on `TranscriptionQualityConfig` enforces this constraint.

### Schema Extension

Validated records include two new fields on `EnrichedRecord`:

```json
{
  "transcription_quality": {
    "script_match_score": 1.0,
    "repetition_score": 0.85,
    "segment_quality_score": 1.0,
    "content_density_score": 0.95,
    "overall_score": 0.94,
    "issues_detected": [],
    "quality_rationale": null
  },
  "is_valid": true
}
```

- `is_valid: true` - Passed quality check
- `is_valid: false` - Failed quality check (issues detected)
- `is_valid: null` - Not yet validated (pre-existing records)

The `null` sentinel distinguishes records that predate validation from those that were actively checked.

## Inline Validation

Validation runs automatically in all transcription entry points:

- `arandu transcribe` - Single file transcription
- `arandu drive-transcribe` - Google Drive file transcription
- `arandu batch-transcribe` - Batch processing

No extra flags are needed. Quality issues are logged as warnings:

```
WARNING - Quality issues detected: ['high_word_repetition:obrigada:30', 'repeated_phrase:obrigada obrigada obrigada:10']
```

### Disabling Inline Validation

```bash
export ARANDU_QUALITY_ENABLED=false
```

When disabled, records are marked `is_valid: true` and `transcription_quality: null`.

## Retroactive CLI

Judge existing transcription files via `arandu judge-transcription`. See the [CLI reference](/reference/cli/#judge-transcription) for usage and flags.

## Validation Dimensions in Detail

### 1. Script/Charset Match (Weight: 0.35)

Checks whether the text uses the expected character set for the configured language.

For Latin-script languages (`pt`, `en`, `es`, `fr`, `de`, `it`), the validator uses `unicodedata.name()` to classify each alphabetic character as Latin, CJK, or other. This approach correctly handles:

- Latin Extended-B characters (e.g., `ǎ`, `ǒ`)
- Combining diacriticals used in Portuguese (e.g., `ã`, `ç`, `ê`)
- Mixed-script text

**Scoring**:
- 100% Latin text → score 1.0
- \>50% CJK characters → score 0.0 (definitive wrong-language signal)
- Non-Latin ratio exceeds `max_non_latin_ratio` (default: 10%) → score 0.2
- No alphabetic content → score 0.5 (neutral)

**Issue tags**: `wrong_script:cjk_detected`, `high_non_latin_ratio`, `no_alphabetic_content`

### 2. Repetition Detection (Weight: 0.30)

Detects both single-word floods and repeated multi-word phrases.

**Word repetition**: If the most common word exceeds `max_word_repetition_ratio` (default: 15%) of total words, it is flagged.

**Phrase repetition**: Checks 3-gram, 4-gram, and 5-gram patterns. If any phrase appears more than `max_phrase_repetition_count` (default: 4) times, it is flagged with its text coverage ratio.

**Scoring**: Based on the worst repetition ratio found (not the count of issues). Score = `1.0 - worst_ratio`, clamped to [0.0, 1.0]. This ensures "Obrigada" x30 (ratio 1.0) scores 0.0.

**Special case**: Transcriptions shorter than 5 words receive a neutral score of 0.7 with a `very_short_transcription` issue.

**Issue tags**: `high_word_repetition:<word>:<count>`, `repeated_phrase:<text>:<count>`, `very_short_transcription`

### 3. Segment Pattern Analysis (Weight: 0.20)

Analyzes Whisper's timestamp segments for anomalies that indicate processing artifacts.

**Uniform intervals**: Flags when 5+ consecutive segments have ~1-second start-time intervals (within `uniform_interval_tolerance` of ±0.1s). This pattern suggests Whisper fell into a timestamp loop.

**Empty segments**: Flags when more than 20% (`max_empty_segment_ratio`) of segments have no text content.

**Scoring**:
- No segments provided → score 1.0 (not penalized)
- Empty segment ratio exceeded → -0.3
- Suspicious uniform intervals → -0.5

**Issue tags**: `suspicious_uniform_intervals:<count>`, `high_empty_segments:<empty>/<total>`

### 4. Content Density (Weight: 0.15)

Checks whether the words-per-minute ratio falls within a reasonable range for spoken language.

**Range**: `min_words_per_minute` (default: 30) to `max_words_per_minute` (default: 300).

**Scoring**:
- Within range → score 1.0
- Below minimum → linearly scaled from 0.0 (at 0 wpm) to 1.0 (at min threshold)
- Above maximum → linearly scaled from 1.0 (at max threshold) to 0.0 (at 2x max)
- Duration unknown (`null`) → score 0.5 (neutral, doesn't skew results)
- Invalid duration (0 or negative) → score 0.3

**Issue tags**: `low_content_density:<wpm>_wpm`, `high_content_density:<wpm>_wpm`, `duration_unknown:neutral_score`, `invalid_duration`

## LLM-Based Criteria (Optional)

Two additional criteria run as a second pipeline stage when `TranscriptionJudge` is given an `LLMClient`. They target failure modes that pure-heuristic checks cannot detect.

- **`language_drift`** — detects when sustained content is in a different Latin-script language than expected (e.g., English content in a Portuguese transcription). The heuristic `script_match` criterion cannot catch this because English and Portuguese share the Latin alphabet. Default threshold: `0.8`.
- **`hallucination_loop`** — detects formulaic Whisper hallucinations that slip past the heuristic `repetition` criterion: YouTube-style openings/closings, short sentence loops that appear only a handful of times, apology/filler loops, channel-name "signatures". Default threshold: `0.7`.

Prompts live under `prompts/judge/criteria/language_drift/{pt,en}/prompt.md` and `prompts/judge/criteria/hallucination_loop/{pt,en}/prompt.md`. Thresholds live in each criterion's `config.json`. Both are domain-neutral by design — they target generic transcription failure modes, not interview-specific content.

### Pipeline Behavior

The pipeline is two filter stages in order:

1. `heuristic_filter` — content length floor + script match + repetition + content density + segment quality. The `content_length_floor` criterion runs first and can short-circuit the rest of the heuristic stage.
2. `llm_filter` — language drift + hallucination loop (only when an `LLMClient` is provided).

If the heuristic stage rejects — including an early rejection from `content_length_floor` — the LLM stage is skipped, so there are no wasted LLM calls on transcriptions already flagged by cheap checks.

### Programmatic Usage

```python
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.transcription.judge import TranscriptionJudge

# Heuristics only (no LLM calls)
judge = TranscriptionJudge(language="pt")

# Heuristics + LLM stage
validator = LLMClient(provider=LLMProvider.OLLAMA, model_id="qwen3:14b")
judge = TranscriptionJudge(language="pt", validator_client=validator)

result = judge.evaluate_transcription(
    text=record.transcription_text,
    duration_ms=record.duration_milliseconds,
    segments=record.segments,
)
if not result.passed:
    print(f"Rejected at stage: {result.rejected_at}")
```

### Smoke-testing the Pipeline

`scripts/test_transcription_judge.py` exercises both stages against real transcription files:

```bash
# Heuristics only
uv run python scripts/test_transcription_judge.py \
    --input-dir results/test-cep-01/transcription/outputs \
    --files 5

# Heuristics + LLM (Ollama)
uv run python scripts/test_transcription_judge.py \
    --validator-model qwen3:14b \
    --input-dir results/test-cep-01/transcription/outputs \
    --files 5

# Single-file mode (useful for reproducing a known-bad case)
uv run python scripts/test_transcription_judge.py \
    --validator-model qwen3:14b \
    --file results/test-cep-01/transcription/outputs/<id>_transcription.json
```

### Limitations of the LLM Criteria

**These criteria do not catch all Whisper hallucinations.** Be explicit about what they do and don't detect before relying on them:

- **`hallucination_loop` only catches formulaic/pattern hallucinations.** It is designed to flag content that looks copied from Whisper's training distribution (stock phrases, channel openings, short loops, implausibly repeated interjections). It does **not** reliably catch:
  - **Plausible silence-fillers** — a single coherent sentence Whisper invents from background noise when no speech occurred.
  - **Low-SNR invention** — phonetically-close but wrong words across a real utterance. The output reads naturally.
  - **Name/number substitutions** — "João" → "Joaquim", "15 anos" → "50 anos".

  These are fundamentally undetectable from text alone — they require audio-aware signals (Whisper `avg_logprob` / `no_speech_prob` per segment, voice-activity detection, or multi-model cross-check). Adding audio-aware heuristics is tracked separately from this criterion.

- **`language_drift` tolerates isolated loanwords and proper nouns by design.** It flags *sustained* non-expected-language content, not single borrowed words, acronyms, or technical terms. Its ceiling is the LLM's own competence in the target languages; exotic code-switching targets (e.g., indigenous languages) may be over-tolerated.

- **Text-only ceiling.** Neither criterion can distinguish "real but unusual speech" from "plausibly fabricated speech" without access to the audio. If an interview contains genuinely ordinary conversational content, the LLM judge has no grounds to flag it — which is the correct behavior, but means well-hidden fabrications will pass.

- **LLM cost + latency.** Each transcription triggers two LLM calls (one per criterion). For large corpora, budget accordingly or keep the LLM stage disabled at ingestion time and run it selectively via the smoke script.

- **Thresholds are defaults, not calibrated.** The 0.8 / 0.7 defaults come from the rubric design. Empirical calibration against a labeled set is tracked as follow-up work.

## Configuration Reference

All settings use the `ARANDU_QUALITY_` environment variable prefix.

### General Settings

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `enabled` | `bool` | `True` | `ARANDU_QUALITY_ENABLED` |
| `quality_threshold` | `float` | `0.5` | `ARANDU_QUALITY_QUALITY_THRESHOLD` |
| `expected_language` | `str` | `"pt"` | `ARANDU_QUALITY_EXPECTED_LANGUAGE` |

### Dimension Weights (must sum to 1.0)

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `script_match_weight` | `float` | `0.35` | `ARANDU_QUALITY_SCRIPT_MATCH_WEIGHT` |
| `repetition_weight` | `float` | `0.30` | `ARANDU_QUALITY_REPETITION_WEIGHT` |
| `segment_quality_weight` | `float` | `0.20` | `ARANDU_QUALITY_SEGMENT_QUALITY_WEIGHT` |
| `content_density_weight` | `float` | `0.15` | `ARANDU_QUALITY_CONTENT_DENSITY_WEIGHT` |

### Detection Thresholds

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `max_non_latin_ratio` | `float` | `0.1` | `ARANDU_QUALITY_MAX_NON_LATIN_RATIO` |
| `max_word_repetition_ratio` | `float` | `0.15` | `ARANDU_QUALITY_MAX_WORD_REPETITION_RATIO` |
| `max_phrase_repetition_count` | `int` | `4` | `ARANDU_QUALITY_MAX_PHRASE_REPETITION_COUNT` |
| `suspicious_uniform_intervals` | `int` | `5` | `ARANDU_QUALITY_SUSPICIOUS_UNIFORM_INTERVALS` |
| `min_words_per_minute` | `float` | `30.0` | `ARANDU_QUALITY_MIN_WORDS_PER_MINUTE` |
| `max_words_per_minute` | `float` | `300.0` | `ARANDU_QUALITY_MAX_WORDS_PER_MINUTE` |
| `max_empty_segment_ratio` | `float` | `0.2` | `ARANDU_QUALITY_MAX_EMPTY_SEGMENT_RATIO` |
| `uniform_interval_tolerance` | `float` | `0.1` | `ARANDU_QUALITY_UNIFORM_INTERVAL_TOLERANCE` |

### Example `.env` Configuration

```bash
# Transcription Quality Validation
ARANDU_QUALITY_ENABLED=true
ARANDU_QUALITY_QUALITY_THRESHOLD=0.6
ARANDU_QUALITY_EXPECTED_LANGUAGE=pt

# Stricter repetition detection
ARANDU_QUALITY_MAX_WORD_REPETITION_RATIO=0.10
ARANDU_QUALITY_MAX_PHRASE_REPETITION_COUNT=3
```

## Interpreting Results

### Common Failure Patterns

| Pattern | Typical Score | Root Cause |
|---------|--------------|------------|
| Japanese text for Portuguese audio | 0.15-0.30 | Whisper language detection failure |
| "Obrigada" x30 | 0.65-0.70 | Whisper repetition loop |
| Uniform 1-second segments | 0.50-0.70 | Whisper timestamp artifact |
| 5 words in 60 seconds | 0.80-0.90 | Mostly silence or background noise |

### Threshold Guidance

| Threshold | Use Case |
|-----------|----------|
| `0.5` (default) | Catches severe failures (wrong script, extreme repetition + other issues) |
| `0.6-0.7` | Balanced — catches most artifacts while keeping borderline transcriptions |
| `0.75-0.8` | Strict — flags single-dimension failures (e.g., pure repetition) |

> **Tip**: A pure repetition case (e.g., "Obrigada" x30) scores ~0.70 overall because repetition weight (0.30) alone cannot pull the weighted average below 0.50 when all other dimensions score well. Use a threshold of 0.75+ to catch these cases.

## Programmatic Usage

```python
from arandu.config import TranscriptionQualityConfig
from arandu.core.transcription_validator import (
    TranscriptionValidator,
    validate_enriched_record,
    get_quality_issues,
)

# Validate a single record (mutates in-place)
validate_enriched_record(record)

# With custom config
config = TranscriptionQualityConfig(quality_threshold=0.7, expected_language="en")
validate_enriched_record(record, config)

# Reuse validator across multiple records (batch optimization)
validator = TranscriptionValidator(config)
for record in records:
    validate_enriched_record(record, validator=validator)

# Check for issues
issues = get_quality_issues(record)
if issues:
    print(f"Failed: {issues}")
```

## Known Limitations

- **Non-Latin script detection**: Currently only detects CJK (Chinese, Japanese, Korean) as wrong-script for Latin languages. Arabic, Cyrillic, and other scripts are tracked via the generic `non_latin_ratio` threshold but lack dedicated detection.
- **Language probability**: Not used as a scoring dimension because valid Portuguese transcriptions can have `language_probability: 0.0`. Character-level script detection is more reliable.
- **Content density with silence**: Long audio with extended silence produces low wpm but may still be a valid transcription. Consider the context when interpreting density scores.

---

**See also**: [Transcription](/guides/transcription/) | [Configuration](/configuration/) | [Getting Started](/getting-started/)
