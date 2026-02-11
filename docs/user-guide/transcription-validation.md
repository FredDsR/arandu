# Transcription Quality Validation Guide

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

- `gtranscriber transcribe` - Single file transcription
- `gtranscriber drive-transcribe` - Google Drive file transcription
- `gtranscriber batch-transcribe` - Batch processing

No extra flags are needed. Quality issues are logged as warnings:

```
WARNING - Quality issues detected: ['high_word_repetition:obrigada:30', 'repeated_phrase:obrigada obrigada obrigada:10']
```

### Disabling Inline Validation

```bash
export GTRANSCRIBER_QUALITY_ENABLED=false
```

When disabled, records are marked `is_valid: true` and `transcription_quality: null`.

## Retroactive Validation CLI

Validate existing transcription files that were produced before this feature:

### Basic Usage

```bash
gtranscriber validate-transcriptions results/
```

This updates each `*_transcription.json` file in-place with quality scores.

### Report Only (No File Changes)

```bash
gtranscriber validate-transcriptions results/ --report-only
```

Displays a validation summary table without modifying any files.

### Custom Threshold

```bash
gtranscriber validate-transcriptions results/ --threshold 0.7
```

Use a higher threshold for stricter quality requirements.

### Custom Language

```bash
gtranscriber validate-transcriptions results/ --language en
```

Override the expected language for script matching (default: `pt`).

### Save to Separate Directory

```bash
gtranscriber validate-transcriptions results/ --output-dir validated/
```

Write validated files to a new directory instead of updating in-place.

### Full Example

```bash
gtranscriber validate-transcriptions results_tupi/ \
  --threshold 0.6 \
  --language pt \
  --report-only
```

### Output

The CLI displays a summary table:

```
         Validation Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ File                        ┃ Valid ┃ Score ┃ Issues ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ abc123_transcription.json   │   ✓   │  0.94 │      0 │
│ def456_transcription.json   │   ✗   │  0.35 │      3 │
│ ghi789_transcription.json   │   ✓   │  0.82 │      1 │
└─────────────────────────────┴───────┴───────┴────────┘

Total files: 3
Valid: 2
Invalid: 1
```

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

## Configuration Reference

All settings use the `GTRANSCRIBER_QUALITY_` environment variable prefix.

### General Settings

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `enabled` | `bool` | `True` | `GTRANSCRIBER_QUALITY_ENABLED` |
| `quality_threshold` | `float` | `0.5` | `GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD` |
| `expected_language` | `str` | `"pt"` | `GTRANSCRIBER_QUALITY_EXPECTED_LANGUAGE` |

### Dimension Weights (must sum to 1.0)

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `script_match_weight` | `float` | `0.35` | `GTRANSCRIBER_QUALITY_SCRIPT_MATCH_WEIGHT` |
| `repetition_weight` | `float` | `0.30` | `GTRANSCRIBER_QUALITY_REPETITION_WEIGHT` |
| `segment_quality_weight` | `float` | `0.20` | `GTRANSCRIBER_QUALITY_SEGMENT_QUALITY_WEIGHT` |
| `content_density_weight` | `float` | `0.15` | `GTRANSCRIBER_QUALITY_CONTENT_DENSITY_WEIGHT` |

### Detection Thresholds

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `max_non_latin_ratio` | `float` | `0.1` | `GTRANSCRIBER_QUALITY_MAX_NON_LATIN_RATIO` |
| `max_word_repetition_ratio` | `float` | `0.15` | `GTRANSCRIBER_QUALITY_MAX_WORD_REPETITION_RATIO` |
| `max_phrase_repetition_count` | `int` | `4` | `GTRANSCRIBER_QUALITY_MAX_PHRASE_REPETITION_COUNT` |
| `suspicious_uniform_intervals` | `int` | `5` | `GTRANSCRIBER_QUALITY_SUSPICIOUS_UNIFORM_INTERVALS` |
| `min_words_per_minute` | `float` | `30.0` | `GTRANSCRIBER_QUALITY_MIN_WORDS_PER_MINUTE` |
| `max_words_per_minute` | `float` | `300.0` | `GTRANSCRIBER_QUALITY_MAX_WORDS_PER_MINUTE` |
| `max_empty_segment_ratio` | `float` | `0.2` | `GTRANSCRIBER_QUALITY_MAX_EMPTY_SEGMENT_RATIO` |
| `uniform_interval_tolerance` | `float` | `0.1` | `GTRANSCRIBER_QUALITY_UNIFORM_INTERVAL_TOLERANCE` |

### Example `.env` Configuration

```bash
# Transcription Quality Validation
GTRANSCRIBER_QUALITY_ENABLED=true
GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD=0.6
GTRANSCRIBER_QUALITY_EXPECTED_LANGUAGE=pt

# Stricter repetition detection
GTRANSCRIBER_QUALITY_MAX_WORD_REPETITION_RATIO=0.10
GTRANSCRIBER_QUALITY_MAX_PHRASE_REPETITION_COUNT=3
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
from gtranscriber.config import TranscriptionQualityConfig
from gtranscriber.core.transcription_validator import (
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

- **Non-Latin script detection**: Currently only detects CJK (Chinese, Japanese, Korean) as wrong-script for Latin languages. Arabic, Cyrillic, and other scripts are tracked via the generic `non_latin_ratio` threshold but lack dedicated detection. See [Issue #19](https://github.com/FredDsR/etno-kgc-preprocessing/issues/19) for follow-up.
- **Language probability**: Not used as a scoring dimension because valid Portuguese transcriptions can have `language_probability: 0.0`. Character-level script detection is more reliable.
- **Content density with silence**: Long audio with extended silence produces low wpm but may still be a valid transcription. Consider the context when interpreting density scores.

---

**See also**: [Transcription](transcription.md) | [Configuration](configuration.md) | [Getting Started](getting-started.md)
