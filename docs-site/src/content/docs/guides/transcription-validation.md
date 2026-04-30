---
title: Transcription Quality Validation
description: Detect common Whisper failure modes using lightweight heuristic checks.
---

Detect common Whisper failure modes in transcription output using a composable judge pipeline (heuristic checks + optional LLM criteria).

## Overview

The transcription judge catches failures that Whisper produces silently:

- **Pre-filter on length** — short records that should be discarded before any other check (`content_length_floor`)
- **Wrong language/script** — Japanese characters when expecting Portuguese (strongest signal)
- **Repeated words/phrases** — e.g., "Obrigada" x30, looping phrase artifacts
- **Suspicious segment patterns** — uniform 1-second intervals, empty segments
- **Abnormal content density** — too few or too many words per minute
- **Latin-script language drift** (LLM stage) — sustained English content in a Portuguese transcription
- **Formulaic Whisper hallucinations** (LLM stage) — YouTube-style openings/closings, apology loops, channel signatures

Judging is a **separate step** in the pipeline — it runs after transcription via the `arandu judge-transcription` CLI, not inline at ingestion time. Verdicts are written back to each `*_transcription.json` record under the `validation` field.

## How It Works

### Multi-Stage Pipeline with Per-Criterion Thresholds

`TranscriptionJudge` runs a two-stage filter pipeline. Each stage applies *individual* thresholds per criterion — there is no weighted-average overall score. A record fails a stage as soon as any criterion's score is below its threshold, and a record fails the pipeline as soon as it fails any stage.

| Stage | Criteria | Type | Skipped when |
|-------|----------|------|--------------|
| `heuristic_filter` | `content_length_floor` → `script_match` → `repetition` → `content_density` → `segment_quality` | Pure-Python heuristics, CPU-only | Never (always runs) |
| `llm_filter` | `language_drift`, `hallucination_loop` | LLM-based | No validator client configured, OR previous stage already rejected |

`content_length_floor` runs first — it short-circuits the rest of the heuristic stage when a record is too short to evaluate meaningfully.

### Schema

Judged records carry the full `JudgePipelineResult` payload under `validation`. The `is_valid` field is derived from `validation.passed`:

```json
{
  "validation": {
    "passed": true,
    "stage_results": {
      "heuristic_filter": {
        "passed": true,
        "criterion_scores": {
          "content_length_floor": { "score": 1.0, "passed": true, "rationale": "..." },
          "script_match":         { "score": 1.0, "passed": true, "rationale": "..." },
          "repetition":           { "score": 0.85, "passed": true, "rationale": "..." },
          "content_density":      { "score": 0.95, "passed": true, "rationale": "..." },
          "segment_quality":      { "score": 1.0, "passed": true, "rationale": "..." }
        }
      },
      "llm_filter": {
        "passed": true,
        "criterion_scores": {
          "language_drift":      { "score": 0.92, "passed": true, "rationale": "..." },
          "hallucination_loop":  { "score": 0.88, "passed": true, "rationale": "..." }
        }
      }
    }
  },
  "is_valid": true
}
```

- `is_valid: true` — every criterion in every stage passed its threshold
- `is_valid: false` — at least one criterion was below its threshold
- `is_valid: null` — record has not been judged yet

The `llm_filter` stage block is absent when the judge ran without a validator client (heuristic-only mode).

## Running the Judge

The judge is a standalone CLI step:

```bash
# Heuristic-only mode (no LLM model needed)
arandu judge-transcription results/

# With LLM stage enabled
arandu judge-transcription results/ --validator-model qwen3:14b

# Resume after an interrupted run (default)
arandu judge-transcription results/ --validator-model qwen3:14b
# Force a fresh pass over every record
arandu judge-transcription results/ --validator-model qwen3:14b --rejudge
```

See the [CLI reference](/reference/cli/#judge-transcription) for the full flag list. Recommended workflow:

1. Transcribe (`arandu batch-transcribe …`) — records land with `validation: null`.
2. Judge (`arandu judge-transcription …`) — records get `validation: <JudgePipelineResult>` and `is_valid` is derived from it.
3. Filter / inspect downstream (e.g. `kg`/`cep`/`qa` stages or report dashboards) using `is_valid`.

## Validation Dimensions in Detail

### 1. Script/Charset Match

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

### 2. Repetition Detection

Detects both single-word floods and repeated multi-word phrases.

**Word repetition**: If the most common word exceeds `max_word_repetition_ratio` (default: 15%) of total words, it is flagged.

**Phrase repetition**: Checks 3-gram, 4-gram, and 5-gram patterns. If any phrase appears more than `max_phrase_repetition_count` (default: 4) times, it is flagged with its text coverage ratio.

**Scoring**: Based on the worst repetition ratio found (not the count of issues). Score = `1.0 - worst_ratio`, clamped to [0.0, 1.0]. This ensures "Obrigada" x30 (ratio 1.0) scores 0.0.

**Special case**: Transcriptions shorter than 5 words receive a neutral score of 0.7 with a `very_short_transcription` issue.

**Issue tags**: `high_word_repetition:<word>:<count>`, `repeated_phrase:<text>:<count>`, `very_short_transcription`

### 3. Segment Pattern Analysis

Analyzes Whisper's timestamp segments for anomalies that indicate processing artifacts.

**Uniform intervals**: Flags when 5+ consecutive segments have ~1-second start-time intervals (within `uniform_interval_tolerance` of ±0.1s). This pattern suggests Whisper fell into a timestamp loop.

**Empty segments**: Flags when more than 20% (`max_empty_segment_ratio`) of segments have no text content.

**Scoring**:
- No segments provided → score 1.0 (not penalized)
- Empty segment ratio exceeded → -0.3
- Suspicious uniform intervals → -0.5

**Issue tags**: `suspicious_uniform_intervals:<count>`, `high_empty_segments:<empty>/<total>`

### 4. Content Density

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

Programmatic usage of `TranscriptionJudge` is documented in the [Programmatic Usage](#programmatic-usage) section below.

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

- **Threshold provenance.** The 0.8 / 0.7 defaults started from the rubric design. Empirical calibration against the project's Portuguese corpus is documented in [`docs/research/judge-pipeline-calibration.md`](https://github.com/FredDsR/arandu/blob/main/docs/research/judge-pipeline-calibration.md), including the dual-class audit protocol (30% rejected + 15% admitted, Clopper-Pearson 95% CIs) used to validate them. Re-calibration is needed if the corpus or the validator model changes substantively.

## Configuration Reference

The judge module uses the `ARANDU_JUDGE_` env-var prefix for runtime knobs and per-criterion JSON files for criterion-level tuning.

### Runtime settings (`ARANDU_JUDGE_*`)

| Setting | Type | Default | Env Var |
|---------|------|---------|---------|
| `language` | `str` | `"pt"` | `ARANDU_JUDGE_LANGUAGE` |
| `temperature` | `float` | `0.3` | `ARANDU_JUDGE_TEMPERATURE` |
| `max_tokens` | `int` | `2048` | `ARANDU_JUDGE_MAX_TOKENS` |
| `validator_model` | `str` \| `None` | `None` | `ARANDU_JUDGE_VALIDATOR_MODEL` |
| `validator_provider` | `str` \| `None` | `None` (inferred) | `ARANDU_JUDGE_VALIDATOR_PROVIDER` |
| `validator_base_url` | `str` \| `None` | `None` | `ARANDU_JUDGE_VALIDATOR_BASE_URL` |

When `validator_model` is unset (and not provided via `--validator-model`), `judge-transcription` runs in heuristic-only mode and skips the LLM stage.

### Per-criterion thresholds and detection bounds

Each criterion ships with a JSON config under `prompts/judge/criteria/<name>/config.json` (LLM criteria) and Python defaults baked into `arandu.transcription.criteria.<name>` (heuristic criteria). Detection thresholds (e.g. `max_non_latin_ratio`, `max_word_repetition_ratio`, words-per-minute bounds) are configured per-criterion rather than via a global `ARANDU_QUALITY_*` block. See the criterion source files / JSON configs for the current defaults.

### Example `.env` Configuration

```bash
# Validator model for the LLM stage (omit to run heuristic-only)
ARANDU_JUDGE_VALIDATOR_MODEL=qwen3:14b
# Optional: pin a specific provider / base URL
ARANDU_JUDGE_VALIDATOR_PROVIDER=ollama
ARANDU_LLM_BASE_URL=http://localhost:11434/v1

# Sampling for LLM criteria (low temperature for consistent evaluation)
ARANDU_JUDGE_TEMPERATURE=0.3
ARANDU_JUDGE_MAX_TOKENS=2048

# Language used for criterion prompts and script-match expectations
ARANDU_JUDGE_LANGUAGE=pt
```

## Interpreting Results

### Common Failure Patterns

| Pattern | Failing criterion | Root cause |
|---------|-------------------|------------|
| Japanese text for Portuguese audio | `script_match` | Whisper language detection failure |
| "Obrigada" x30 | `repetition` | Whisper repetition loop |
| Uniform 1-second segments | `segment_quality` | Whisper timestamp artifact |
| 5 words in 60 seconds | `content_length_floor` (short-circuit) | Mostly silence or background noise |
| English narration in PT corpus | `language_drift` (LLM stage) | Whisper transcribed against the wrong language model |
| YouTube intro/outro phrasing | `hallucination_loop` (LLM stage) | Whisper hallucinated training-distribution text |

### Threshold Tuning

Thresholds are **per-criterion**, not global. To make any single failure mode stricter or more lenient, edit the criterion's config rather than chasing a single overall-score cutoff. The overall `passed` outcome flips as soon as *any* criterion's score is below its threshold, so adjusting one criterion can be done without touching the others.

## Programmatic Usage

```python
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.transcription.judge import TranscriptionJudge, build_validator_client

# Heuristic-only mode (no LLM calls)
judge = TranscriptionJudge(language="pt")
result = judge.evaluate_transcription(
    text=record.transcription_text,
    duration_ms=record.duration_milliseconds,
    segments=record.segments or [],
)

# Heuristics + LLM stage
validator = build_validator_client(model_id="qwen3:14b")  # respects ARANDU_LLM_BASE_URL
judge = TranscriptionJudge(language="pt", validator_client=validator)
result = judge.evaluate_transcription(text=..., duration_ms=..., segments=...)

# Persist back into the EnrichedRecord — is_valid is derived from result.passed
record.validation = result
```

## Known Limitations

- **Non-Latin script detection**: Currently only detects CJK (Chinese, Japanese, Korean) as wrong-script for Latin languages. Arabic, Cyrillic, and other scripts are tracked via the generic `non_latin_ratio` threshold but lack dedicated detection.
- **Language probability**: Not used as a scoring dimension because valid Portuguese transcriptions can have `language_probability: 0.0`. Character-level script detection is more reliable.
- **Content density with silence**: Long audio with extended silence produces low wpm but may still be a valid transcription. Consider the context when interpreting density scores.

---

**See also**: [Transcription](/guides/transcription/) | [Configuration](/configuration/) | [Getting Started](/getting-started/)
