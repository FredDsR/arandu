"""Transcription quality validation using heuristic checks.

Validates Whisper transcription output for common failure modes:
- Wrong language/script (e.g., Japanese when expecting Portuguese)
- Repeated words and phrases
- Suspicious segment patterns (uniform timestamps, empty segments)
- Abnormal content density (words per minute)
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from arandu.config import TranscriptionQualityConfig, get_transcription_quality_config
from arandu.schemas import EnrichedRecord, TranscriptionQualityScore, TranscriptionSegment

# Languages that use Latin script
_LATIN_LANGS = frozenset({"pt", "en", "es", "fr", "de", "it"})


class TranscriptionValidator:
    """Validate transcription quality using heuristics.

    Evaluates transcriptions on four dimensions:
    1. Script/charset match - Text uses expected character set
    2. Repetition - Detects excessive word/phrase repetition
    3. Segment quality - Analyzes timestamp patterns for anomalies
    4. Content density - Checks words per minute ratio

    Each dimension produces a score (0.0-1.0) and list of issues.
    The overall score is a weighted average based on config weights.
    """

    def __init__(self, config: TranscriptionQualityConfig | None = None) -> None:
        """Initialize validator with configuration.

        Args:
            config: Quality validation config. If None, loads from env vars.
        """
        self.config = config or get_transcription_quality_config()

    def validate(self, record: EnrichedRecord) -> TranscriptionQualityScore:
        """Validate single transcription record.

        Args:
            record: EnrichedRecord with transcription text and segments.

        Returns:
            TranscriptionQualityScore with individual and overall scores.
        """
        text = record.transcription_text
        segments = record.segments or []
        duration_ms = record.duration_milliseconds

        # Run all checks
        script_score, script_issues = self._check_script_match(text, self.config.expected_language)
        repetition_score, repetition_issues = self._check_repetition(text)
        segment_score, segment_issues = self._check_segment_patterns(segments)
        density_score, density_issues = self._check_content_density(text, duration_ms)

        # Compute weighted overall score
        overall_score = (
            script_score * self.config.script_match_weight
            + repetition_score * self.config.repetition_weight
            + segment_score * self.config.segment_quality_weight
            + density_score * self.config.content_density_weight
        )

        # Aggregate all issues
        all_issues = script_issues + repetition_issues + segment_issues + density_issues

        # Generate rationale if issues detected
        rationale = None
        if all_issues:
            rationale = f"Quality score {overall_score:.2f}: {len(all_issues)} issues detected"

        return TranscriptionQualityScore(
            script_match_score=script_score,
            repetition_score=repetition_score,
            segment_quality_score=segment_score,
            content_density_score=density_score,
            overall_score=overall_score,
            issues_detected=all_issues,
            quality_rationale=rationale,
        )

    def _check_script_match(self, text: str, expected_lang: str) -> tuple[float, list[str]]:
        """Check if text uses expected character set.

        Uses unicodedata.name() for robust script detection rather than
        hardcoded Unicode ranges, which miss Latin Extended-B and combining
        diacriticals used in Portuguese.

        Args:
            text: Transcription text to check.
            expected_lang: Expected language code (e.g., 'pt', 'en').

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        # For Portuguese/English/Romance languages, expect Latin characters
        if expected_lang in _LATIN_LANGS:
            latin_chars = 0
            cjk_chars = 0
            total_alpha = 0

            for c in text:
                if not c.isalpha():
                    continue
                total_alpha += 1
                # Use Unicode script property via character name
                name = unicodedata.name(c, "")
                if "LATIN" in name or "COMBINING" in name:
                    latin_chars += 1
                elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
                    cjk_chars += 1

            if total_alpha == 0:
                return 0.5, ["no_alphabetic_content"]

            non_latin_ratio = (total_alpha - latin_chars) / total_alpha

            if cjk_chars > total_alpha * 0.5:
                # More than 50% CJK characters = wrong language
                return 0.0, [f"wrong_script:cjk_detected:{cjk_chars}_chars"]

            if non_latin_ratio > self.config.max_non_latin_ratio:
                return 0.2, [f"high_non_latin_ratio:{non_latin_ratio:.2f}"]

        return 1.0, []

    def _check_repetition(self, text: str) -> tuple[float, list[str]]:
        """Detect repeated words and phrases.

        Scores based on the WORST repetition ratio found, not by counting
        issue strings. This ensures a single extreme case (e.g., "Obrigada" x30)
        scores worse than several mild repetitions.

        Args:
            text: Transcription text to check.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues = []
        words = text.lower().split()

        if len(words) < 5:
            return 0.7, ["very_short_transcription"]

        worst_ratio = 0.0

        # Word repetition
        word_counts = Counter(words)
        most_common, count = word_counts.most_common(1)[0]
        word_ratio = count / len(words)

        if word_ratio > self.config.max_word_repetition_ratio:
            issues.append(f"high_word_repetition:{most_common}:{count}")
            worst_ratio = max(worst_ratio, word_ratio)

        # Phrase repetition (3+ consecutive words appearing multiple times)
        for n in [3, 4, 5]:
            if len(words) < n:
                continue
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            for phrase, cnt in ngram_counts.most_common(3):
                if cnt >= self.config.max_phrase_repetition_count:
                    # Ratio: how much of the text is covered by this repeated phrase
                    phrase_coverage = (cnt * n) / len(words)
                    issues.append(f"repeated_phrase:{phrase[:25]}:{cnt}")
                    worst_ratio = max(worst_ratio, phrase_coverage)

        # Score inversely proportional to worst repetition found
        # worst_ratio=0.0 → score 1.0; worst_ratio>=1.0 → score 0.0
        score = max(0.0, 1.0 - worst_ratio)
        return score, issues

    def _check_segment_patterns(
        self, segments: list[TranscriptionSegment]
    ) -> tuple[float, list[str]]:
        """Analyze segment timestamps for anomalies.

        Detects suspicious patterns like:
        - Many consecutive segments with uniform 1-second intervals
        - Empty segments (no text)

        Args:
            segments: List of transcription segments with timestamps.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues = []

        if not segments:
            return 1.0, []

        # Check for empty segments
        empty_count = sum(1 for seg in segments if not seg.text.strip())
        empty_ratio = empty_count / len(segments)
        if empty_ratio > self.config.max_empty_segment_ratio:
            issues.append(f"high_empty_segments:{empty_count}/{len(segments)}")

        # Check for suspicious uniform intervals
        consecutive_uniform = 0
        max_consecutive_uniform = 0

        # Calculate tolerance bounds for 1-second intervals
        tolerance = self.config.uniform_interval_tolerance
        lower_bound = 1.0 - tolerance
        upper_bound = 1.0 + tolerance

        for i in range(len(segments) - 1):
            duration = segments[i + 1].start - segments[i].start
            # Check if duration is approximately 1.0 second (configurable tolerance)
            if lower_bound <= duration <= upper_bound:
                consecutive_uniform += 1
                max_consecutive_uniform = max(max_consecutive_uniform, consecutive_uniform)
            else:
                consecutive_uniform = 0

        # Check if we exceeded the suspicious interval threshold
        exceeds_threshold = max_consecutive_uniform >= self.config.suspicious_uniform_intervals
        if exceeds_threshold:
            issues.append(f"suspicious_uniform_intervals:{max_consecutive_uniform}")

        # Calculate score based on issues found
        if not issues:
            return 1.0, []

        # Penalize based on severity
        score = 1.0
        if empty_ratio > self.config.max_empty_segment_ratio:
            score -= 0.3
        if exceeds_threshold:
            score -= 0.5

        return max(0.0, score), issues

    def _check_content_density(self, text: str, duration_ms: int | None) -> tuple[float, list[str]]:
        """Check words per minute ratio.

        Guards against duration_ms=None (InputRecord allows it). When duration
        is unavailable, returns a neutral score so the check doesn't skew results.

        Args:
            text: Transcription text to check.
            duration_ms: Duration in milliseconds, may be None.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues = []
        words = text.split()

        if duration_ms is None:
            return 0.5, ["duration_unknown:neutral_score"]

        if duration_ms <= 0:
            return 0.3, ["invalid_duration"]

        duration_min = duration_ms / 60_000
        wpm = len(words) / duration_min

        if wpm < self.config.min_words_per_minute:
            issues.append(f"low_content_density:{wpm:.1f}_wpm")
            # Scale: 0 wpm → 0.0, min_wpm → 1.0
            return max(0.0, wpm / self.config.min_words_per_minute), issues

        if wpm > self.config.max_words_per_minute:
            issues.append(f"high_content_density:{wpm:.1f}_wpm")
            # Scale: max_wpm → 1.0, 2x max_wpm → 0.0
            return max(0.0, 2.0 - wpm / self.config.max_words_per_minute), issues

        return 1.0, []


def validate_enriched_record(
    record: EnrichedRecord,
    config: TranscriptionQualityConfig | None = None,
    *,
    validator: TranscriptionValidator | None = None,
) -> EnrichedRecord:
    """Validate and mutate record in-place with quality scores.

    Sets record.transcription_quality and record.is_valid based on
    validation results and configured quality threshold.

    Args:
        record: EnrichedRecord to validate and update.
        config: Quality validation config. If None, loads from env vars.
        validator: Optional pre-instantiated validator. If provided, config is ignored.

    Returns:
        The same record (mutated in-place) for chaining.
    """
    if validator is None:
        if config is None:
            config = get_transcription_quality_config()

        if not config.enabled:
            # If validation disabled, mark as valid and skip
            record.is_valid = True
            return record

        validator = TranscriptionValidator(config)

    quality_score = validator.validate(record)

    record.transcription_quality = quality_score
    record.is_valid = quality_score.overall_score >= validator.config.quality_threshold

    return record


def get_quality_issues(record: EnrichedRecord) -> list[str] | None:
    """Get quality issues from a validated record.

    Args:
        record: EnrichedRecord that has been validated.

    Returns:
        List of quality issues if validation failed and quality score exists,
        None otherwise.
    """
    if record.is_valid is False and record.transcription_quality:
        return record.transcription_quality.issues_detected
    return None
