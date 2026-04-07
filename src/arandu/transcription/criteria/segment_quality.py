"""Segment quality criterion for transcription quality validation.

Analyzes segment timestamp patterns for anomalies like uniform 1-second
intervals (Whisper hallucination) and empty segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arandu.shared.judge.criterion import HeuristicCriterion
from arandu.shared.judge.schemas import CriterionScore

if TYPE_CHECKING:
    from arandu.shared.schemas import TranscriptionSegment


class SegmentQualityCriterion(HeuristicCriterion):
    """Evaluate transcription segment patterns for anomalies.

    Detects suspicious patterns like many consecutive segments with
    uniform 1-second intervals and empty segments (no text).

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass.
        max_empty_segment_ratio: Maximum ratio of empty segments before flagging.
        suspicious_uniform_intervals: Number of consecutive uniform intervals to flag.
        uniform_interval_tolerance: Tolerance for detecting uniform 1-second intervals.
    """

    name: str = "segment_quality"
    threshold: float = 0.4

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        max_empty_segment_ratio: float = 0.2,
        suspicious_uniform_intervals: int = 5,
        uniform_interval_tolerance: float = 0.1,
    ) -> None:
        """Initialize segment quality criterion.

        Args:
            threshold: Minimum score to pass.
            max_empty_segment_ratio: Maximum ratio of empty segments before flagging.
            suspicious_uniform_intervals: Number of consecutive uniform 1-second
                intervals to flag.
            uniform_interval_tolerance: Tolerance (seconds) for detecting uniform
                1-second intervals.
        """
        self.threshold = threshold
        self.max_empty_segment_ratio = max_empty_segment_ratio
        self.suspicious_uniform_intervals = suspicious_uniform_intervals
        self.uniform_interval_tolerance = uniform_interval_tolerance

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        """Evaluate segment patterns for anomalies.

        Overrides the base evaluate to handle the empty-segments edge case
        before delegating to _check.

        Args:
            **kwargs: Must contain ``segments`` (list of TranscriptionSegment).

        Returns:
            CriterionScore with score 0.0-1.0 and rationale.
        """
        segments: list[TranscriptionSegment] = kwargs.get("segments", [])
        if not segments:
            return CriterionScore(
                score=0.5,
                threshold=self.threshold,
                rationale="No segments available for analysis.",
            )
        return super().evaluate(**kwargs)

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Check segment patterns for anomalies.

        Args:
            **kwargs: Must contain ``segments`` (list of TranscriptionSegment).

        Returns:
            Tuple of (score, rationale).
        """
        segments: list[TranscriptionSegment] = kwargs["segments"]
        score, issues = self._check_segment_patterns(segments)
        rationale = "; ".join(issues) if issues else "Segment patterns are within normal range."
        return score, rationale

    def _check_segment_patterns(
        self, segments: list[TranscriptionSegment]
    ) -> tuple[float, list[str]]:
        """Analyze segment timestamps for anomalies.

        Args:
            segments: List of transcription segments with timestamps.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues: list[str] = []

        # Check for empty segments
        empty_count = sum(1 for seg in segments if not seg.text.strip())
        empty_ratio = empty_count / len(segments)
        if empty_ratio > self.max_empty_segment_ratio:
            issues.append(f"high_empty_segments:{empty_count}/{len(segments)}")

        # Check for suspicious uniform intervals
        consecutive_uniform = 0
        max_consecutive_uniform = 0

        tolerance = self.uniform_interval_tolerance
        lower_bound = 1.0 - tolerance
        upper_bound = 1.0 + tolerance

        for i in range(len(segments) - 1):
            duration = segments[i + 1].start - segments[i].start
            if lower_bound <= duration <= upper_bound:
                consecutive_uniform += 1
                max_consecutive_uniform = max(max_consecutive_uniform, consecutive_uniform)
            else:
                consecutive_uniform = 0

        exceeds_threshold = max_consecutive_uniform >= self.suspicious_uniform_intervals
        if exceeds_threshold:
            issues.append(f"suspicious_uniform_intervals:{max_consecutive_uniform}")

        if not issues:
            return 1.0, []

        # Penalize based on severity
        score = 1.0
        if empty_ratio > self.max_empty_segment_ratio:
            score -= 0.3
        if exceeds_threshold:
            score -= 0.5

        return max(0.0, score), issues
