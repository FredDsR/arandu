"""Content density criterion for transcription quality validation.

Checks words per minute ratio to detect abnormally sparse or dense
transcriptions, which indicate Whisper hallucination or truncation.
"""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion


class ContentDensityCriterion(HeuristicCriterion):
    """Evaluate whether transcription has a reasonable words-per-minute ratio.

    Guards against duration_ms=None (InputRecord allows it). When duration
    is unavailable, returns a neutral score so the check doesn't skew results.

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass.
        min_words_per_minute: Minimum words per minute threshold.
        max_words_per_minute: Maximum words per minute threshold.
    """

    name: str = "content_density"
    threshold: float = 0.4

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        min_words_per_minute: float = 30.0,
        max_words_per_minute: float = 300.0,
    ) -> None:
        """Initialize content density criterion.

        Args:
            threshold: Minimum score to pass.
            min_words_per_minute: Minimum words per minute threshold.
            max_words_per_minute: Maximum words per minute threshold.
        """
        self.threshold = threshold
        self.min_words_per_minute = min_words_per_minute
        self.max_words_per_minute = max_words_per_minute

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Check content density of transcription text.

        Args:
            **kwargs: Must contain ``text`` (str) and ``duration_ms``
                (int | None).

        Returns:
            Tuple of (score, rationale).
        """
        text: str = kwargs["text"]
        duration_ms: int | None = kwargs["duration_ms"]
        score, issues = self._check_content_density(text, duration_ms)
        rationale = "; ".join(issues) if issues else "Content density is within normal range."
        return score, rationale

    def _check_content_density(self, text: str, duration_ms: int | None) -> tuple[float, list[str]]:
        """Check words per minute ratio.

        Args:
            text: Transcription text to check.
            duration_ms: Duration in milliseconds, may be None.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues: list[str] = []
        words = text.split()

        if duration_ms is None:
            return 0.5, ["duration_unknown:neutral_score"]

        if duration_ms <= 0:
            return 0.3, ["invalid_duration"]

        duration_min = duration_ms / 60_000
        wpm = len(words) / duration_min

        if wpm < self.min_words_per_minute:
            issues.append(f"low_content_density:{wpm:.1f}_wpm")
            return max(0.0, wpm / self.min_words_per_minute), issues

        if wpm > self.max_words_per_minute:
            issues.append(f"high_content_density:{wpm:.1f}_wpm")
            return max(0.0, 2.0 - wpm / self.max_words_per_minute), issues

        return 1.0, []
