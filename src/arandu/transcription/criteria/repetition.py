"""Repetition detection criterion for transcription quality validation.

Detects excessive word and phrase repetition in transcription text,
a common Whisper failure mode (e.g., "Obrigada" repeated 30 times).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion


class RepetitionCriterion(HeuristicCriterion):
    """Evaluate whether transcription text contains excessive repetition.

    Scores based on the WORST repetition ratio found, not by counting
    issue strings. This ensures a single extreme case (e.g., "Obrigada" x30)
    scores worse than several mild repetitions.

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass.
        max_word_repetition_ratio: Maximum ratio of most repeated word.
        max_phrase_repetition_count: Maximum allowed repetitions of same phrase.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        max_word_repetition_ratio: float = 0.15,
        max_phrase_repetition_count: int = 4,
    ) -> None:
        """Initialize repetition criterion.

        Args:
            threshold: Minimum score to pass.
            max_word_repetition_ratio: Maximum ratio of most repeated word.
            max_phrase_repetition_count: Maximum allowed repetitions of same phrase.
        """
        super().__init__(name="repetition", threshold=threshold)
        self.max_word_repetition_ratio = max_word_repetition_ratio
        self.max_phrase_repetition_count = max_phrase_repetition_count

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Check whether text contains excessive repetition.

        Args:
            **kwargs: Must contain ``text`` (str).

        Returns:
            Tuple of (score, rationale).
        """
        text: str = kwargs["text"]
        score, issues = self._check_repetition(text)
        rationale = "; ".join(issues) if issues else "No excessive repetition detected."
        return score, rationale

    def _check_repetition(self, text: str) -> tuple[float, list[str]]:
        """Detect repeated words and phrases.

        Args:
            text: Transcription text to check.

        Returns:
            Tuple of (score, issues) where score is 0.0-1.0.
        """
        issues: list[str] = []
        words = text.lower().split()

        if len(words) < 5:
            return 0.7, ["very_short_transcription"]

        worst_ratio = 0.0

        # Word repetition
        word_counts = Counter(words)
        most_common, count = word_counts.most_common(1)[0]
        word_ratio = count / len(words)

        if word_ratio > self.max_word_repetition_ratio:
            issues.append(f"high_word_repetition:{most_common}:{count}")
            worst_ratio = max(worst_ratio, word_ratio)

        # Phrase repetition (3+ consecutive words appearing multiple times)
        for n in [3, 4, 5]:
            if len(words) < n:
                continue
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            for phrase, cnt in ngram_counts.most_common(3):
                if cnt >= self.max_phrase_repetition_count:
                    phrase_coverage = (cnt * n) / len(words)
                    issues.append(f"repeated_phrase:{phrase[:25]}:{cnt}")
                    worst_ratio = max(worst_ratio, phrase_coverage)

        # Score inversely proportional to worst repetition found
        score = max(0.0, 1.0 - worst_ratio)
        return score, issues
