"""Content-length floor criterion for transcription quality validation.

Hard-rejects records whose transcription text is too short to carry
extractable content. Runs first in the heuristic stage so very short
records short-circuit the pipeline before any other criterion or LLM
call.

Closes the silence-filler calibration gap documented in
``docs/research/judge-pipeline-calibration.md`` §4.5: ``content_density``
scales linearly and admits records like 8.4 s / "Thank you." (14 wpm,
score 0.475 vs threshold 0.40); a non-scaled length floor catches them
deterministically.
"""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion


class ContentLengthFloorCriterion(HeuristicCriterion):
    """Reject transcriptions below a minimum length floor.

    Binary criterion: scores ``1.0`` when text passes both the character
    and word floors, ``0.0`` otherwise. There is no partial credit — the
    point of this gate is to remove records that cannot carry extractable
    content regardless of any other quality signal.

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass (``0.5`` — binary gate).
        min_chars: Minimum character count after stripping.
        min_words: Minimum whitespace-tokenised word count.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_chars: int = 200,
        min_words: int = 30,
    ) -> None:
        """Initialise the length-floor criterion.

        Args:
            threshold: Minimum score to pass; default ``0.5`` makes the
                criterion act as a binary 0/1 gate.
            min_chars: Minimum character count (default ``200``, matching
                the previous QA-generation hard floor).
            min_words: Minimum word count (default ``30``); rejects records
                that meet ``min_chars`` only by repeating a single token.
        """
        super().__init__(name="content_length_floor", threshold=threshold)
        self.min_chars = min_chars
        self.min_words = min_words

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Check whether the transcription clears the length floor.

        Args:
            **kwargs: Must contain ``text`` (str).

        Returns:
            Tuple of (score, rationale). ``1.0`` if both floors are met,
            ``0.0`` otherwise.
        """
        text: str = kwargs["text"]
        stripped = text.strip()
        n_chars = len(stripped)
        n_words = len(stripped.split())

        if n_chars < self.min_chars or n_words < self.min_words:
            return 0.0, (
                f"text_too_short:{n_chars}_chars_{n_words}_words "
                f"(floor: {self.min_chars}_chars/{self.min_words}_words)"
            )
        return 1.0, "length_within_floor"
