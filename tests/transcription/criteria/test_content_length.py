"""Tests for ContentLengthFloorCriterion."""

from __future__ import annotations

from arandu.shared.judge.criterion import JudgeCriterion
from arandu.shared.judge.schemas import CriterionScore
from arandu.transcription.criteria.content_length import ContentLengthFloorCriterion


class TestContentLengthFloorCriterion:
    """Tests for the binary length-floor heuristic gate."""

    def test_implements_judge_criterion_protocol(self) -> None:
        criterion = ContentLengthFloorCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_default_name_and_threshold(self) -> None:
        criterion = ContentLengthFloorCriterion()
        assert criterion.name == "content_length_floor"
        assert criterion.threshold == 0.5
        assert criterion.min_chars == 200
        assert criterion.min_words == 30

    def test_passes_when_both_floors_met(self) -> None:
        criterion = ContentLengthFloorCriterion()
        text = "palavra " * 40  # 40 words, > 200 chars
        result = criterion.evaluate(text=text)

        assert isinstance(result, CriterionScore)
        assert result.score == 1.0
        assert result.passed is True
        assert "length_within_floor" in result.rationale

    def test_rejects_silence_filler_thank_you(self) -> None:
        """Reproduces the §3.4 false negative: 'Thank you.' (8.4 s, 14 wpm)."""
        criterion = ContentLengthFloorCriterion()
        result = criterion.evaluate(text="Thank you.")

        assert result.score == 0.0
        assert result.passed is False
        assert "text_too_short" in result.rationale
        assert "10_chars" in result.rationale
        assert "2_words" in result.rationale

    def test_rejects_empty_text(self) -> None:
        criterion = ContentLengthFloorCriterion()
        result = criterion.evaluate(text="")

        assert result.score == 0.0
        assert result.passed is False

    def test_rejects_whitespace_only(self) -> None:
        criterion = ContentLengthFloorCriterion()
        result = criterion.evaluate(text="   \n\t   ")

        assert result.score == 0.0
        assert result.passed is False

    def test_strips_before_measuring(self) -> None:
        criterion = ContentLengthFloorCriterion(min_chars=10, min_words=2)
        # Padded with whitespace but content meets both floors after strip.
        result = criterion.evaluate(text="   hello world how are    ")
        assert result.score == 1.0

    def test_rejects_when_chars_pass_but_words_fail(self) -> None:
        """A single 250-char token fails the word floor even though chars pass."""
        criterion = ContentLengthFloorCriterion()
        text = "a" * 250
        result = criterion.evaluate(text=text)

        assert result.score == 0.0
        assert "text_too_short" in result.rationale

    def test_rejects_when_words_pass_but_chars_fail(self) -> None:
        """30 single-letter words clear the word floor but not the char floor."""
        criterion = ContentLengthFloorCriterion()
        text = " ".join(["a"] * 30)  # 30 words, 59 chars
        result = criterion.evaluate(text=text)

        assert result.score == 0.0
        assert "text_too_short" in result.rationale

    def test_custom_floors(self) -> None:
        criterion = ContentLengthFloorCriterion(min_chars=10, min_words=3)
        assert criterion.evaluate(text="hello world today").score == 1.0
        assert criterion.evaluate(text="hi there").score == 0.0  # 2 words

    def test_error_returns_criterion_score_with_error(self) -> None:
        criterion = ContentLengthFloorCriterion()
        result = criterion.evaluate(text=12345)  # type: ignore[arg-type]

        assert result.score is None
        assert result.error is not None
        assert result.passed is False
