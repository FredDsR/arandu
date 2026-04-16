"""Tests for RepetitionCriterion."""

from __future__ import annotations

from arandu.shared.judge.schemas import CriterionScore
from arandu.transcription.criteria.repetition import RepetitionCriterion


class TestRepetitionCriterion:
    """Tests for word and phrase repetition criterion."""

    def test_implements_judge_criterion_protocol(self) -> None:
        """Test that RepetitionCriterion satisfies JudgeCriterion protocol."""
        from arandu.shared.judge.criterion import JudgeCriterion

        criterion = RepetitionCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_name_and_threshold(self) -> None:
        """Test default name and threshold."""
        criterion = RepetitionCriterion()
        assert criterion.name == "repetition"
        assert criterion.threshold == 0.5

    def test_no_repetition(self) -> None:
        """Test text without excessive repetition."""
        criterion = RepetitionCriterion()
        result = criterion.evaluate(
            text="This is a normal transcription with varied vocabulary and no repeated patterns.",
        )

        assert isinstance(result, CriterionScore)
        assert result.score is not None
        assert result.score >= 0.8
        assert result.passed is True

    def test_single_word_flood(self) -> None:
        """Test text with a single word repeated many times (Obrigada x30 case)."""
        criterion = RepetitionCriterion()
        result = criterion.evaluate(text=" ".join(["Obrigada"] * 30))

        assert result.score is not None
        assert result.score < 0.2
        assert result.passed is False
        assert "high_word_repetition" in result.rationale.lower()

    def test_phrase_repetition(self) -> None:
        """Test text with repeated multi-word phrases."""
        criterion = RepetitionCriterion()
        phrase = "Eu não falei muito no chão mas"
        result = criterion.evaluate(text=" ".join([phrase] * 5))

        assert result.score is not None
        assert result.score < 0.5
        assert "repeated_phrase" in result.rationale

    def test_short_transcription(self) -> None:
        """Test very short transcription (< 5 words)."""
        criterion = RepetitionCriterion()
        result = criterion.evaluate(text="Hi there")

        assert result.score == 0.7
        assert "very_short_transcription" in result.rationale

    def test_mild_repetition_acceptable(self) -> None:
        """Test that mild repetition is acceptable."""
        criterion = RepetitionCriterion()
        text = "I went to the store and I bought some milk. Then I went home and I made coffee."
        result = criterion.evaluate(text=text)

        assert result.score is not None
        assert result.score >= 0.7
        assert result.passed is True

    def test_custom_max_word_repetition_ratio(self) -> None:
        """Test with custom max_word_repetition_ratio."""
        # Very strict threshold
        criterion = RepetitionCriterion(max_word_repetition_ratio=0.05)
        text = "I went to the store and I bought some milk. Then I went home."
        result = criterion.evaluate(text=text)
        # "i" appears 3 times in 13 words = 0.23 ratio, exceeds 0.05
        assert "high_word_repetition" in result.rationale

    def test_custom_max_phrase_repetition_count(self) -> None:
        """Test with custom max_phrase_repetition_count."""
        criterion = RepetitionCriterion(max_phrase_repetition_count=2)
        phrase = "the big dog"
        text = f"{phrase} went home. {phrase} was happy. {phrase} ate food."
        result = criterion.evaluate(text=text)
        assert "repeated_phrase" in result.rationale

    def test_error_returns_criterion_score_with_error(self) -> None:
        """Test that exceptions are caught and returned as error CriterionScore."""
        criterion = RepetitionCriterion()
        result = criterion.evaluate(text=12345)

        assert result.score is None
        assert result.error is not None
        assert result.passed is False
