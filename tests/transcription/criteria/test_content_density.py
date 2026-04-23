"""Tests for ContentDensityCriterion."""

from __future__ import annotations

from arandu.shared.judge.schemas import CriterionScore
from arandu.transcription.criteria.content_density import ContentDensityCriterion


class TestContentDensityCriterion:
    """Tests for content density (words per minute) criterion."""

    def test_implements_judge_criterion_protocol(self) -> None:
        """Test that ContentDensityCriterion satisfies JudgeCriterion protocol."""
        from arandu.shared.judge.criterion import JudgeCriterion

        criterion = ContentDensityCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_name_and_threshold(self) -> None:
        """Test default name and threshold."""
        criterion = ContentDensityCriterion()
        assert criterion.name == "content_density"
        assert criterion.threshold == 0.4

    def test_normal_word_density(self) -> None:
        """Test normal speaking rate (around 150 wpm)."""
        criterion = ContentDensityCriterion()
        text = " ".join(["word"] * 150)  # 150 words
        result = criterion.evaluate(text=text, duration_ms=60_000)  # 1 minute

        assert isinstance(result, CriterionScore)
        assert result.score == 1.0
        assert result.passed is True

    def test_too_sparse(self) -> None:
        """Test transcription with too few words per minute."""
        criterion = ContentDensityCriterion()
        result = criterion.evaluate(text="just a few words", duration_ms=60_000)

        assert result.score is not None
        assert result.score < 1.0
        assert "low_content_density" in result.rationale

    def test_too_dense(self) -> None:
        """Test transcription with too many words per minute."""
        criterion = ContentDensityCriterion()
        text = " ".join(["word"] * 400)  # 400 words
        result = criterion.evaluate(text=text, duration_ms=60_000)

        assert result.score is not None
        assert result.score < 1.0
        assert "high_content_density" in result.rationale

    def test_duration_none_returns_neutral(self) -> None:
        """Test that duration_ms=None returns neutral score."""
        criterion = ContentDensityCriterion()
        result = criterion.evaluate(text="Some transcription text", duration_ms=None)

        assert result.score == 0.5
        assert "duration_unknown" in result.rationale

    def test_invalid_duration(self) -> None:
        """Test that invalid duration (0 or negative) is penalized."""
        criterion = ContentDensityCriterion()
        result = criterion.evaluate(text="Some text", duration_ms=0)

        assert result.score == 0.3
        assert "invalid_duration" in result.rationale

    def test_negative_duration(self) -> None:
        """Test that negative duration is penalized."""
        criterion = ContentDensityCriterion()
        result = criterion.evaluate(text="Some text", duration_ms=-1000)

        assert result.score == 0.3
        assert "invalid_duration" in result.rationale

    def test_custom_min_words_per_minute(self) -> None:
        """Test with custom min_words_per_minute."""
        criterion = ContentDensityCriterion(min_words_per_minute=100.0)
        # 50 wpm is below 100 threshold
        text = " ".join(["word"] * 50)
        result = criterion.evaluate(text=text, duration_ms=60_000)
        assert "low_content_density" in result.rationale

    def test_custom_max_words_per_minute(self) -> None:
        """Test with custom max_words_per_minute."""
        criterion = ContentDensityCriterion(max_words_per_minute=200.0)
        # 250 wpm exceeds 200 threshold
        text = " ".join(["word"] * 250)
        result = criterion.evaluate(text=text, duration_ms=60_000)
        assert "high_content_density" in result.rationale

    def test_error_returns_criterion_score_with_error(self) -> None:
        """Test that exceptions are caught and returned as error CriterionScore."""
        criterion = ContentDensityCriterion()
        result = criterion.evaluate(text=12345, duration_ms=60_000)

        assert result.score is None
        assert result.error is not None
        assert result.passed is False
