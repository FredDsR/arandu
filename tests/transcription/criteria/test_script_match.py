"""Tests for ScriptMatchCriterion."""

from __future__ import annotations

from arandu.shared.judge.schemas import CriterionScore
from arandu.transcription.criteria.script_match import ScriptMatchCriterion


class TestScriptMatchCriterion:
    """Tests for script/charset matching criterion."""

    def test_implements_judge_criterion_protocol(self) -> None:
        """Test that ScriptMatchCriterion satisfies JudgeCriterion protocol."""
        from arandu.shared.judge.criterion import JudgeCriterion

        criterion = ScriptMatchCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_name_and_threshold(self) -> None:
        """Test default name and threshold."""
        criterion = ScriptMatchCriterion()
        assert criterion.name == "script_match"
        assert criterion.threshold == 0.6

    def test_valid_portuguese_text(self) -> None:
        """Test that valid Portuguese text scores well."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="Esta é uma transcrição válida em português com acentuação correta.",
            expected_language="pt",
        )

        assert isinstance(result, CriterionScore)
        assert result.score == 1.0
        assert result.passed is True

    def test_japanese_text_when_expecting_portuguese(self) -> None:
        """Test that Japanese text scores poorly when expecting Portuguese."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="これは日本語のテキストです",
            expected_language="pt",
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "wrong_script:cjk_detected" in result.rationale

    def test_mixed_latin_cjk_text(self) -> None:
        """Test text with both Latin and CJK characters."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="This is some text これは日本語",
            expected_language="pt",
        )

        assert result.score is not None
        assert result.score <= 1.0

    def test_text_with_special_characters(self) -> None:
        """Test that special characters don't break detection."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="Olá! Como está? Tudo bem? #hashtag @mention 123",
            expected_language="pt",
        )

        assert result.score == 1.0
        assert result.passed is True

    def test_no_alphabetic_content(self) -> None:
        """Test text with no alphabetic characters."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="123 456 789 !@# $%^",
            expected_language="pt",
        )

        assert result.score == 0.5
        assert "no_alphabetic_content" in result.rationale

    def test_default_expected_language(self) -> None:
        """Test that default expected_language is 'pt'."""
        criterion = ScriptMatchCriterion()
        result = criterion.evaluate(
            text="Esta é uma transcrição válida em português.",
        )

        assert result.score == 1.0

    def test_custom_max_non_latin_ratio(self) -> None:
        """Test with custom max_non_latin_ratio."""
        criterion = ScriptMatchCriterion(max_non_latin_ratio=0.0)
        # Text with some non-Latin chars mixed in (Greek alpha, etc.)
        # Use default text that's all-Latin — should still pass at 0.0
        result = criterion.evaluate(
            text="Simple Latin text only.",
            expected_language="pt",
        )
        assert result.score == 1.0

    def test_error_returns_criterion_score_with_error(self) -> None:
        """Test that exceptions are caught and returned as error CriterionScore."""
        criterion = ScriptMatchCriterion()
        # Pass a non-string to trigger an error
        result = criterion.evaluate(text=12345, expected_language="pt")

        assert result.score is None
        assert result.error is not None
        assert result.passed is False
