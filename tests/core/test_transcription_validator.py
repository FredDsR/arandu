"""Tests for transcription quality validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gtranscriber.config import TranscriptionQualityConfig
from gtranscriber.core.transcription_validator import (
    TranscriptionValidator,
    validate_enriched_record,
)
from gtranscriber.schemas import EnrichedRecord, TranscriptionSegment


class TestTranscriptionQualityConfig:
    """Tests for TranscriptionQualityConfig."""

    def test_default_initialization(self) -> None:
        """Test default configuration initialization."""
        config = TranscriptionQualityConfig()

        assert config.enabled is True
        assert config.quality_threshold == 0.5
        assert config.expected_language == "pt"
        assert config.script_match_weight == 0.35
        assert config.repetition_weight == 0.30
        assert config.segment_quality_weight == 0.20
        assert config.content_density_weight == 0.15

    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to 1.0."""
        config = TranscriptionQualityConfig()

        total = (
            config.script_match_weight
            + config.repetition_weight
            + config.segment_quality_weight
            + config.content_density_weight
        )

        assert 0.99 <= total <= 1.01

    def test_invalid_weights_raise_error(self) -> None:
        """Test that invalid weights raise ValidationError."""
        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            TranscriptionQualityConfig(
                script_match_weight=0.5,
                repetition_weight=0.3,
                segment_quality_weight=0.1,
                content_density_weight=0.05,  # Total = 0.95, not 1.0
            )

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_QUALITY_ENABLED", "false")
        monkeypatch.setenv("GTRANSCRIBER_QUALITY_QUALITY_THRESHOLD", "0.7")
        monkeypatch.setenv("GTRANSCRIBER_QUALITY_EXPECTED_LANGUAGE", "en")

        config = TranscriptionQualityConfig()

        assert config.enabled is False
        assert config.quality_threshold == 0.7
        assert config.expected_language == "en"


class TestTranscriptionValidator:
    """Tests for TranscriptionValidator."""

    @pytest.fixture
    def validator(self) -> TranscriptionValidator:
        """Create a validator with default config."""
        return TranscriptionValidator()

    @pytest.fixture
    def sample_record(self, sample_enriched_record_data: dict) -> EnrichedRecord:
        """Create a sample EnrichedRecord for testing."""
        return EnrichedRecord(**sample_enriched_record_data)

    def test_validator_initialization(self, validator: TranscriptionValidator) -> None:
        """Test validator initialization."""
        assert validator.config is not None
        assert isinstance(validator.config, TranscriptionQualityConfig)

    def test_validator_with_custom_config(self) -> None:
        """Test validator initialization with custom config."""
        config = TranscriptionQualityConfig(quality_threshold=0.7)
        validator = TranscriptionValidator(config)
        assert validator.config.quality_threshold == 0.7

    def test_validate_returns_quality_score(
        self, validator: TranscriptionValidator, sample_record: EnrichedRecord
    ) -> None:
        """Test that validate returns a TranscriptionQualityScore."""
        score = validator.validate(sample_record)

        assert score.script_match_score >= 0.0
        assert score.script_match_score <= 1.0
        assert score.repetition_score >= 0.0
        assert score.repetition_score <= 1.0
        assert score.segment_quality_score >= 0.0
        assert score.segment_quality_score <= 1.0
        assert score.content_density_score >= 0.0
        assert score.content_density_score <= 1.0
        assert score.overall_score >= 0.0
        assert score.overall_score <= 1.0
        assert isinstance(score.issues_detected, list)


class TestScriptMatch:
    """Tests for script/charset matching detection."""

    @pytest.fixture
    def validator(self) -> TranscriptionValidator:
        """Create a validator for Latin languages."""
        config = TranscriptionQualityConfig(expected_language="pt")
        return TranscriptionValidator(config)

    def test_valid_portuguese_text(self, validator: TranscriptionValidator) -> None:
        """Test that valid Portuguese text scores well."""
        text = "Esta é uma transcrição válida em português com acentuação correta."
        score, issues = validator._check_script_match(text, "pt")

        assert score == 1.0
        assert len(issues) == 0

    def test_japanese_text_when_expecting_portuguese(
        self, validator: TranscriptionValidator
    ) -> None:
        """Test that Japanese text scores poorly when expecting Portuguese."""
        text = "これは日本語のテキストです"
        score, issues = validator._check_script_match(text, "pt")

        assert score == 0.0
        assert any("wrong_script:cjk_detected" in issue for issue in issues)

    def test_mixed_latin_cjk_text(self, validator: TranscriptionValidator) -> None:
        """Test text with both Latin and CJK characters."""
        text = "This is some text これは日本語"
        score, _issues = validator._check_script_match(text, "pt")

        # Should detect CJK presence if more than 50% CJK
        assert score <= 1.0

    def test_text_with_special_characters(self, validator: TranscriptionValidator) -> None:
        """Test that special characters don't break detection."""
        text = "Olá! Como está? Tudo bem? #hashtag @mention 123"
        score, issues = validator._check_script_match(text, "pt")

        assert score == 1.0
        assert len(issues) == 0

    def test_no_alphabetic_content(self, validator: TranscriptionValidator) -> None:
        """Test text with no alphabetic characters."""
        text = "123 456 789 !@# $%^"
        score, issues = validator._check_script_match(text, "pt")

        assert score == 0.5
        assert "no_alphabetic_content" in issues


class TestRepetitionDetection:
    """Tests for word and phrase repetition detection."""

    @pytest.fixture
    def validator(self) -> TranscriptionValidator:
        """Create a validator with default config."""
        return TranscriptionValidator()

    def test_no_repetition(self, validator: TranscriptionValidator) -> None:
        """Test text without excessive repetition."""
        text = "This is a normal transcription with varied vocabulary and no repeated patterns."
        score, issues = validator._check_repetition(text)

        assert score >= 0.8
        assert len(issues) == 0

    def test_single_word_flood(self, validator: TranscriptionValidator) -> None:
        """Test text with a single word repeated many times (Obrigada x30 case)."""
        text = " ".join(["Obrigada"] * 30)
        score, issues = validator._check_repetition(text)

        assert score < 0.2  # Should score very poorly
        assert any("high_word_repetition:obrigada" in issue.lower() for issue in issues)

    def test_phrase_repetition(self, validator: TranscriptionValidator) -> None:
        """Test text with repeated multi-word phrases."""
        phrase = "Eu não falei muito no chão mas"
        text = " ".join([phrase] * 5)
        score, issues = validator._check_repetition(text)

        assert score < 0.5
        assert any("repeated_phrase" in issue for issue in issues)

    def test_short_transcription(self, validator: TranscriptionValidator) -> None:
        """Test very short transcription (< 5 words)."""
        text = "Hi there"
        score, issues = validator._check_repetition(text)

        assert score == 0.7
        assert "very_short_transcription" in issues

    def test_mild_repetition_acceptable(self, validator: TranscriptionValidator) -> None:
        """Test that mild repetition is acceptable."""
        # Create text where common words repeat but not excessively
        text = "I went to the store and I bought some milk. Then I went home and I made coffee."
        score, _issues = validator._check_repetition(text)

        # Should be acceptable (common words like "I" repeat naturally)
        assert score >= 0.7


class TestSegmentPatterns:
    """Tests for segment pattern analysis."""

    @pytest.fixture
    def validator(self) -> TranscriptionValidator:
        """Create a validator with default config."""
        return TranscriptionValidator()

    def test_no_segments(self, validator: TranscriptionValidator) -> None:
        """Test with no segments provided."""
        score, issues = validator._check_segment_patterns([])

        assert score == 1.0
        assert len(issues) == 0

    def test_normal_segments(self, validator: TranscriptionValidator) -> None:
        """Test with normal, varied segment durations."""
        segments = [
            TranscriptionSegment(text="Hello", start=0.0, end=1.5),
            TranscriptionSegment(text="how are you", start=1.5, end=3.2),
            TranscriptionSegment(text="I'm fine", start=3.2, end=5.1),
            TranscriptionSegment(text="thanks", start=5.1, end=6.0),
        ]
        score, issues = validator._check_segment_patterns(segments)

        assert score == 1.0
        assert len(issues) == 0

    def test_suspicious_uniform_intervals(self, validator: TranscriptionValidator) -> None:
        """Test detection of suspicious uniform 1-second intervals."""
        # Create 7 consecutive segments with 1-second intervals
        segments = [
            TranscriptionSegment(text=f"Segment {i}", start=float(i), end=float(i) + 0.5)
            for i in range(7)
        ]
        score, issues = validator._check_segment_patterns(segments)

        assert score < 1.0
        assert any("suspicious_uniform_intervals" in issue for issue in issues)

    def test_empty_segments(self, validator: TranscriptionValidator) -> None:
        """Test detection of empty segments."""
        segments = [
            TranscriptionSegment(text="", start=0.0, end=1.0),
            TranscriptionSegment(text="", start=1.0, end=2.0),
            TranscriptionSegment(text="Some text", start=2.0, end=3.0),
            TranscriptionSegment(text="", start=3.0, end=4.0),
        ]
        score, issues = validator._check_segment_patterns(segments)

        assert score < 1.0
        assert any("high_empty_segments" in issue for issue in issues)


class TestContentDensity:
    """Tests for content density (words per minute) checks."""

    @pytest.fixture
    def validator(self) -> TranscriptionValidator:
        """Create a validator with default config."""
        return TranscriptionValidator()

    def test_normal_word_density(self, validator: TranscriptionValidator) -> None:
        """Test normal speaking rate (around 150 wpm)."""
        text = " ".join(["word"] * 150)  # 150 words
        duration_ms = 60_000  # 1 minute
        score, issues = validator._check_content_density(text, duration_ms)

        assert score == 1.0
        assert len(issues) == 0

    def test_too_sparse(self, validator: TranscriptionValidator) -> None:
        """Test transcription with too few words per minute."""
        text = "just a few words"  # 4 words
        duration_ms = 60_000  # 1 minute
        score, issues = validator._check_content_density(text, duration_ms)

        assert score < 1.0
        assert any("low_content_density" in issue for issue in issues)

    def test_too_dense(self, validator: TranscriptionValidator) -> None:
        """Test transcription with too many words per minute."""
        text = " ".join(["word"] * 400)  # 400 words
        duration_ms = 60_000  # 1 minute
        score, issues = validator._check_content_density(text, duration_ms)

        assert score < 1.0
        assert any("high_content_density" in issue for issue in issues)

    def test_duration_none_guard(self, validator: TranscriptionValidator) -> None:
        """Test that duration_ms=None is handled gracefully."""
        text = "Some transcription text"
        score, issues = validator._check_content_density(text, None)

        assert score == 0.5  # Neutral score
        assert "duration_unknown:neutral_score" in issues

    def test_invalid_duration(self, validator: TranscriptionValidator) -> None:
        """Test that invalid duration (0 or negative) is penalized."""
        text = "Some text"
        score, issues = validator._check_content_density(text, 0)

        assert score == 0.3
        assert "invalid_duration" in issues


class TestValidateEnrichedRecord:
    """Tests for the validate_enriched_record convenience function."""

    @pytest.fixture
    def sample_record(self, sample_enriched_record_data: dict) -> EnrichedRecord:
        """Create a sample EnrichedRecord for testing."""
        return EnrichedRecord(**sample_enriched_record_data)

    def test_validate_sets_quality_field(self, sample_record: EnrichedRecord) -> None:
        """Test that validation sets the transcription_quality field."""
        assert sample_record.transcription_quality is None

        validate_enriched_record(sample_record)

        assert sample_record.transcription_quality is not None
        assert sample_record.transcription_quality.overall_score >= 0.0
        assert sample_record.transcription_quality.overall_score <= 1.0

    def test_validate_sets_is_valid_field(self, sample_record: EnrichedRecord) -> None:
        """Test that validation sets the is_valid field."""
        assert sample_record.is_valid is None

        validate_enriched_record(sample_record)

        assert sample_record.is_valid is not None
        assert isinstance(sample_record.is_valid, bool)

    def test_validation_disabled_marks_as_valid(self, sample_record: EnrichedRecord) -> None:
        """Test that disabled validation marks records as valid."""
        config = TranscriptionQualityConfig(enabled=False)
        validate_enriched_record(sample_record, config)

        assert sample_record.is_valid is True
        assert sample_record.transcription_quality is None

    def test_good_transcription_marked_valid(self) -> None:
        """Test that a good transcription is marked as valid."""
        record = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["parent_id"],
            webContentLink="https://example.com",
            duration_milliseconds=60000,
            transcription_text=(
                "This is a normal transcription with good quality and varied vocabulary."
            ),
            detected_language="en",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.0,
            transcription_status="completed",
        )

        config = TranscriptionQualityConfig(expected_language="en")
        validate_enriched_record(record, config)

        assert record.is_valid is True
        assert record.transcription_quality.overall_score >= 0.5

    def test_bad_transcription_marked_invalid(self) -> None:
        """Test that a bad transcription (Japanese when expecting Portuguese) is invalid."""
        record = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["parent_id"],
            webContentLink="https://example.com",
            duration_milliseconds=60000,
            transcription_text="これは日本語のテキストです" * 10,
            detected_language="ja",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.0,
            transcription_status="completed",
        )

        config = TranscriptionQualityConfig(expected_language="pt")
        validate_enriched_record(record, config)

        assert record.is_valid is False
        assert record.transcription_quality.overall_score < 0.5

    def test_repeated_word_transcription_marked_invalid(self) -> None:
        """Test that repetitive transcription is marked invalid."""
        record = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["parent_id"],
            webContentLink="https://example.com",
            duration_milliseconds=60000,
            transcription_text=" ".join(["Obrigada"] * 30),
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.0,
            transcription_status="completed",
        )

        # Repetition weight (0.30) alone can't pull the overall score below
        # the default 0.5 threshold when other dimensions score well, so use
        # a higher threshold to catch this failure mode.
        config = TranscriptionQualityConfig(quality_threshold=0.75)
        validate_enriched_record(record, config)

        assert record.is_valid is False
        assert record.transcription_quality.repetition_score == 0.0

    def test_function_returns_record_for_chaining(self, sample_record: EnrichedRecord) -> None:
        """Test that the function returns the record for chaining."""
        result = validate_enriched_record(sample_record)

        assert result is sample_record
