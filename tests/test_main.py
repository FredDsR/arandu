"""Tests for main CLI module helper functions."""

from __future__ import annotations

import pytest

from gtranscriber.main import (
    _create_segments_from_result,
    _ensure_float,
    _safe_int_conversion,
)


class TestEnsureFloat:
    """Tests for _ensure_float helper function."""

    def test_ensure_float_valid_float(self) -> None:
        """Test converting valid float."""
        result = _ensure_float(3.14, 0.0)
        assert result == 3.14

    def test_ensure_float_valid_int(self) -> None:
        """Test converting valid int."""
        result = _ensure_float(42, 0.0)
        assert result == 42.0

    def test_ensure_float_valid_string(self) -> None:
        """Test converting valid string."""
        result = _ensure_float("123.45", 0.0)
        assert result == 123.45

    def test_ensure_float_none_uses_default(self) -> None:
        """Test that None returns default."""
        result = _ensure_float(None, 99.0)
        assert result == 99.0

    def test_ensure_float_invalid_string_uses_default(self) -> None:
        """Test that invalid string returns default."""
        result = _ensure_float("not a number", 50.0)
        assert result == 50.0

    def test_ensure_float_object_uses_default(self) -> None:
        """Test that arbitrary object returns default."""
        result = _ensure_float(object(), 10.0)
        assert result == 10.0


class TestSafeIntConversion:
    """Tests for _safe_int_conversion helper function."""

    def test_safe_int_conversion_valid_string(self) -> None:
        """Test converting valid integer string."""
        result = _safe_int_conversion("42")
        assert result == 42

    def test_safe_int_conversion_valid_string_with_default(self) -> None:
        """Test converting valid string with default provided."""
        result = _safe_int_conversion("100", default=50)
        assert result == 100

    def test_safe_int_conversion_none_with_default(self) -> None:
        """Test that None returns default."""
        result = _safe_int_conversion(None, default=99)
        assert result == 99

    def test_safe_int_conversion_none_without_default(self) -> None:
        """Test that None without default returns None."""
        result = _safe_int_conversion(None)
        assert result is None

    def test_safe_int_conversion_invalid_string_with_default(self) -> None:
        """Test that invalid string returns default."""
        result = _safe_int_conversion("not a number", default=50)
        assert result == 50

    def test_safe_int_conversion_invalid_string_without_default(self) -> None:
        """Test that invalid string without default returns None."""
        result = _safe_int_conversion("not a number")
        assert result is None

    def test_safe_int_conversion_float_string(self) -> None:
        """Test converting float string truncates to int."""
        result = _safe_int_conversion("42.7")
        assert result == 42

    def test_safe_int_conversion_negative_number(self) -> None:
        """Test converting negative number."""
        result = _safe_int_conversion("-42")
        assert result == -42


class TestCreateSegmentsFromResult:
    """Tests for _create_segments_from_result function."""

    def test_create_segments_with_chunks(self) -> None:
        """Test creating segments from result with chunks."""
        from gtranscriber.core.engine import TranscriptionResult

        result = TranscriptionResult(
            text="Test transcription",
            segments=[
                {"text": "Test", "start": 0.0, "end": 1.0},
                {"text": "transcription", "start": 1.0, "end": 2.5},
            ],
            detected_language="en",
            language_probability=0.95,
            processing_duration_sec=5.0,
            model_id="test-model",
            device="cpu",
        )

        segments = _create_segments_from_result(result)

        assert segments is not None
        assert len(segments) == 2
        assert segments[0].text == "Test"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0
        assert segments[1].text == "transcription"

    def test_create_segments_without_chunks(self) -> None:
        """Test creating segments from result without chunks."""
        from gtranscriber.core.engine import TranscriptionResult

        result = TranscriptionResult(
            text="Test transcription",
            segments=None,
            detected_language="en",
            language_probability=0.95,
            processing_duration_sec=5.0,
            model_id="test-model",
            device="cpu",
        )

        segments = _create_segments_from_result(result)

        assert segments is None

    def test_create_segments_empty_list(self) -> None:
        """Test creating segments from empty segments list."""
        from gtranscriber.core.engine import TranscriptionResult

        result = TranscriptionResult(
            text="Test",
            segments=[],
            detected_language="en",
            language_probability=0.95,
            processing_duration_sec=1.0,
            model_id="test-model",
            device="cpu",
        )

        segments = _create_segments_from_result(result)

        assert segments is not None
        assert len(segments) == 0


class TestMainModuleImports:
    """Tests for main module imports and structure."""

    def test_import_main_functions(self) -> None:
        """Test that main functions can be imported."""
        from gtranscriber.main import info, main, transcribe

        assert callable(main)
        assert callable(transcribe)
        assert callable(info)

    def test_import_version_callback(self) -> None:
        """Test importing version callback."""
        from gtranscriber.main import version_callback

        assert callable(version_callback)

    def test_version_callback_with_true(self) -> None:
        """Test version callback with True value."""
        import typer

        from gtranscriber.main import version_callback

        with pytest.raises(typer.Exit) as exc_info:
            version_callback(True)
        assert exc_info.value.exit_code == 0

    def test_version_callback_with_false(self) -> None:
        """Test version callback with False value does nothing."""
        from gtranscriber.main import version_callback

        # Should return None and not raise any exception
        result = version_callback(False)
        assert result is None

    def test_setup_logging(self) -> None:
        """Test setup_logging function."""
        from gtranscriber.main import setup_logging

        # Should execute without errors
        setup_logging()

    def test_main_function(self) -> None:
        """Test main callback function."""
        from gtranscriber.main import main

        # Should execute without errors
        main()


# ---------------------------------------------------------------------------
# Deprecation warning for report command
# ---------------------------------------------------------------------------


class TestReportCommandDeprecation:
    """Tests for the deprecation warning on the report command."""

    def test_report_command_deprecation_warning(self) -> None:
        """report command prints deprecation warning via print_warning."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from gtranscriber.main import app

        runner = CliRunner()

        # Patch print_warning and the heavy imports to avoid filesystem access
        with (
            patch("gtranscriber.main.print_warning") as mock_warn,
            patch("gtranscriber.core.report.ResultsCollector") as mock_collector_cls,
        ):
            mock_collector = MagicMock()
            mock_collector.load_all_runs.return_value = []
            mock_collector_cls.return_value = mock_collector

            runner.invoke(app, ["report", "--no-png"])

        # Verify the deprecation warning was printed
        assert mock_warn.called
        call_args = mock_warn.call_args[0][0]
        assert "deprecated" in call_args.lower()
        assert "serve-report" in call_args
