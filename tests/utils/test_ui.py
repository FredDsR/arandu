"""Tests for UI utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from arandu.schemas import EnrichedRecord
from arandu.utils.ui import (
    MAX_DISPLAY_FILES,
    _truncate_text,
    create_download_progress,
    create_progress,
    create_transcription_progress,
    display_config_table,
    display_file_list,
    display_result_panel,
)


class TestTruncateText:
    """Tests for _truncate_text function."""

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        text = "This is a short text"
        result = _truncate_text(text, max_length=100)

        assert result == text
        assert "..." not in result

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated."""
        text = "This is a very long text " * 50  # Much longer than 500 chars
        result = _truncate_text(text, max_length=500)

        assert len(result) <= 503  # 500 + "..."
        assert result.endswith("...")

    def test_truncate_at_word_boundary(self) -> None:
        """Test that truncation happens at word boundary."""
        text = "word1 word2 word3 word4 word5 " * 20
        result = _truncate_text(text, max_length=50)

        assert result.endswith("...")
        # Should not end with partial word before ...
        text_before_ellipsis = result[:-3]
        assert not text_before_ellipsis.endswith(" w")

    def test_truncate_exact_length(self) -> None:
        """Test text at exactly max_length."""
        text = "a" * 500
        result = _truncate_text(text, max_length=500)

        assert result == text
        assert "..." not in result

    def test_truncate_custom_max_length(self) -> None:
        """Test with custom max_length parameter."""
        text = "This is a test " * 10
        result = _truncate_text(text, max_length=30)

        assert len(result) <= 33  # 30 + "..."


class TestCreateProgress:
    """Tests for progress bar creation functions."""

    def test_create_progress_context(self) -> None:
        """Test create_progress context manager."""
        with create_progress("Test") as progress:
            assert progress is not None
            assert hasattr(progress, "add_task")

    def test_create_download_progress(self) -> None:
        """Test create_download_progress function."""
        progress = create_download_progress()

        assert progress is not None
        assert hasattr(progress, "add_task")
        assert hasattr(progress, "update")

    def test_create_transcription_progress(self) -> None:
        """Test create_transcription_progress function."""
        progress = create_transcription_progress()

        assert progress is not None
        assert hasattr(progress, "add_task")
        assert hasattr(progress, "update")


class TestDisplayResultPanel:
    """Tests for display_result_panel function."""

    @patch("arandu.utils.ui.console")
    def test_display_result_panel(self, mock_console: MagicMock) -> None:
        """Test displaying result panel."""
        record = EnrichedRecord(
            file_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder1"],
            webContentLink="http://example.com",
            transcription_text="This is a test transcription.",
            detected_language="en",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.5,
            transcription_status="completed",
        )

        display_result_panel(record)

        mock_console.print.assert_called_once()
        # Check that Panel was passed to print
        panel_arg = mock_console.print.call_args[0][0]
        assert hasattr(panel_arg, "title")

    @patch("arandu.utils.ui.console")
    def test_display_result_panel_long_text(self, mock_console: MagicMock) -> None:
        """Test displaying result panel with long transcription text."""
        long_text = "This is a very long transcription. " * 50
        record = EnrichedRecord(
            file_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder1"],
            webContentLink="http://example.com",
            transcription_text=long_text,
            detected_language="en",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.5,
            transcription_status="completed",
        )

        display_result_panel(record)

        mock_console.print.assert_called_once()


class TestDisplayConfigTable:
    """Tests for display_config_table function."""

    @patch("arandu.utils.ui.console")
    def test_display_config_table(self, mock_console: MagicMock) -> None:
        """Test displaying configuration table."""
        display_config_table(
            model_id="openai/whisper-large-v3",
            device="cuda:0",
            quantize=True,
            source="/path/to/file.mp3",
        )

        mock_console.print.assert_called_once()
        # Check that Table was passed to print
        table_arg = mock_console.print.call_args[0][0]
        assert hasattr(table_arg, "title")

    @patch("arandu.utils.ui.console")
    def test_display_config_table_no_quantization(self, mock_console: MagicMock) -> None:
        """Test displaying config table with quantization disabled."""
        display_config_table(
            model_id="openai/whisper-tiny",
            device="cpu",
            quantize=False,
            source="/path/to/folder",
        )

        mock_console.print.assert_called_once()


class TestDisplayFileList:
    """Tests for display_file_list function."""

    @patch("arandu.utils.ui.console")
    def test_display_file_list_few_files(self, mock_console: MagicMock) -> None:
        """Test displaying file list with few files."""
        files = [
            {
                "name": "file1.mp3",
                "mimeType": "audio/mpeg",
                "size": 1024 * 1024,  # 1 MB
            },
            {
                "name": "file2.mp4",
                "mimeType": "video/mp4",
                "size": 2048 * 1024,  # 2 MB
            },
        ]

        display_file_list(files)

        mock_console.print.assert_called_once()
        table_arg = mock_console.print.call_args[0][0]
        assert hasattr(table_arg, "title")

    @patch("arandu.utils.ui.console")
    def test_display_file_list_many_files(self, mock_console: MagicMock) -> None:
        """Test displaying file list with many files (more than MAX_DISPLAY_FILES)."""
        files = [
            {
                "name": f"file{i}.mp3",
                "mimeType": "audio/mpeg",
                "size": 1024 * 1024 * i,
            }
            for i in range(30)
        ]

        display_file_list(files)

        mock_console.print.assert_called_once()

    @patch("arandu.utils.ui.console")
    def test_display_file_list_with_string_size(self, mock_console: MagicMock) -> None:
        """Test displaying file list with size as string."""
        files = [
            {
                "name": "file1.mp3",
                "mimeType": "audio/mpeg",
                "size": "Unknown",
            },
        ]

        display_file_list(files)

        mock_console.print.assert_called_once()

    @patch("arandu.utils.ui.console")
    def test_display_file_list_missing_fields(self, mock_console: MagicMock) -> None:
        """Test displaying file list with missing fields."""
        files = [
            {"name": "file1.mp3"},  # Missing mimeType and size
        ]

        display_file_list(files)

        mock_console.print.assert_called_once()

    @patch("arandu.utils.ui.console")
    def test_display_file_list_empty(self, mock_console: MagicMock) -> None:
        """Test displaying empty file list."""
        files = []

        display_file_list(files)

        mock_console.print.assert_called_once()


class TestConstants:
    """Tests for module constants."""

    def test_max_display_files_constant(self) -> None:
        """Test MAX_DISPLAY_FILES constant value."""
        assert MAX_DISPLAY_FILES == 20
        assert isinstance(MAX_DISPLAY_FILES, int)
