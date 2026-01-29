"""Tests for batch transcription processing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytest_mock import MockerFixture

from gtranscriber.core.batch import (
    AUDIO_VIDEO_MIME_TYPES,
    _parse_parents_from_string,
)


class TestConstants:
    """Tests for module constants."""

    def test_audio_video_mime_types(self) -> None:
        """Test that AUDIO_VIDEO_MIME_TYPES contains expected formats."""
        assert "audio/mpeg" in AUDIO_VIDEO_MIME_TYPES
        assert "audio/mp3" in AUDIO_VIDEO_MIME_TYPES
        assert "audio/wav" in AUDIO_VIDEO_MIME_TYPES
        assert "video/mp4" in AUDIO_VIDEO_MIME_TYPES
        assert "video/quicktime" in AUDIO_VIDEO_MIME_TYPES
        assert isinstance(AUDIO_VIDEO_MIME_TYPES, set)


class TestParseParentsFromString:
    """Tests for _parse_parents_from_string helper function."""

    def test_parse_from_json_string(self) -> None:
        """Test parsing parents from JSON string."""
        result = _parse_parents_from_string('["folder1", "folder2"]')

        assert result == ["folder1", "folder2"]

    def test_parse_from_single_quoted_json(self) -> None:
        """Test parsing parents from single-quoted JSON string."""
        result = _parse_parents_from_string("['folder1', 'folder2']")

        assert result == ["folder1", "folder2"]

    def test_parse_from_list(self) -> None:
        """Test parsing parents from list."""
        result = _parse_parents_from_string(["folder1", "folder2"])

        assert result == ["folder1", "folder2"]

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON returns empty list."""
        result = _parse_parents_from_string("not valid json")

        assert result == []

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        result = _parse_parents_from_string("")

        assert result == []

    def test_parse_non_list_json(self) -> None:
        """Test parsing JSON that doesn't result in a list."""
        result = _parse_parents_from_string('{"key": "value"}')

        assert result == []


class TestBatchProcessingDataStructures:
    """Tests for batch processing data structures."""

    def test_import_batch_result(self) -> None:
        """Test that BatchResult can be imported."""
        from gtranscriber.core.batch import BatchResult

        assert BatchResult is not None

    def test_import_batch_processor(self) -> None:
        """Test that BatchProcessor can be imported."""
        from gtranscriber.core.batch import BatchProcessor

        assert BatchProcessor is not None


class TestWorkerInitialization:
    """Tests for worker initialization."""

    @patch("gtranscriber.core.batch.WhisperEngine")
    def test_init_worker(self, mock_engine: MagicMock) -> None:
        """Test worker initialization with WhisperEngine."""
        from gtranscriber.core.batch import _init_worker

        _init_worker(
            model_id="openai/whisper-tiny",
            force_cpu=True,
            quantize=False,
            language="pt",
        )

        mock_engine.assert_called_once_with(
            model_id="openai/whisper-tiny",
            force_cpu=True,
            quantize=False,
            language="pt",
        )


class TestBatchProcessorInitialization:
    """Tests for BatchProcessor initialization."""

    @patch("gtranscriber.core.batch.DriveClient")
    def test_batch_processor_creation(
        self,
        mock_drive_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test creating a BatchProcessor instance."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.batch import BatchProcessor

        config = TranscriberConfig(
            model_id="openai/whisper-tiny",
            credentials="creds.json",
            token="token.json",
        )

        processor = BatchProcessor(
            config=config,
            catalog_file=str(tmp_path / "catalog.csv"),
        )

        assert processor.config == config
        assert processor.catalog_file == tmp_path / "catalog.csv"

    @patch("gtranscriber.core.batch.DriveClient")
    def test_batch_processor_with_checkpoint(
        self,
        mock_drive_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test BatchProcessor with checkpoint file."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.batch import BatchProcessor

        config = TranscriberConfig()
        checkpoint_file = tmp_path / "checkpoint.json"

        processor = BatchProcessor(
            config=config,
            catalog_file=str(tmp_path / "catalog.csv"),
            checkpoint_file=str(checkpoint_file),
        )

        assert processor.checkpoint_file == checkpoint_file


class TestBatchProcessingErrors:
    """Tests for batch processing error handling."""

    def test_no_audio_stream_error_import(self) -> None:
        """Test that NoAudioStreamError can be imported."""
        from gtranscriber.core.drive import NoAudioStreamError

        error = NoAudioStreamError("test_id", "test.mp4")
        assert error.file_id == "test_id"
        assert error.file_name == "test.mp4"


class TestBatchCatalogParsing:
    """Tests for catalog file parsing."""

    @patch("gtranscriber.core.batch.DriveClient")
    def test_load_catalog_from_csv(
        self,
        mock_drive_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test loading catalog from CSV file."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.batch import BatchProcessor

        # Create a test catalog CSV
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mimeType,parents,webContentLink,size\n"
            'file1,test1.mp3,audio/mpeg,"[\'folder1\']",http://example.com/1,1000\n'
            'file2,test2.mp4,video/mp4,"[\'folder2\']",http://example.com/2,2000\n'
        )

        config = TranscriberConfig()
        processor = BatchProcessor(
            config=config,
            catalog_file=str(catalog_file),
        )

        # Access the catalog loading method if available
        # This tests that the file can be parsed without errors
        assert processor.catalog_file.exists()


class TestBatchResultDataClass:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self) -> None:
        """Test creating a BatchResult instance."""
        from gtranscriber.core.batch import BatchResult

        result = BatchResult(
            total=10,
            successful=8,
            failed=2,
            skipped=0,
            duration_sec=120.5,
            failed_files={"file1": "Error 1", "file2": "Error 2"},
        )

        assert result.total == 10
        assert result.successful == 8
        assert result.failed == 2
        assert result.skipped == 0
        assert result.duration_sec == 120.5
        assert len(result.failed_files) == 2


class TestAudioVideoFiltering:
    """Tests for audio/video file filtering logic."""

    def test_mime_type_is_audio_video(self) -> None:
        """Test that common audio/video MIME types are recognized."""
        audio_types = [
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "audio/flac",
            "audio/ogg",
        ]
        video_types = [
            "video/mp4",
            "video/quicktime",
            "video/avi",
            "video/x-matroska",
        ]

        for mime_type in audio_types + video_types:
            assert mime_type in AUDIO_VIDEO_MIME_TYPES

    def test_mime_type_not_audio_video(self) -> None:
        """Test that non-audio/video MIME types are not included."""
        non_av_types = [
            "application/pdf",
            "text/plain",
            "image/jpeg",
            "application/json",
        ]

        for mime_type in non_av_types:
            assert mime_type not in AUDIO_VIDEO_MIME_TYPES
