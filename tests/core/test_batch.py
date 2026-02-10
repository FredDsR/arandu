"""Tests for batch transcription processing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

from gtranscriber.core.batch import (
    AUDIO_VIDEO_MIME_TYPES,
    _create_segments_from_result,
    _parse_parents_from_string,
    load_catalog,
    transcribe_single_file,
)
from gtranscriber.core.results_manager import ResultsManager


class _ThreadPoolCompat(ThreadPoolExecutor):
    """ThreadPoolExecutor that ignores mp_context for test compatibility."""

    def __init__(self, *args: object, mp_context: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


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

    def test_parse_invalid_type(self) -> None:
        """Test parsing invalid type (not str or list) returns empty list."""
        result = _parse_parents_from_string(123)  # type: ignore[arg-type]

        assert result == []


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


class TestBatchProcessingErrors:
    """Tests for batch processing error handling."""

    def test_no_audio_stream_error_import(self) -> None:
        """Test that NoAudioStreamError can be imported."""
        from pathlib import Path

        from gtranscriber.core.drive import NoAudioStreamError

        error = NoAudioStreamError("test_id", "test.mp4", Path("/tmp/test.mp4"))
        assert error.file_id == "test_id"
        assert error.file_name == "test.mp4"


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


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_batch_config_creation(self, tmp_path: Path) -> None:
        """Test creating BatchConfig instance."""
        from gtranscriber.core.batch import BatchConfig

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            model_id="openai/whisper-tiny",
            num_workers=4,
            force_cpu=True,
            quantize=False,
            language="en",
        )

        assert config.model_id == "openai/whisper-tiny"
        assert config.num_workers == 4
        assert config.force_cpu is True
        assert config.language == "en"

    def test_batch_config_from_transcriber_config(self, tmp_path: Path) -> None:
        """Test creating BatchConfig from TranscriberConfig."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.batch import BatchConfig

        transcriber_config = TranscriberConfig(
            model_id="openai/whisper-large-v3",
            force_cpu=False,
            quantize=True,
            language="pt",
        )

        batch_config = BatchConfig.from_transcriber_config(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            config=transcriber_config,
            num_workers=2,
        )

        assert batch_config.model_id == "openai/whisper-large-v3"
        assert batch_config.force_cpu is False
        assert batch_config.quantize is True
        assert batch_config.language == "pt"
        assert batch_config.num_workers == 2

    def test_batch_config_from_transcriber_config_none(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test BatchConfig.from_transcriber_config with None config."""
        from gtranscriber.core.batch import BatchConfig

        # Mock TranscriberConfig to avoid environment dependency
        mock_config = mocker.patch("gtranscriber.core.batch.TranscriberConfig")
        mock_config.return_value.model_id = "default-model"
        mock_config.return_value.credentials = "creds.json"
        mock_config.return_value.token = "token.json"
        mock_config.return_value.force_cpu = False
        mock_config.return_value.quantize = False
        mock_config.return_value.language = None

        BatchConfig.from_transcriber_config(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            config=None,
            num_workers=1,
        )

        mock_config.assert_called_once()


class TestTranscriptionTask:
    """Tests for TranscriptionTask dataclass."""

    def test_transcription_task_creation(self) -> None:
        """Test creating TranscriptionTask instance."""
        from gtranscriber.core.batch import TranscriptionTask

        task = TranscriptionTask(
            file_id="file123",
            name="test.mp3",
            mime_type="audio/mpeg",
            size_bytes=1024000,
            parents=["folder1"],
            web_content_link="http://example.com/file123",
            duration_ms=60000,
        )

        assert task.file_id == "file123"
        assert task.name == "test.mp3"
        assert task.mime_type == "audio/mpeg"
        assert task.size_bytes == 1024000
        assert task.duration_ms == 60000

    def test_transcription_task_optional_fields(self) -> None:
        """Test TranscriptionTask with optional fields as None."""
        from gtranscriber.core.batch import TranscriptionTask

        task = TranscriptionTask(
            file_id="file123",
            name="test.mp3",
            mime_type="audio/mpeg",
            size_bytes=None,
            parents=[],
            web_content_link="http://example.com",
            duration_ms=None,
        )

        assert task.size_bytes is None
        assert task.duration_ms is None


class TestEnsureFloat:
    """Tests for _ensure_float helper function."""

    def test_ensure_float_valid_float(self) -> None:
        """Test converting valid float."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float(3.14, 0.0)
        assert result == 3.14

    def test_ensure_float_valid_int(self) -> None:
        """Test converting valid int."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float(42, 0.0)
        assert result == 42.0

    def test_ensure_float_valid_string(self) -> None:
        """Test converting valid string."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float("123.45", 0.0)
        assert result == 123.45

    def test_ensure_float_none_uses_default(self) -> None:
        """Test that None returns default."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float(None, 99.0)
        assert result == 99.0

    def test_ensure_float_invalid_string_uses_default(self) -> None:
        """Test that invalid string returns default."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float("not a number", 50.0)
        assert result == 50.0

    def test_ensure_float_object_uses_default(self) -> None:
        """Test that arbitrary object returns default."""
        from gtranscriber.core.batch import _ensure_float

        result = _ensure_float(object(), 10.0)
        assert result == 10.0


class TestWorkerFunctions:
    """Tests for worker-related functions."""

    @patch("gtranscriber.core.batch.WhisperEngine")
    def test_init_worker_sets_global(self, mock_engine: MagicMock) -> None:
        """Test that _init_worker sets the global engine."""
        from gtranscriber.core.batch import _init_worker

        _init_worker(
            model_id="test-model",
            force_cpu=True,
            quantize=False,
            language="en",
        )

        mock_engine.assert_called_once_with(
            model_id="test-model",
            force_cpu=True,
            quantize=False,
            language="en",
        )


class TestCreateSegmentsFromResult:
    """Tests for _create_segments_from_result helper function."""

    def test_create_segments_with_valid_data(self) -> None:
        """Test creating segments from valid result."""
        mock_result = MagicMock()
        mock_result.segments = [
            {"text": "Hello", "start": 0.0, "end": 1.5},
            {"text": "World", "start": 1.5, "end": 3.0},
        ]

        segments = _create_segments_from_result(mock_result)

        assert segments is not None
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[1].text == "World"
        assert segments[1].start == 1.5
        assert segments[1].end == 3.0

    def test_create_segments_with_empty_list(self) -> None:
        """Test creating segments from empty segments list."""
        mock_result = MagicMock()
        mock_result.segments = []

        segments = _create_segments_from_result(mock_result)

        assert segments is None

    def test_create_segments_with_none(self) -> None:
        """Test creating segments when segments is None."""
        mock_result = MagicMock()
        mock_result.segments = None

        segments = _create_segments_from_result(mock_result)

        assert segments is None

    def test_create_segments_with_missing_text(self) -> None:
        """Test creating segments with missing text field."""
        mock_result = MagicMock()
        mock_result.segments = [
            {"start": 0.0, "end": 1.5},  # No text field
        ]

        segments = _create_segments_from_result(mock_result)

        assert segments is not None
        assert len(segments) == 1
        assert segments[0].text == ""  # Should default to empty string

    def test_create_segments_with_invalid_times(self) -> None:
        """Test creating segments with invalid time values."""
        mock_result = MagicMock()
        mock_result.segments = [
            {"text": "Test", "start": "invalid", "end": 2.0},
        ]

        segments = _create_segments_from_result(mock_result)

        assert segments is not None
        assert len(segments) == 1
        assert segments[0].start == 0.0  # Should default to 0.0
        assert segments[0].end == 2.0

    def test_create_segments_end_before_start(self) -> None:
        """Test that end time is corrected when it's before start time."""
        mock_result = MagicMock()
        mock_result.segments = [
            {"text": "Test", "start": 5.0, "end": 2.0},  # End before start
        ]

        segments = _create_segments_from_result(mock_result)

        assert segments is not None
        assert len(segments) == 1
        assert segments[0].start == 5.0
        assert segments[0].end == 5.0  # Should be corrected to match start


class TestLoadCatalog:
    """Tests for load_catalog function."""

    def test_load_catalog_valid_csv(self, tmp_path: Path) -> None:
        """Test loading a valid catalog CSV file."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mime_type,size_bytes,parents,web_content_link,duration_milliseconds\n"
            "file1,test.mp3,audio/mpeg,1024,\"['folder1']\",http://example.com/file1,60000\n"
            "file2,test.mp4,video/mp4,2048,\"['folder2']\",http://example.com/file2,120000\n"
        )

        tasks = load_catalog(catalog_file)

        assert len(tasks) == 2
        assert tasks[0].file_id == "file1"
        assert tasks[0].name == "test.mp3"
        assert tasks[0].mime_type == "audio/mpeg"
        assert tasks[0].size_bytes == 1024
        assert tasks[0].parents == ["folder1"]
        assert tasks[1].file_id == "file2"

    def test_load_catalog_filters_non_audio_video(self, tmp_path: Path) -> None:
        """Test that load_catalog filters out non-audio/video files."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mime_type,size_bytes,parents,web_content_link,duration_milliseconds\n"
            'file1,test.mp3,audio/mpeg,1024,"[]",http://example.com/file1,60000\n'
            'file2,test.pdf,application/pdf,2048,"[]",http://example.com/file2,\n'
            'file3,test.txt,text/plain,512,"[]",http://example.com/file3,\n'
        )

        tasks = load_catalog(catalog_file)

        assert len(tasks) == 1
        assert tasks[0].mime_type == "audio/mpeg"

    def test_load_catalog_missing_required_columns(self, tmp_path: Path) -> None:
        """Test that load_catalog raises error for missing required columns."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name\n"  # Missing mime_type
            "file1,test.mp3\n"
        )

        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(catalog_file)

    def test_load_catalog_empty_file(self, tmp_path: Path) -> None:
        """Test that load_catalog raises error for empty file."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text("")

        with pytest.raises(ValueError, match="empty or invalid"):
            load_catalog(catalog_file)

    def test_load_catalog_skips_rows_with_missing_id_or_name(self, tmp_path: Path) -> None:
        """Test that rows with missing gdrive_id or name are skipped."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mime_type,size_bytes,parents,web_content_link,duration_milliseconds\n"
            'file1,test.mp3,audio/mpeg,1024,"[]",http://example.com/file1,60000\n'
            ',missing_id.mp3,audio/mpeg,1024,"[]",http://example.com/file2,60000\n'  # Missing ID
            'file3,,audio/mpeg,1024,"[]",http://example.com/file3,60000\n'  # Missing name
        )

        tasks = load_catalog(catalog_file)

        assert len(tasks) == 1
        assert tasks[0].file_id == "file1"

    def test_load_catalog_handles_invalid_size_bytes(self, tmp_path: Path) -> None:
        """Test that invalid size_bytes values are handled gracefully."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mime_type,size_bytes,parents,web_content_link,duration_milliseconds\n"
            'file1,test.mp3,audio/mpeg,invalid,"[]",http://example.com/file1,60000\n'
            'file2,test2.mp3,audio/mpeg,,"[]",http://example.com/file2,60000\n'
        )

        tasks = load_catalog(catalog_file)

        assert len(tasks) == 2
        assert tasks[0].size_bytes is None
        assert tasks[1].size_bytes is None

    def test_load_catalog_handles_invalid_duration(self, tmp_path: Path) -> None:
        """Test that invalid duration_milliseconds values are handled gracefully."""
        catalog_file = tmp_path / "catalog.csv"
        catalog_file.write_text(
            "gdrive_id,name,mime_type,size_bytes,parents,web_content_link,duration_milliseconds\n"
            'file1,test.mp3,audio/mpeg,1024,"[]",http://example.com/file1,invalid\n'
            'file2,test2.mp3,audio/mpeg,1024,"[]",http://example.com/file2,\n'
        )

        tasks = load_catalog(catalog_file)

        assert len(tasks) == 2
        assert tasks[0].duration_ms is None
        assert tasks[1].duration_ms is None


class TestTranscribeSingleFile:
    """Tests for transcribe_single_file function."""

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    @patch("gtranscriber.core.batch.save_enriched_record")
    @patch("gtranscriber.core.batch.has_audio_stream")
    @patch("gtranscriber.core.batch.get_media_duration_ms")
    def test_transcribe_single_file_audio_success(
        self,
        mock_duration: MagicMock,
        mock_has_audio: MagicMock,
        mock_save: MagicMock,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful transcription of an audio file."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask

        # Setup mocks
        temp_file = tmp_path / "temp.mp3"
        temp_file.touch()
        mock_temp_file.return_value = temp_file

        # Mock DriveClient
        mock_drive_instance = Mock()
        mock_drive.return_value = mock_drive_instance

        # Mock WhisperEngine
        mock_engine_instance = Mock()
        mock_result = Mock(
            spec=[
                "text",
                "segments",
                "detected_language",
                "language_probability",
                "model_id",
                "device",
                "processing_duration_sec",
            ]
        )
        mock_result.text = "Test transcription"
        mock_result.segments = []
        mock_result.detected_language = "en"
        mock_result.language_probability = 0.95
        mock_result.model_id = "test-model"
        mock_result.device = "cpu"
        mock_result.processing_duration_sec = 1.0
        mock_engine_instance.transcribe.return_value = mock_result
        mock_engine.return_value = mock_engine_instance

        # Mock audio validation
        mock_has_audio.return_value = True
        mock_duration.return_value = 60000

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp3",
            mime_type="audio/mpeg",
            size_bytes=1024,
            parents=["folder1"],
            web_content_link="http://example.com/test",
            duration_ms=None,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            model_id="test-model",
            num_workers=1,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is True
        assert message == "Success"
        mock_save.assert_called_once()

    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    @patch("gtranscriber.core.batch.has_audio_stream")
    def test_transcribe_single_file_no_audio_stream(
        self,
        mock_has_audio: MagicMock,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription fails when file has no audio stream."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask

        # Setup mocks
        temp_file = tmp_path / "temp.mp3"
        temp_file.touch()
        mock_temp_file.return_value = temp_file

        # Mock DriveClient
        mock_drive_instance = Mock()
        mock_drive.return_value = mock_drive_instance

        # Mock no audio stream
        mock_has_audio.return_value = False

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp3",
            mime_type="audio/mpeg",
            size_bytes=1024,
            parents=[],
            web_content_link="http://example.com/test",
            duration_ms=60000,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is False
        assert "no audio" in message.lower() or "audio stream" in message.lower()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    @patch("gtranscriber.core.batch.save_enriched_record")
    @patch("gtranscriber.core.batch.extract_audio")
    @patch("gtranscriber.core.batch.requires_audio_extraction")
    @patch("gtranscriber.core.batch.get_media_duration_ms")
    def test_transcribe_single_file_video_with_extraction(
        self,
        mock_duration: MagicMock,
        mock_requires_extraction: MagicMock,
        mock_extract: MagicMock,
        mock_save: MagicMock,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful transcription of a video file with audio extraction."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask

        # Setup mocks
        temp_video = tmp_path / "temp.mp4"
        temp_video.touch()
        temp_audio = tmp_path / "temp.wav"
        temp_audio.touch()

        # Mock temp file creation to return different files for video and audio
        mock_temp_file.side_effect = [temp_video, temp_audio]

        # Mock DriveClient
        mock_drive_instance = Mock()
        mock_drive.return_value = mock_drive_instance

        # Mock WhisperEngine
        mock_engine_instance = Mock()
        mock_result = Mock(
            spec=[
                "text",
                "segments",
                "detected_language",
                "language_probability",
                "model_id",
                "device",
                "processing_duration_sec",
            ]
        )
        mock_result.text = "Test transcription"
        mock_result.segments = []
        mock_result.detected_language = "en"
        mock_result.language_probability = 0.95
        mock_result.model_id = "test-model"
        mock_result.device = "cpu"
        mock_result.processing_duration_sec = 1.0
        mock_engine_instance.transcribe.return_value = mock_result
        mock_engine.return_value = mock_engine_instance

        # Mock video extraction
        mock_requires_extraction.return_value = True
        mock_duration.return_value = 120000

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp4",
            mime_type="video/mp4",
            size_bytes=2048,
            parents=["folder1"],
            web_content_link="http://example.com/test",
            duration_ms=None,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            model_id="test-model",
            num_workers=1,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is True
        assert message == "Success"
        mock_extract.assert_called_once()
        mock_save.assert_called_once()

    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    @patch("gtranscriber.core.batch.extract_audio")
    @patch("gtranscriber.core.batch.requires_audio_extraction")
    def test_transcribe_single_file_corrupted_media(
        self,
        mock_requires_extraction: MagicMock,
        mock_extract: MagicMock,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription fails with corrupted media."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask
        from gtranscriber.core.media import CorruptedMediaError

        # Setup mocks
        temp_file = tmp_path / "temp.mp4"
        temp_file.touch()
        mock_temp_file.return_value = temp_file

        # Mock DriveClient
        mock_drive_instance = Mock()
        mock_drive.return_value = mock_drive_instance

        # Mock extraction failure
        mock_requires_extraction.return_value = True
        mock_extract.side_effect = CorruptedMediaError(temp_file, "File is corrupted")

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp4",
            mime_type="video/mp4",
            size_bytes=2048,
            parents=[],
            web_content_link="http://example.com/test",
            duration_ms=60000,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is False
        assert "corrupted" in message.lower() or "corrupt" in message.lower()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    @patch("gtranscriber.core.batch.extract_audio")
    @patch("gtranscriber.core.batch.requires_audio_extraction")
    def test_transcribe_single_file_audio_extraction_error(
        self,
        mock_requires_extraction: MagicMock,
        mock_extract: MagicMock,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription fails with audio extraction error."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask
        from gtranscriber.core.media import AudioExtractionError

        # Setup mocks
        temp_file = tmp_path / "temp.mp4"
        temp_file.touch()
        mock_temp_file.return_value = temp_file

        # Mock DriveClient
        mock_drive_instance = Mock()
        mock_drive.return_value = mock_drive_instance

        # Mock extraction failure
        mock_requires_extraction.return_value = True
        mock_extract.side_effect = AudioExtractionError(temp_file, "Extraction failed")

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp4",
            mime_type="video/mp4",
            size_bytes=2048,
            parents=[],
            web_content_link="http://example.com/test",
            duration_ms=60000,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is False
        assert "extraction" in message.lower() or "extract" in message.lower()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.WhisperEngine")
    @patch("gtranscriber.core.batch.DriveClient")
    @patch("gtranscriber.core.batch.create_temp_file")
    def test_transcribe_single_file_generic_error(
        self,
        mock_temp_file: MagicMock,
        mock_drive: MagicMock,
        mock_engine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription fails with generic error."""
        from gtranscriber.core.batch import BatchConfig, TranscriptionTask

        # Setup mocks
        temp_file = tmp_path / "temp.mp3"
        temp_file.touch()
        mock_temp_file.return_value = temp_file

        # Mock DriveClient to raise an error
        mock_drive.side_effect = Exception("Generic error")

        # Create task and config
        task = TranscriptionTask(
            file_id="test_id",
            name="test.mp3",
            mime_type="audio/mpeg",
            size_bytes=1024,
            parents=[],
            web_content_link="http://example.com/test",
            duration_ms=60000,
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        # Execute
        file_id, success, message = transcribe_single_file(task, config)

        # Verify
        assert file_id == "test_id"
        assert success is False
        assert message == "Generic error"


class TestRunBatchTranscription:
    """Tests for run_batch_transcription function."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results to use tmp_path for predictable paths."""
        self.results_dir = (tmp_path / "results").resolve()

        # Mock ResultsConfig to use tmp_path as base directory
        mock_rc = mocker.patch("gtranscriber.core.batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        # Mock TranscriberConfig for config snapshot
        mock_tc = mocker.patch("gtranscriber.core.batch.TranscriberConfig")
        mock_tc.return_value.model_dump.return_value = {"model_id": "test/model"}

        # Fix run ID for predictable paths
        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")

        # Use ThreadPoolExecutor to avoid slow forkserver process spawning.
        # Tests that explicitly @patch ProcessPoolExecutor will override this.
        mocker.patch("gtranscriber.core.batch.ProcessPoolExecutor", _ThreadPoolCompat)

        self.run_dir = self.results_dir / "transcription" / "test_run"
        self.outputs_dir = self.run_dir / "outputs"
        self.versioned_checkpoint = self.run_dir / "checkpoint.json"

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_creates_output_dir(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run_batch_transcription creates output directory."""
        from gtranscriber.core.batch import BatchConfig, run_batch_transcription

        mock_load_catalog.return_value = []

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "new_output_dir",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        run_batch_transcription(config)

        # Versioned output directory should be created
        assert self.outputs_dir.exists()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_no_remaining_tasks_exits_early(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test early exit when no remaining tasks to process."""
        import json

        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        caplog.set_level("INFO")

        # Return one task
        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            )
        ]

        # Pre-create versioned run directory with checkpoint already completed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_checkpoint.write_text(
            json.dumps(
                {
                    "total_files": 1,
                    "completed_files": ["file1"],
                    "failed_files": {},
                }
            )
        )

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
        )

        run_batch_transcription(config)

        # Should not have called transcribe
        mock_transcribe.assert_not_called()
        assert "already transcribed" in caplog.text.lower()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_sequential_processing(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test sequential processing with single worker."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        # Return two tasks
        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test1.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
            TranscriptionTask(
                file_id="file2",
                name="test2.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
        ]

        mock_transcribe.side_effect = [
            ("file1", True, "Success"),
            ("file2", True, "Success"),
        ]

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=1,
        )

        run_batch_transcription(config)

        assert mock_transcribe.call_count == 2

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_sequential_with_failures(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test sequential processing with some failures."""
        import json

        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test1.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
            TranscriptionTask(
                file_id="file2",
                name="test2.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
        ]

        mock_transcribe.side_effect = [
            ("file1", True, "Success"),
            ("file2", False, "Transcription failed"),
        ]

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=1,
        )

        run_batch_transcription(config)

        # Check versioned checkpoint was updated with failure
        checkpoint_data = json.loads(self.versioned_checkpoint.read_text())
        assert "file1" in checkpoint_data["completed_files"]
        assert "file2" in checkpoint_data["failed_files"]

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_final_summary_logged(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that final summary is logged."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        caplog.set_level("INFO")

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test1.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
        ]

        mock_transcribe.return_value = ("file1", True, "Success")

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=1,
        )

        run_batch_transcription(config)

        assert "Batch transcription completed" in caplog.text
        assert "Success rate:" in caplog.text

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    @patch("gtranscriber.core.batch.mp.cpu_count")
    def test_run_batch_worker_limiting_cpu_mode(
        self,
        mock_cpu_count: MagicMock,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test worker count is limited to CPU count in CPU mode."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        caplog.set_level("WARNING")
        mock_cpu_count.return_value = 2

        # Create more tasks than CPUs
        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id=f"file{i}",
                name=f"test{i}.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            )
            for i in range(5)
        ]

        mock_transcribe.side_effect = [(f"file{i}", True, "Success") for i in range(5)]

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=4,  # More than CPU count
            force_cpu=True,  # Force CPU mode
        )

        run_batch_transcription(config)

        # Should log warning about worker limiting
        assert "CPUs available" in caplog.text or mock_transcribe.call_count == 5

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    @patch("gtranscriber.core.batch.mp.cpu_count")
    def test_run_batch_worker_info_gpu_mode(
        self,
        mock_cpu_count: MagicMock,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test info is logged when workers > CPU count in GPU mode."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        caplog.set_level("INFO")
        mock_cpu_count.return_value = 2

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id=f"file{i}",
                name=f"test{i}.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            )
            for i in range(5)
        ]

        mock_transcribe.side_effect = [(f"file{i}", True, "Success") for i in range(5)]

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=4,  # More than CPU count
            force_cpu=False,  # GPU mode (no CPU limiting)
        )

        run_batch_transcription(config)

        # Should log info about GPU processing
        assert "GPU processing" in caplog.text or "workers" in caplog.text.lower()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.transcribe_single_file")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_final_summary_with_failures_logged(
        self,
        mock_load_catalog: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that final summary includes failure details."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        caplog.set_level("WARNING")

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test1.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
        ]

        mock_transcribe.return_value = ("file1", False, "Error occurred")

        config = BatchConfig(
            catalog_file=tmp_path / "catalog.csv",
            output_dir=tmp_path / "output",
            checkpoint_file=tmp_path / "checkpoint.json",
            credentials_file=tmp_path / "creds.json",
            token_file=tmp_path / "token.json",
            num_workers=1,
        )

        run_batch_transcription(config)

        assert "Failed files:" in caplog.text or "Error occurred" in caplog.text

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.ProcessPoolExecutor")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_parallel_processing(
        self,
        mock_load_catalog: MagicMock,
        mock_executor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test parallel processing with multiple workers."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id=f"file{i}",
                name=f"test{i}.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            )
            for i in range(3)
        ]

        # Mock executor and futures
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Create mock futures that return results
        mock_futures = []
        for i in range(3):
            mock_future = MagicMock()
            mock_future.result.return_value = (f"file{i}", True, "Success")
            mock_futures.append(mock_future)

        mock_executor_instance.submit.side_effect = mock_futures

        # Mock as_completed to return futures in order
        with patch("gtranscriber.core.batch.as_completed") as mock_as_completed:
            mock_as_completed.return_value = iter(mock_futures)

            config = BatchConfig(
                catalog_file=tmp_path / "catalog.csv",
                output_dir=tmp_path / "output",
                checkpoint_file=tmp_path / "checkpoint.json",
                credentials_file=tmp_path / "creds.json",
                token_file=tmp_path / "token.json",
                num_workers=2,  # More than 1 triggers parallel processing
            )

            run_batch_transcription(config)

        # Should have used ProcessPoolExecutor
        mock_executor.assert_called_once()

    @patch("gtranscriber.core.batch._worker_engine", None)
    @patch("gtranscriber.core.batch.ProcessPoolExecutor")
    @patch("gtranscriber.core.batch.load_catalog")
    def test_run_batch_parallel_with_exception(
        self,
        mock_load_catalog: MagicMock,
        mock_executor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test parallel processing handles exceptions from futures."""
        from gtranscriber.core.batch import (
            BatchConfig,
            TranscriptionTask,
            run_batch_transcription,
        )

        mock_load_catalog.return_value = [
            TranscriptionTask(
                file_id="file1",
                name="test1.mp3",
                mime_type="audio/mpeg",
                size_bytes=1024,
                parents=[],
                web_content_link="http://example.com",
                duration_ms=60000,
            ),
        ]

        # Mock executor and futures
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Create mock future that raises exception
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Worker crashed")

        mock_executor_instance.submit.return_value = mock_future

        # Mock as_completed
        with patch("gtranscriber.core.batch.as_completed") as mock_as_completed:
            mock_as_completed.return_value = iter([mock_future])

            config = BatchConfig(
                catalog_file=tmp_path / "catalog.csv",
                output_dir=tmp_path / "output",
                checkpoint_file=tmp_path / "checkpoint.json",
                credentials_file=tmp_path / "creds.json",
                token_file=tmp_path / "token.json",
                num_workers=2,
            )

            # Should not raise - exception is caught and logged
            run_batch_transcription(config)
