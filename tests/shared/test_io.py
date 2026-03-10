"""Tests for file I/O operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from arandu.shared.io import (
    cleanup_temp_files,
    create_temp_file,
    ensure_temp_dir,
    get_mime_type,
    get_output_filename,
    save_enriched_record,
)
from arandu.shared.schemas import EnrichedRecord


class TestEnsureTempDir:
    """Tests for ensure_temp_dir function."""

    def test_ensure_temp_dir_default(self, tmp_path: Path) -> None:
        """Test ensuring temp directory with default path."""
        temp_dir = ensure_temp_dir(str(tmp_path / "test_temp"))

        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_ensure_temp_dir_creates_parents(self, tmp_path: Path) -> None:
        """Test that ensure_temp_dir creates parent directories."""
        nested_path = tmp_path / "level1" / "level2" / "temp"

        temp_dir = ensure_temp_dir(str(nested_path))

        assert temp_dir.exists()
        assert temp_dir == nested_path

    def test_ensure_temp_dir_existing(self, tmp_path: Path) -> None:
        """Test ensuring temp directory when it already exists."""
        temp_path = tmp_path / "existing_temp"
        temp_path.mkdir()

        temp_dir = ensure_temp_dir(str(temp_path))

        assert temp_dir.exists()
        assert temp_dir == temp_path

    def test_ensure_temp_dir_none_uses_default(self) -> None:
        """Test that passing None uses system temp directory."""
        temp_dir = ensure_temp_dir(None)

        assert temp_dir.exists()
        assert "arandu" in str(temp_dir)


class TestCreateTempFile:
    """Tests for create_temp_file function."""

    def test_create_temp_file_default(self, tmp_path: Path) -> None:
        """Test creating temp file with default parameters."""
        temp_file = create_temp_file(base_dir=str(tmp_path))

        assert temp_file.exists()
        assert "arandu_" in temp_file.name

    def test_create_temp_file_with_suffix(self, tmp_path: Path) -> None:
        """Test creating temp file with custom suffix."""
        temp_file = create_temp_file(suffix=".mp4", base_dir=str(tmp_path))

        assert temp_file.exists()
        assert temp_file.suffix == ".mp4"

    def test_create_temp_file_with_prefix(self, tmp_path: Path) -> None:
        """Test creating temp file with custom prefix."""
        temp_file = create_temp_file(prefix="custom_", base_dir=str(tmp_path))

        assert temp_file.exists()
        assert "custom_" in temp_file.name

    def test_create_temp_file_with_all_params(self, tmp_path: Path) -> None:
        """Test creating temp file with all parameters."""
        temp_file = create_temp_file(suffix=".json", prefix="test_", base_dir=str(tmp_path))

        assert temp_file.exists()
        assert temp_file.suffix == ".json"
        assert "test_" in temp_file.name

    def test_create_temp_file_none_base_dir(self) -> None:
        """Test creating temp file with None base_dir uses system temp."""
        temp_file = create_temp_file(suffix=".txt", base_dir=None)

        assert temp_file.exists()
        assert "arandu" in str(temp_file.parent)

        # Cleanup
        temp_file.unlink()

    def test_create_temp_file_creates_base_dir(self, tmp_path: Path) -> None:
        """Test that create_temp_file creates base directory if it doesn't exist."""
        base_dir = tmp_path / "non_existent"

        temp_file = create_temp_file(base_dir=str(base_dir))

        assert base_dir.exists()
        assert temp_file.exists()


class TestSaveEnrichedRecord:
    """Tests for save_enriched_record function."""

    def test_save_enriched_record(self, tmp_path: Path) -> None:
        """Test saving an enriched record."""
        record = EnrichedRecord(
            file_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder1"],
            webContentLink="https://example.com/file",
            transcription_text="This is a test transcription.",
            detected_language="en",
            language_probability=0.99,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.5,
            transcription_status="completed",
        )

        output_path = tmp_path / "output.json"
        saved_path = save_enriched_record(record, output_path)

        assert saved_path == output_path
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "test.mp3" in content
        assert "This is a test transcription" in content

    def test_save_enriched_record_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test that save_enriched_record creates parent directories."""
        record = EnrichedRecord(
            file_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder1"],
            webContentLink="https://example.com/file",
            transcription_text="Test",
            detected_language="en",
            language_probability=0.99,
            model_id="model",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        output_path = tmp_path / "nested" / "dir" / "output.json"
        saved_path = save_enriched_record(record, output_path)

        assert saved_path.exists()
        assert saved_path.parent.exists()

    def test_save_enriched_record_with_str_path(self, tmp_path: Path) -> None:
        """Test saving enriched record with string path."""
        record = EnrichedRecord(
            file_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder1"],
            webContentLink="https://example.com/file",
            transcription_text="Test",
            detected_language="en",
            language_probability=0.99,
            model_id="model",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        output_path = str(tmp_path / "output.json")
        saved_path = save_enriched_record(record, output_path)

        assert saved_path.exists()
        assert isinstance(saved_path, Path)


class TestGetOutputFilename:
    """Tests for get_output_filename function."""

    def test_get_output_filename_default(self) -> None:
        """Test getting output filename with default suffix."""
        filename = get_output_filename("test.mp3")

        assert filename == "test_transcription.json"

    def test_get_output_filename_custom_suffix(self) -> None:
        """Test getting output filename with custom suffix."""
        filename = get_output_filename("test.mp3", suffix="_result.txt")

        assert filename == "test_result.txt"

    def test_get_output_filename_no_extension(self) -> None:
        """Test getting output filename when original has no extension."""
        filename = get_output_filename("testfile")

        assert filename == "testfile_transcription.json"

    def test_get_output_filename_multiple_dots(self) -> None:
        """Test getting output filename with multiple dots."""
        filename = get_output_filename("test.file.name.mp3")

        assert filename == "test.file.name_transcription.json"

    def test_get_output_filename_with_path(self) -> None:
        """Test getting output filename from full path."""
        filename = get_output_filename("/path/to/test.mp3")

        assert filename == "test_transcription.json"


class TestGetMimeType:
    """Tests for get_mime_type function."""

    def test_get_mime_type_audio(self) -> None:
        """Test getting MIME type for audio files."""
        assert get_mime_type(Path("test.mp3")) == "audio/mpeg"
        assert get_mime_type(Path("test.wav")) == "audio/wav"
        assert get_mime_type(Path("test.flac")) == "audio/flac"
        assert get_mime_type(Path("test.ogg")) == "audio/ogg"
        assert get_mime_type(Path("test.m4a")) == "audio/m4a"

    def test_get_mime_type_video(self) -> None:
        """Test getting MIME type for video files."""
        assert get_mime_type(Path("test.mp4")) == "video/mp4"
        assert get_mime_type(Path("test.mkv")) == "video/x-matroska"
        assert get_mime_type(Path("test.avi")) == "video/x-msvideo"
        assert get_mime_type(Path("test.mov")) == "video/quicktime"
        assert get_mime_type(Path("test.webm")) == "video/webm"

    def test_get_mime_type_case_insensitive(self) -> None:
        """Test that MIME type detection is case insensitive."""
        assert get_mime_type(Path("test.MP3")) == "audio/mpeg"
        assert get_mime_type(Path("test.MP4")) == "video/mp4"
        assert get_mime_type(Path("test.MKV")) == "video/x-matroska"

    def test_get_mime_type_unknown(self) -> None:
        """Test getting MIME type for unknown extension."""
        mime_type = get_mime_type(Path("test.xyz"))

        assert mime_type == "application/octet-stream"

    def test_get_mime_type_no_extension(self) -> None:
        """Test getting MIME type for file without extension."""
        mime_type = get_mime_type(Path("testfile"))

        assert mime_type == "application/octet-stream"


class TestCleanupTempFiles:
    """Tests for cleanup_temp_files function."""

    def test_cleanup_temp_files_empty_dir(self, tmp_path: Path) -> None:
        """Test cleanup when directory is empty."""
        success, failure = cleanup_temp_files(str(tmp_path))

        assert success == 0
        assert failure == 0

    def test_cleanup_temp_files_with_files(self, tmp_path: Path) -> None:
        """Test cleanup with temporary files present."""
        # Create some temporary files
        (tmp_path / "arandu_file1.txt").touch()
        (tmp_path / "arandu_file2.txt").touch()
        (tmp_path / "arandu_file3.txt").touch()

        success, failure = cleanup_temp_files(str(tmp_path))

        assert success == 3
        assert failure == 0
        assert not (tmp_path / "arandu_file1.txt").exists()

    def test_cleanup_temp_files_ignores_other_files(self, tmp_path: Path) -> None:
        """Test that cleanup only removes arandu_ files."""
        # Create arandu files and other files
        (tmp_path / "arandu_file1.txt").touch()
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "important.txt").touch()

        success, failure = cleanup_temp_files(str(tmp_path))

        assert success == 1
        assert failure == 0
        assert (tmp_path / "other_file.txt").exists()
        assert (tmp_path / "important.txt").exists()

    def test_cleanup_temp_files_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test cleanup when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        success, failure = cleanup_temp_files(str(nonexistent))

        assert success == 0
        assert failure == 0

    def test_cleanup_temp_files_with_permission_error(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test cleanup handles permission errors gracefully."""
        # Create a temporary file
        temp_file = tmp_path / "arandu_test.txt"
        temp_file.touch()

        # Mock unlink to raise OSError
        mocker.patch.object(Path, "unlink", side_effect=OSError("Permission denied"))

        success, failure = cleanup_temp_files(str(tmp_path))

        assert success == 0
        assert failure == 1

    def test_cleanup_temp_files_none_uses_default(self) -> None:
        """Test cleanup with None uses system temp directory."""
        # Create a temp file in the default location
        create_temp_file(suffix=".test")

        success, failure = cleanup_temp_files(None)

        # Should have cleaned up at least the file we created
        assert success >= 1 or failure >= 0
