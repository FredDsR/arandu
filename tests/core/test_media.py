"""Tests for media file utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from gtranscriber.core.media import (
    AudioExtractionError,
    CorruptedMediaError,
    MediaError,
    UnsupportedCodecError,
    get_audio_stream_info,
    get_media_duration_ms,
    has_audio_stream,
)


class TestMediaExceptions:
    """Tests for custom media exceptions."""

    def test_media_error(self) -> None:
        """Test base MediaError exception."""
        error = MediaError("Test error")
        assert str(error) == "Test error"

    def test_corrupted_media_error(self) -> None:
        """Test CorruptedMediaError exception."""
        error = CorruptedMediaError("test.mp4", "Invalid header")

        assert error.file_path == Path("test.mp4")
        assert error.reason == "Invalid header"
        assert "Corrupted media file" in str(error)
        assert "test.mp4" in str(error)
        assert "Invalid header" in str(error)

    def test_audio_extraction_error(self) -> None:
        """Test AudioExtractionError exception."""
        error = AudioExtractionError("test.mp4", "No audio stream")

        assert error.file_path == Path("test.mp4")
        assert error.reason == "No audio stream"
        assert "Failed to extract audio" in str(error)
        assert "test.mp4" in str(error)
        assert "No audio stream" in str(error)

    def test_unsupported_codec_error(self) -> None:
        """Test UnsupportedCodecError exception."""
        error = UnsupportedCodecError("test.mp4", "opus")

        assert error.file_path == Path("test.mp4")
        assert error.codec == "opus"
        assert "Unsupported audio codec" in str(error)
        assert "opus" in str(error)
        assert "test.mp4" in str(error)


class TestHasAudioStream:
    """Tests for has_audio_stream function."""

    def test_has_audio_stream_success(self, mocker: MockerFixture) -> None:
        """Test detecting audio stream in media file."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"streams": [{"codec_type": "audio"}]})
        mock_run = mocker.patch("subprocess.run", return_value=mock_result)

        result = has_audio_stream("test.mp4")

        assert result is True
        mock_run.assert_called_once()
        assert "ffprobe" in mock_run.call_args[0][0]

    def test_has_audio_stream_no_streams(self, mocker: MockerFixture) -> None:
        """Test when media file has no audio streams."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"streams": []})
        mocker.patch("subprocess.run", return_value=mock_result)

        result = has_audio_stream("test.mp4")

        assert result is False

    def test_has_audio_stream_subprocess_error(self, mocker: MockerFixture) -> None:
        """Test handling subprocess.CalledProcessError."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe", stderr="Error"),
        )

        result = has_audio_stream("test.mp4")

        assert result is False

    def test_has_audio_stream_timeout(self, mocker: MockerFixture) -> None:
        """Test handling subprocess timeout."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        )

        result = has_audio_stream("test.mp4")

        assert result is False

    def test_has_audio_stream_json_decode_error(self, mocker: MockerFixture) -> None:
        """Test handling invalid JSON response."""
        mock_result = Mock()
        mock_result.stdout = "invalid json"
        mocker.patch("subprocess.run", return_value=mock_result)

        result = has_audio_stream("test.mp4")

        assert result is False

    def test_has_audio_stream_unexpected_error(self, mocker: MockerFixture) -> None:
        """Test handling unexpected exceptions."""
        mocker.patch(
            "subprocess.run",
            side_effect=Exception("Unexpected error"),
        )

        result = has_audio_stream("test.mp4")

        assert result is False


class TestGetAudioStreamInfo:
    """Tests for get_audio_stream_info function."""

    def test_get_audio_stream_info_success(self, mocker: MockerFixture) -> None:
        """Test getting audio stream information."""
        mock_result = Mock()
        mock_result.stdout = json.dumps(
            {
                "streams": [
                    {
                        "codec_name": "aac",
                        "codec_type": "audio",
                        "sample_rate": "44100",
                        "channels": 2,
                        "duration": "120.5",
                    }
                ]
            }
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        info = get_audio_stream_info("test.mp4")

        assert info is not None
        assert info["codec_name"] == "aac"
        assert info["sample_rate"] == "44100"
        assert info["channels"] == 2

    def test_get_audio_stream_info_no_streams(self, mocker: MockerFixture) -> None:
        """Test when no audio streams are found."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"streams": []})
        mocker.patch("subprocess.run", return_value=mock_result)

        info = get_audio_stream_info("test.mp4")

        assert info is None

    def test_get_audio_stream_info_subprocess_error(self, mocker: MockerFixture) -> None:
        """Test handling subprocess error."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        )

        info = get_audio_stream_info("test.mp4")

        assert info is None

    def test_get_audio_stream_info_timeout(self, mocker: MockerFixture) -> None:
        """Test handling timeout."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        )

        info = get_audio_stream_info("test.mp4")

        assert info is None


class TestGetMediaDurationMs:
    """Tests for get_media_duration_ms function."""

    def test_get_duration_success(self, mocker: MockerFixture) -> None:
        """Test extracting media duration successfully."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {"duration": "123.456"}})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration == 123456  # 123.456 seconds = 123456 ms

    def test_get_duration_zero(self, mocker: MockerFixture) -> None:
        """Test when duration is zero."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {"duration": "0"}})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration is None  # Zero duration returns None

    def test_get_duration_no_duration_field(self, mocker: MockerFixture) -> None:
        """Test when duration field is missing."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {}})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_no_format_field(self, mocker: MockerFixture) -> None:
        """Test when format field is missing."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_subprocess_error(self, mocker: MockerFixture) -> None:
        """Test handling subprocess error."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        )

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_timeout(self, mocker: MockerFixture) -> None:
        """Test handling timeout."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        )

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_json_decode_error(self, mocker: MockerFixture) -> None:
        """Test handling invalid JSON."""
        mock_result = Mock()
        mock_result.stdout = "invalid json"
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_value_error(self, mocker: MockerFixture) -> None:
        """Test handling invalid duration value."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {"duration": "invalid"}})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms("test.mp4")

        assert duration is None

    def test_get_duration_ffprobe_command(self, mocker: MockerFixture) -> None:
        """Test that ffprobe is called with correct arguments."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {"duration": "100.0"}})
        mock_run = mocker.patch("subprocess.run", return_value=mock_result)

        get_media_duration_ms("test.mp4")

        # Verify command structure
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffprobe"
        assert "-v" in call_args
        assert "quiet" in call_args
        assert "-print_format" in call_args
        assert "json" in call_args
        assert "-show_format" in call_args
        assert "test.mp4" in call_args

    def test_get_duration_with_path_object(self, mocker: MockerFixture) -> None:
        """Test duration extraction with Path object."""
        mock_result = Mock()
        mock_result.stdout = json.dumps({"format": {"duration": "50.0"}})
        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_media_duration_ms(Path("test.mp4"))

        assert duration == 50000


class TestValidateMediaFile:
    """Tests for validate_media_file function."""

    def test_validate_media_file_success(self, mocker: MockerFixture) -> None:
        """Test validating a valid media file."""
        from gtranscriber.core.media import validate_media_file

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mocker.patch("subprocess.run", return_value=mock_result)

        # Should not raise any exception
        validate_media_file("test.mp4")

    def test_validate_media_file_moov_atom_not_found(
        self, mocker: MockerFixture
    ) -> None:
        """Test validation error for missing moov atom."""
        from gtranscriber.core.media import validate_media_file

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "moov atom not found"
        mocker.patch("subprocess.run", return_value=mock_result)

        with pytest.raises(CorruptedMediaError) as exc_info:
            validate_media_file("test.mp4")

        assert "moov atom" in str(exc_info.value).lower()

    def test_validate_media_file_invalid_data(self, mocker: MockerFixture) -> None:
        """Test validation error for invalid data."""
        from gtranscriber.core.media import validate_media_file

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found in stream"
        mocker.patch("subprocess.run", return_value=mock_result)

        with pytest.raises(CorruptedMediaError) as exc_info:
            validate_media_file("test.mp4")

        assert "Invalid data" in str(exc_info.value) or "damaged" in str(exc_info.value)

    def test_validate_media_file_timeout(self, mocker: MockerFixture) -> None:
        """Test validation timeout handling."""
        from gtranscriber.core.media import validate_media_file

        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        )

        with pytest.raises(CorruptedMediaError) as exc_info:
            validate_media_file("test.mp4")

        assert "timed out" in str(exc_info.value).lower()

    def test_validate_media_file_generic_error(self, mocker: MockerFixture) -> None:
        """Test validation with generic ffprobe error."""
        from gtranscriber.core.media import validate_media_file

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Some other error"
        mocker.patch("subprocess.run", return_value=mock_result)

        with pytest.raises(CorruptedMediaError) as exc_info:
            validate_media_file("test.mp4")

        assert "Some other error" in str(exc_info.value)


class TestRequiresAudioExtraction:
    """Tests for requires_audio_extraction function."""

    def test_requires_extraction_video_mp4(self) -> None:
        """Test that video/mp4 requires extraction."""
        from gtranscriber.core.media import requires_audio_extraction

        assert requires_audio_extraction("video/mp4") is True

    def test_requires_extraction_video_quicktime(self) -> None:
        """Test that video/quicktime requires extraction."""
        from gtranscriber.core.media import requires_audio_extraction

        assert requires_audio_extraction("video/quicktime") is True

    def test_no_extraction_audio_mpeg(self) -> None:
        """Test that audio/mpeg does not require extraction."""
        from gtranscriber.core.media import requires_audio_extraction

        assert requires_audio_extraction("audio/mpeg") is False

    def test_no_extraction_audio_wav(self) -> None:
        """Test that audio/wav does not require extraction."""
        from gtranscriber.core.media import requires_audio_extraction

        assert requires_audio_extraction("audio/wav") is False


class TestExtractAudio:
    """Tests for extract_audio function."""

    def test_extract_audio_success(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Test successful audio extraction."""
        from gtranscriber.core.media import extract_audio

        # Mock validate_media_file
        mocker.patch("gtranscriber.core.media.validate_media_file")

        # Mock has_audio_stream
        mocker.patch("gtranscriber.core.media.has_audio_stream", return_value=True)

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.wav"

        # Mock ffmpeg subprocess that creates the output file
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        def create_output_file(*args, **kwargs):
            output_file.write_bytes(b"fake audio data")
            return mock_result

        mocker.patch("subprocess.run", side_effect=create_output_file)

        result = extract_audio(input_file, output_file)

        assert result == output_file

    def test_extract_audio_no_audio_stream(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test extraction when no audio stream is found."""
        from gtranscriber.core.media import extract_audio

        mocker.patch("gtranscriber.core.media.validate_media_file")
        mocker.patch("gtranscriber.core.media.has_audio_stream", return_value=False)

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.wav"

        with pytest.raises(AudioExtractionError) as exc_info:
            extract_audio(input_file, output_file)

        assert "No audio stream" in str(exc_info.value)

    def test_extract_audio_timeout(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Test extraction timeout handling."""
        from gtranscriber.core.media import extract_audio

        mocker.patch("gtranscriber.core.media.validate_media_file")
        mocker.patch("gtranscriber.core.media.has_audio_stream", return_value=True)

        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 300),
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.wav"

        with pytest.raises(AudioExtractionError) as exc_info:
            extract_audio(input_file, output_file)

        assert "timed out" in str(exc_info.value).lower()

    def test_extract_audio_stereo(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test audio extraction with stereo output."""
        from gtranscriber.core.media import extract_audio

        mocker.patch("gtranscriber.core.media.validate_media_file")
        mocker.patch("gtranscriber.core.media.has_audio_stream", return_value=True)

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.wav"

        mock_result = Mock()
        mock_result.returncode = 0

        def create_output_file(*args, **kwargs):
            output_file.write_bytes(b"fake audio data")
            return mock_result

        mock_run = mocker.patch("subprocess.run", side_effect=create_output_file)

        extract_audio(input_file, output_file, mono=False)

        # Check that -ac 1 (mono) was not in the command
        call_args = mock_run.call_args[0][0]
        assert "-ac" not in call_args

    def test_extract_audio_custom_sample_rate(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test audio extraction with custom sample rate."""
        from gtranscriber.core.media import extract_audio

        mocker.patch("gtranscriber.core.media.validate_media_file")
        mocker.patch("gtranscriber.core.media.has_audio_stream", return_value=True)

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.wav"

        mock_result = Mock()
        mock_result.returncode = 0

        def create_output_file(*args, **kwargs):
            output_file.write_bytes(b"fake audio data")
            return mock_result

        mock_run = mocker.patch("subprocess.run", side_effect=create_output_file)

        extract_audio(input_file, output_file, sample_rate=48000)

        # Check sample rate in command
        call_args = mock_run.call_args[0][0]
        assert "48000" in call_args
