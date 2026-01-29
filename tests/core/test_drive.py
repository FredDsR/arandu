"""Tests for Google Drive integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytest_mock import MockerFixture

from gtranscriber.core.drive import (
    DownloadError,
    EmptyDownloadError,
    IncompleteDownloadError,
)


class TestDriveExceptions:
    """Tests for custom Drive exceptions."""

    def test_download_error(self) -> None:
        """Test base DownloadError exception."""
        error = DownloadError("Download failed")
        assert str(error) == "Download failed"
        assert isinstance(error, Exception)

    def test_incomplete_download_error(self) -> None:
        """Test IncompleteDownloadError with file size mismatch."""
        error = IncompleteDownloadError(
            file_id="abc123",
            file_name="test.mp4",
            expected_size=10485760,  # 10 MB
            actual_size=5242880,  # 5 MB
            destination=Path("/tmp/test.mp4"),
        )

        assert error.file_id == "abc123"
        assert error.file_name == "test.mp4"
        assert error.expected_size == 10485760
        assert error.actual_size == 5242880
        assert error.destination == Path("/tmp/test.mp4")

        # Check error message contains useful info
        error_msg = str(error)
        assert "test.mp4" in error_msg
        assert "abc123" in error_msg
        assert "10.00 MB" in error_msg
        assert "5.00 MB" in error_msg
        assert "50.0%" in error_msg  # Percentage calculation

    def test_incomplete_download_error_zero_expected(self) -> None:
        """Test IncompleteDownloadError with zero expected size."""
        error = IncompleteDownloadError(
            file_id="abc123",
            file_name="test.mp4",
            expected_size=0,
            actual_size=100,
            destination=Path("/tmp/test.mp4"),
        )

        # Should handle division by zero gracefully
        assert error.expected_size == 0
        assert error.actual_size == 100

    def test_empty_download_error(self) -> None:
        """Test EmptyDownloadError exception."""
        error = EmptyDownloadError(
            file_id="abc123",
            file_name="test.mp4",
            destination=Path("/tmp/test.mp4"),
        )

        assert error.file_id == "abc123"
        assert error.file_name == "test.mp4"
        assert error.destination == Path("/tmp/test.mp4")

        error_msg = str(error)
        assert "empty file" in error_msg.lower()
        assert "test.mp4" in error_msg
        assert "abc123" in error_msg


class TestDriveClientInitialization:
    """Tests for DriveClient initialization."""

    def test_import_drive_client(self) -> None:
        """Test that DriveClient can be imported."""
        from gtranscriber.core.drive import DriveClient

        assert DriveClient is not None


class TestDriveClientMethods:
    """Tests for DriveClient methods with mocking."""

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    def test_drive_client_service_initialization(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that DriveClient initializes Google Drive service."""
        from gtranscriber.core.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        # Create token file
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        # Initialize client
        client = DriveClient(
            credentials_file=str(tmp_path / "credentials.json"),
            token_file=str(token_file),
        )

        # Verify service was built
        assert client.service is not None
        mock_build.assert_called_once()

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    def test_drive_client_without_valid_token(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test DriveClient initialization without valid token."""
        from gtranscriber.core.drive import DriveClient

        # Setup mocks - no valid token
        mock_credentials.from_authorized_user_file.side_effect = FileNotFoundError

        token_file = tmp_path / "token.json"

        # Should handle missing token gracefully (would trigger OAuth flow)
        # In actual usage, this would prompt for authentication
        try:
            client = DriveClient(
                credentials_file=str(tmp_path / "credentials.json"),
                token_file=str(token_file),
            )
        except FileNotFoundError:
            # Expected if credentials file doesn't exist
            pass


class TestDriveRetryLogic:
    """Tests for retry logic in Drive operations."""

    def test_download_error_inheritance(self) -> None:
        """Test that custom errors inherit from DownloadError."""
        incomplete_error = IncompleteDownloadError(
            "id", "name", 100, 50, Path("/tmp/test")
        )
        empty_error = EmptyDownloadError("id", "name", Path("/tmp/test"))

        assert isinstance(incomplete_error, DownloadError)
        assert isinstance(empty_error, DownloadError)


class TestDriveHelperFunctions:
    """Tests for helper functions in drive module."""

    def test_incomplete_download_percentage_calculation(self) -> None:
        """Test percentage calculation in IncompleteDownloadError."""
        # 50% downloaded
        error = IncompleteDownloadError(
            file_id="test",
            file_name="file.mp4",
            expected_size=1000,
            actual_size=500,
            destination=Path("/tmp/file.mp4"),
        )

        error_msg = str(error)
        assert "50.0%" in error_msg

    def test_incomplete_download_mb_formatting(self) -> None:
        """Test MB formatting in error messages."""
        error = IncompleteDownloadError(
            file_id="test",
            file_name="file.mp4",
            expected_size=10 * 1024 * 1024,  # 10 MB
            actual_size=5 * 1024 * 1024,  # 5 MB
            destination=Path("/tmp/file.mp4"),
        )

        error_msg = str(error)
        assert "10.00 MB" in error_msg
        assert "5.00 MB" in error_msg

    def test_error_messages_include_retry_hint(self) -> None:
        """Test that error messages include retry information."""
        incomplete_error = IncompleteDownloadError(
            "id", "file.mp4", 1000, 500, Path("/tmp/file.mp4")
        )

        assert "retry" in str(incomplete_error).lower()


class TestValidateFileId:
    """Tests for _validate_file_id helper function."""

    def test_valid_file_id_alphanumeric(self) -> None:
        """Test validation of valid alphanumeric file ID."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("abc123DEF456") is True

    def test_valid_file_id_with_dash(self) -> None:
        """Test validation of file ID with dashes."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("abc-123-def") is True

    def test_valid_file_id_with_underscore(self) -> None:
        """Test validation of file ID with underscores."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("abc_123_def") is True

    def test_invalid_file_id_special_chars(self) -> None:
        """Test validation fails for file ID with invalid characters."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("abc@123") is False
        assert _validate_file_id("abc#123") is False
        assert _validate_file_id("abc.123") is False

    def test_invalid_file_id_empty(self) -> None:
        """Test validation fails for empty file ID."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("") is False

    def test_invalid_file_id_spaces(self) -> None:
        """Test validation fails for file ID with spaces."""
        from gtranscriber.core.drive import _validate_file_id

        assert _validate_file_id("abc 123") is False


class TestCheckFilePermissions:
    """Tests for _check_file_permissions helper function."""

    def test_check_permissions_nonexistent_file(self, tmp_path: Path) -> None:
        """Test checking permissions for nonexistent file."""
        from gtranscriber.core.drive import _check_file_permissions

        # Should not raise an exception for nonexistent file
        _check_file_permissions(tmp_path / "nonexistent.txt")

    def test_check_permissions_existing_file(self, tmp_path: Path) -> None:
        """Test checking permissions for existing file."""
        from gtranscriber.core.drive import _check_file_permissions

        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        # Should not raise an exception
        _check_file_permissions(file_path)


class TestDriveClientAuthentication:
    """Tests for DriveClient authentication logic."""

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    @patch("gtranscriber.core.drive.InstalledAppFlow")
    def test_authenticate_new_user(
        self,
        mock_flow: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication flow for new user without token."""
        from gtranscriber.core.drive import DriveClient

        # Setup mocks
        mock_credentials.from_authorized_user_file.side_effect = FileNotFoundError
        mock_creds = Mock()
        mock_creds.valid = True
        mock_flow_instance = Mock()
        mock_flow_instance.run_local_server.return_value = mock_creds
        mock_flow.from_client_secrets_file.return_value = mock_flow_instance

        credentials_file = tmp_path / "credentials.json"
        credentials_file.write_text('{"installed": {}}')
        token_file = tmp_path / "token.json"

        # Initialize client
        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        # This would trigger authentication in real scenario
        # but we're testing the initialization
        assert client.credentials_file == str(credentials_file)
        assert client.token_file == str(token_file)

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    def test_authenticate_existing_valid_token(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication with existing valid token."""
        from gtranscriber.core.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        credentials_file = tmp_path / "credentials.json"
        credentials_file.write_text('{"installed": {}}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        # Access service to trigger authentication
        _ = client.service

        mock_credentials.from_authorized_user_file.assert_called_once()
        mock_build.assert_called_once()


class TestDriveScopes:
    """Tests for Drive API scopes constant."""

    def test_scopes_constant(self) -> None:
        """Test that SCOPES constant is defined correctly."""
        from gtranscriber.core.drive import SCOPES

        assert isinstance(SCOPES, list)
        assert len(SCOPES) > 0
        assert "drive" in SCOPES[0].lower()


class TestNoAudioStreamError:
    """Tests for NoAudioStreamError exception."""

    def test_no_audio_stream_error_creation(self) -> None:
        """Test creating NoAudioStreamError."""
        from gtranscriber.core.drive import NoAudioStreamError

        error = NoAudioStreamError("file123", "test.mp4")

        assert error.file_id == "file123"
        assert error.file_name == "test.mp4"
        assert "test.mp4" in str(error)
        assert "file123" in str(error)
        assert "no audio" in str(error).lower()


class TestDriveClientConfiguration:
    """Tests for DriveClient configuration handling."""

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    def test_client_with_config_object(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test DriveClient initialization with TranscriberConfig object."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.drive import DriveClient

        config = TranscriberConfig(
            credentials="custom_creds.json",
            token="custom_token.json",
        )

        client = DriveClient(config=config)

        assert client.credentials_file == "custom_creds.json"
        assert client.token_file == "custom_token.json"

    @patch("gtranscriber.core.drive.build")
    @patch("gtranscriber.core.drive.Credentials")
    def test_client_explicit_paths_override_config(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that explicit paths override config values."""
        from gtranscriber.config import TranscriberConfig
        from gtranscriber.core.drive import DriveClient

        config = TranscriberConfig(
            credentials="config_creds.json",
            token="config_token.json",
        )

        client = DriveClient(
            credentials_file="explicit_creds.json",
            token_file="explicit_token.json",
            config=config,
        )

        assert client.credentials_file == "explicit_creds.json"
        assert client.token_file == "explicit_token.json"
