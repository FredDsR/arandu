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
