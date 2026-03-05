"""Tests for Google Drive integration."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from arandu.shared.drive import (
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
        from arandu.shared.drive import DriveClient

        assert DriveClient is not None


class TestDriveClientMethods:
    """Tests for DriveClient methods with mocking."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_drive_client_service_initialization(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that DriveClient initializes Google Drive service."""
        from arandu.shared.drive import DriveClient

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

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_drive_client_without_valid_token(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test DriveClient initialization without valid token."""
        from arandu.shared.drive import DriveClient

        # Setup mocks - no valid token
        mock_credentials.from_authorized_user_file.side_effect = FileNotFoundError

        token_file = tmp_path / "token.json"

        # Should handle missing token gracefully (would trigger OAuth flow)
        # In actual usage, this would prompt for authentication
        with contextlib.suppress(FileNotFoundError):
            DriveClient(
                credentials_file=str(tmp_path / "credentials.json"),
                token_file=str(token_file),
            )


class TestDriveRetryLogic:
    """Tests for retry logic in Drive operations."""

    def test_download_error_inheritance(self) -> None:
        """Test that custom errors inherit from DownloadError."""
        incomplete_error = IncompleteDownloadError("id", "name", 100, 50, Path("/tmp/test"))
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

        error_msg = str(incomplete_error).lower()
        assert "retried automatically" in error_msg


class TestValidateFileId:
    """Tests for _validate_file_id helper function."""

    def test_valid_file_id_alphanumeric(self) -> None:
        """Test validation of valid alphanumeric file ID."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("abc123DEF456") is True

    def test_valid_file_id_with_dash(self) -> None:
        """Test validation of file ID with dashes."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("abc-123-def") is True

    def test_valid_file_id_with_underscore(self) -> None:
        """Test validation of file ID with underscores."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("abc_123_def") is True

    def test_invalid_file_id_special_chars(self) -> None:
        """Test validation fails for file ID with invalid characters."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("abc@123") is False
        assert _validate_file_id("abc#123") is False
        assert _validate_file_id("abc.123") is False

    def test_invalid_file_id_empty(self) -> None:
        """Test validation fails for empty file ID."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("") is False

    def test_invalid_file_id_spaces(self) -> None:
        """Test validation fails for file ID with spaces."""
        from arandu.shared.drive import _validate_file_id

        assert _validate_file_id("abc 123") is False


class TestCheckFilePermissions:
    """Tests for _check_file_permissions helper function."""

    def test_check_permissions_nonexistent_file(self, tmp_path: Path) -> None:
        """Test checking permissions for nonexistent file."""
        from arandu.shared.drive import _check_file_permissions

        # Should not raise an exception for nonexistent file
        _check_file_permissions(tmp_path / "nonexistent.txt")

    def test_check_permissions_existing_file(self, tmp_path: Path) -> None:
        """Test checking permissions for existing file."""
        from arandu.shared.drive import _check_file_permissions

        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        # Should not raise an exception
        _check_file_permissions(file_path)


class TestDriveClientAuthentication:
    """Tests for DriveClient authentication logic."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.InstalledAppFlow")
    def test_authenticate_new_user(
        self,
        mock_flow: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication flow for new user without token."""
        from arandu.shared.drive import DriveClient

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

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_authenticate_existing_valid_token(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication with existing valid token."""
        from arandu.shared.drive import DriveClient

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
        from arandu.shared.drive import SCOPES

        assert isinstance(SCOPES, list)
        assert len(SCOPES) > 0
        assert "drive" in SCOPES[0].lower()


class TestNoAudioStreamError:
    """Tests for NoAudioStreamError exception."""

    def test_no_audio_stream_error_creation(self) -> None:
        """Test creating NoAudioStreamError."""
        from pathlib import Path

        from arandu.shared.drive import NoAudioStreamError

        error = NoAudioStreamError("file123", "test.mp4", Path("/tmp/test.mp4"))

        assert error.file_id == "file123"
        assert error.file_name == "test.mp4"
        assert "test.mp4" in str(error)
        assert "file123" in str(error)
        assert "no audio" in str(error).lower()


class TestDriveClientConfiguration:
    """Tests for DriveClient configuration handling."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_client_with_config_object(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test DriveClient initialization with TranscriberConfig object."""
        from arandu.config import TranscriberConfig
        from arandu.shared.drive import DriveClient

        config = TranscriberConfig(
            credentials="custom_creds.json",
            token="custom_token.json",
        )

        client = DriveClient(config=config)

        assert client.credentials_file == "custom_creds.json"
        assert client.token_file == "custom_token.json"

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_client_explicit_paths_override_config(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that explicit paths override config values."""
        from arandu.config import TranscriberConfig
        from arandu.shared.drive import DriveClient

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


class TestDriveClientGetFileMetadata:
    """Tests for get_file_metadata method."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_get_file_metadata_success(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test getting file metadata successfully."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        mock_service = Mock()
        mock_files = Mock()
        mock_get = Mock()
        mock_execute = Mock(return_value={"id": "file123", "name": "test.mp3"})

        mock_get.execute = mock_execute
        mock_files.get.return_value = mock_get
        mock_service.files.return_value = mock_files
        mock_build.return_value = mock_service

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        metadata = client.get_file_metadata("abc123")

        assert metadata["id"] == "file123"
        assert metadata["name"] == "test.mp3"

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_get_file_metadata_invalid_file_id(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that invalid file ID raises ValueError."""
        from arandu.shared.drive import DriveClient

        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        with pytest.raises(ValueError) as exc_info:
            client.get_file_metadata("invalid@file#id")

        assert "Invalid file_id format" in str(exc_info.value)


class TestDriveAuthenticationEdgeCases:
    """Tests for authentication edge cases."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.Request")
    def test_authenticate_refresh_token_expired(
        self,
        mock_request: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication when token is expired but can be refreshed."""
        from arandu.shared.drive import DriveClient

        # Mock expired credentials that can be refreshed
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token_value"
        mock_creds.to_json.return_value = '{"token": "refreshed"}'
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        # Trigger authentication
        _ = client.service

        # Should have called refresh
        mock_creds.refresh.assert_called_once()

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.Request")
    def test_authenticate_refresh_token_fails(
        self,
        mock_request: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication when token refresh fails."""
        from arandu.shared.drive import DriveClient

        # Mock expired credentials with failing refresh
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token_value"
        mock_creds.refresh.side_effect = Exception("Refresh failed")
        mock_credentials.from_authorized_user_file.return_value = mock_creds

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        with pytest.raises(RuntimeError) as exc_info:
            _ = client.service

        assert "Failed to refresh OAuth token" in str(exc_info.value)

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.InstalledAppFlow")
    def test_authenticate_new_user_flow_fails(
        self,
        mock_flow: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test authentication when new user flow fails (no browser)."""
        from arandu.shared.drive import DriveClient

        # Mock no existing credentials
        mock_credentials.from_authorized_user_file.side_effect = FileNotFoundError

        # Mock flow failure
        mock_flow_instance = Mock()
        mock_flow_instance.run_local_server.side_effect = Exception("No browser available")
        mock_flow.from_client_secrets_file.return_value = mock_flow_instance

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"installed": {"client_id": "test"}}')
        token_file = tmp_path / "token.json"

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        with pytest.raises(RuntimeError) as exc_info:
            _ = client.service

        assert "OAuth authentication requires a browser" in str(exc_info.value)


class TestUploadFile:
    """Tests for upload_file method."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaFileUpload")
    def test_upload_file_success(
        self,
        mock_media_upload: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test uploading a file to Google Drive."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock upload response
        mock_service.files().create().execute.return_value = {
            "id": "uploaded_file_id",
            "name": "test.json",
            "webViewLink": "https://drive.google.com/file/d/uploaded_file_id/view",
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        # Create a file to upload
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        result = client.upload_file(
            file_path=test_file,
            parent_folder_id="parent_folder_id",
            mime_type="application/json",
        )

        assert result["id"] == "uploaded_file_id"
        assert result["name"] == "test.json"
        mock_media_upload.assert_called_once()


class TestUpdateFileProperties:
    """Tests for update_file_properties method."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_update_file_properties_success(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test updating file properties."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock update response
        mock_service.files().update().execute.return_value = {
            "id": "file_id",
            "appProperties": {
                "x-transcription-status": "completed",
                "x-model-id": "test-model",
            },
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        result = client.update_file_properties(
            file_id="file_id",
            properties={
                "x-transcription-status": "completed",
                "x-model-id": "test-model",
            },
        )

        assert result["id"] == "file_id"
        assert "appProperties" in result


class TestListMediaFiles:
    """Tests for list_media_files method."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_list_media_files_without_filters(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test listing files without folder or status filters."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock list response
        mock_service.files().list().execute.return_value = {
            "files": [
                {"id": "file1", "name": "test1.mp3", "mimeType": "audio/mpeg"},
                {"id": "file2", "name": "test2.mp4", "mimeType": "video/mp4"},
            ]
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        result = client.list_media_files()

        assert len(result) == 2
        assert result[0]["id"] == "file1"
        assert result[1]["id"] == "file2"

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_list_media_files_with_folder_filter(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test listing files with folder filter."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock list response
        mock_service.files().list().execute.return_value = {
            "files": [
                {"id": "file1", "name": "test1.mp3", "mimeType": "audio/mpeg"},
            ]
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        result = client.list_media_files(folder_id="test_folder_id")

        assert len(result) == 1

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    def test_list_media_files_with_status_filter(
        self,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test listing files with status filter."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock list response
        mock_service.files().list().execute.return_value = {
            "files": [
                {"id": "file1", "name": "test1.mp3", "mimeType": "audio/mpeg"},
            ]
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        result = client.list_media_files(status_filter="completed")

        assert len(result) == 1


class TestDownloadFile:
    """Tests for download_file method."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_success(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful file download."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock get_media request
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        # Setup download mock to write file content
        destination = tmp_path / "downloaded_file.mp3"
        expected_size = 1024

        def mock_next_chunk() -> tuple[Mock, bool]:
            # Write content to the destination file
            destination.write_bytes(b"x" * expected_size)
            status = Mock()
            status.progress.return_value = 1.0
            return status, True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        result = client.download_file(
            file_id="abc123",
            destination=destination,
            expected_size=expected_size,
            file_name="test.mp3",
        )

        assert result == destination
        assert destination.exists()

    @patch("tenacity.nap.time.sleep", return_value=None)  # Skip retry delays
    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_empty_raises_error(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that empty download raises EmptyDownloadError."""
        from arandu.shared.drive import DriveClient, EmptyDownloadError

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"

        def mock_next_chunk() -> tuple[Mock, bool]:
            # Create empty file
            destination.touch()
            return Mock(), True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        with pytest.raises(EmptyDownloadError):
            client.download_file(
                file_id="abc123",
                destination=destination,
                expected_size=1024,
                file_name="test.mp3",
            )

    @patch("tenacity.nap.time.sleep", return_value=None)  # Skip retry delays
    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_incomplete_raises_error(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that incomplete download raises IncompleteDownloadError."""
        from arandu.shared.drive import DriveClient, IncompleteDownloadError

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"
        expected_size = 10000
        actual_size = 5000  # Only half downloaded

        def mock_next_chunk() -> tuple[Mock, bool]:
            # Create partially downloaded file
            destination.write_bytes(b"x" * actual_size)
            return Mock(), True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        with pytest.raises(IncompleteDownloadError):
            client.download_file(
                file_id="abc123",
                destination=destination,
                expected_size=expected_size,
                file_name="test.mp3",
            )

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_within_tolerance(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that download within tolerance (0.1%) succeeds."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"
        expected_size = 10000
        # Slightly different size (within 0.1% tolerance)
        actual_size = expected_size + 5

        def mock_next_chunk() -> tuple[Mock, bool]:
            destination.write_bytes(b"x" * actual_size)
            return Mock(), True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        # Should not raise
        result = client.download_file(
            file_id="abc123",
            destination=destination,
            expected_size=expected_size,
            file_name="test.mp3",
        )

        assert result == destination

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_fetches_metadata_if_no_size(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that download fetches metadata if expected_size not provided."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        # Mock get_file_metadata call
        mock_files_get = Mock()
        mock_files_get.execute.return_value = {
            "id": "abc123",
            "name": "test.mp3",
            "size": "1024",
        }
        mock_service.files().get.return_value = mock_files_get

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"

        def mock_next_chunk() -> tuple[Mock, bool]:
            destination.write_bytes(b"x" * 1024)
            return Mock(), True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        # Download without expected_size
        result = client.download_file(
            file_id="abc123",
            destination=destination,
        )

        assert result == destination
        # Should have called get to fetch metadata
        mock_service.files().get.assert_called()

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_with_progress(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test download with progress bar updates."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"
        expected_size = 1024

        # Track progress calls
        progress_mock = Mock()
        task_id = 1

        call_count = [0]

        def mock_next_chunk() -> tuple[Mock, bool]:
            call_count[0] += 1
            if call_count[0] == 1:
                status = Mock()
                status.progress.return_value = 0.5
                return status, False
            else:
                destination.write_bytes(b"x" * expected_size)
                status = Mock()
                status.progress.return_value = 1.0
                return status, True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        result = client.download_file(
            file_id="abc123",
            destination=destination,
            expected_size=expected_size,
            file_name="test.mp3",
            progress=progress_mock,
            task_id=task_id,
        )

        assert result == destination
        # Progress should have been updated
        assert progress_mock.update.call_count >= 1

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    @patch("arandu.shared.drive.io.FileIO")
    def test_download_file_os_error_cleans_up(
        self,
        mock_file_io: MagicMock,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that OS errors are raised and cleanup occurs."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"

        # Mock FileIO to raise an error
        mock_file_io.side_effect = OSError("Disk full")

        with pytest.raises(OSError) as exc_info:
            client.download_file(
                file_id="abc123",
                destination=destination,
                expected_size=1024,
                file_name="test.mp3",
            )

        assert "Disk full" in str(exc_info.value)


class TestUploadFileWithProgress:
    """Tests for upload_file with progress updates."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaFileUpload")
    def test_upload_file_with_progress(
        self,
        mock_media_upload: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test uploading a file with progress tracking."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock upload response
        mock_service.files().create().execute.return_value = {
            "id": "uploaded_file_id",
            "name": "test.json",
            "webViewLink": "https://drive.google.com/file/d/uploaded_file_id/view",
        }

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        # Create a file to upload
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        progress_mock = Mock()
        task_id = 1

        result = client.upload_file(
            file_path=test_file,
            parent_folder_id="parent_folder_id",
            mime_type="application/json",
            progress=progress_mock,
            task_id=task_id,
        )

        assert result["id"] == "uploaded_file_id"
        # Progress should have been updated to 100
        progress_mock.update.assert_called_with(task_id, completed=100)


class TestDownloadFileMetadataFetch:
    """Tests for metadata fetch behavior in download_file."""

    @patch("arandu.shared.drive.build")
    @patch("arandu.shared.drive.Credentials")
    @patch("arandu.shared.drive.MediaIoBaseDownload")
    def test_download_file_metadata_fetch_fails_gracefully(
        self,
        mock_download: MagicMock,
        mock_credentials: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that download continues when metadata fetch fails."""
        from arandu.shared.drive import DriveClient

        # Setup mocks
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials.from_authorized_user_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_service.files().get_media.return_value = Mock()

        # Mock get_file_metadata to fail
        mock_files_get = Mock()
        mock_files_get.execute.side_effect = Exception("API error")
        mock_service.files().get.return_value = mock_files_get

        credentials_file = tmp_path / "creds.json"
        credentials_file.write_text('{"test": "data"}')
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "test"}')

        client = DriveClient(
            credentials_file=str(credentials_file),
            token_file=str(token_file),
        )

        destination = tmp_path / "downloaded_file.mp3"

        def mock_next_chunk() -> tuple[Mock, bool]:
            # Create file with some content
            destination.write_bytes(b"x" * 1024)
            return Mock(), True

        mock_downloader = Mock()
        mock_downloader.next_chunk.side_effect = mock_next_chunk
        mock_download.return_value = mock_downloader

        # Download without expected_size - should try to fetch metadata and continue
        result = client.download_file(
            file_id="abc123",
            destination=destination,
            # expected_size not provided - triggers metadata fetch attempt
        )

        # Should succeed despite metadata fetch failure
        assert result == destination
        assert destination.exists()
