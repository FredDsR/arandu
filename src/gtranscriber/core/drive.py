"""Google Drive integration for G-Transcriber.

Implements resilient download and upload operations with support for
resumable media transfers and retry logic.
"""

from __future__ import annotations

import io
import logging
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from gtranscriber.config import TranscriberConfig


class DownloadError(Exception):
    """Base exception for download-related errors."""

    pass


class IncompleteDownloadError(DownloadError):
    """Raised when a download completes but file size doesn't match expected."""

    def __init__(
        self,
        file_id: str,
        file_name: str,
        expected_size: int,
        actual_size: int,
        destination: Path,
    ) -> None:
        self.file_id = file_id
        self.file_name = file_name
        self.expected_size = expected_size
        self.actual_size = actual_size
        self.destination = destination

        # Calculate percentage downloaded
        percentage = (actual_size / expected_size * 100) if expected_size > 0 else 0

        # Format sizes for readability
        expected_mb = expected_size / (1024 * 1024)
        actual_mb = actual_size / (1024 * 1024)

        super().__init__(
            f"Incomplete download for '{file_name}' (ID: {file_id}): "
            f"expected {expected_mb:.2f} MB but got {actual_mb:.2f} MB "
            f"({percentage:.1f}% complete). "
            f"This may be due to network issues or Google Drive rate limiting. "
            f"The file will be retried automatically."
        )


class EmptyDownloadError(DownloadError):
    """Raised when a download results in an empty file."""

    def __init__(self, file_id: str, file_name: str, destination: Path) -> None:
        self.file_id = file_id
        self.file_name = file_name
        self.destination = destination

        super().__init__(
            f"Download resulted in empty file for '{file_name}' (ID: {file_id}). "
            f"This typically indicates a Google Drive API error or permission issue. "
            f"Verify that the file exists and is accessible with the current credentials."
        )


class NoAudioStreamError(DownloadError):
    """Raised when a media file has no audio stream to transcribe."""

    def __init__(self, file_id: str, file_name: str, destination: Path) -> None:
        self.file_id = file_id
        self.file_name = file_name
        self.destination = destination

        super().__init__(
            f"No audio stream found in '{file_name}' (ID: {file_id}). "
            f"The file may be a video without audio, or the audio codec is not supported. "
            f"Use ffprobe to inspect the file: ffprobe -v error -show_streams <file>"
        )


if TYPE_CHECKING:
    from googleapiclient.discovery import Resource
    from rich.progress import Progress, TaskID


# Google Drive API scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

logger = logging.getLogger(__name__)


def _validate_file_id(file_id: str) -> bool:
    """Validate that file_id matches expected format.

    Args:
        file_id: Google Drive file ID to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Google Drive file IDs are typically alphanumeric with dashes and underscores
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", file_id))


def _check_file_permissions(file_path: Path) -> None:
    """Check and warn about overly permissive file permissions.

    Args:
        file_path: Path to the file to check.
    """
    if not file_path.exists():
        return

    # Only check on Unix-like systems
    if hasattr(os, "stat") and hasattr(os.stat(str(file_path)), "st_mode"):
        mode = os.stat(str(file_path)).st_mode
        # Check if group/other write permissions are set
        if mode & 0o022:  # Check if group/other write permissions are set
            logger.warning(
                f"File {file_path} has write permissions for group or others. "
                f"Consider restricting to owner-only access (chmod 600)."
            )


class DriveClient:
    """Client for Google Drive operations with resilient transfers."""

    def __init__(
        self,
        credentials_file: str | None = None,
        token_file: str | None = None,
        config: TranscriberConfig | None = None,
    ) -> None:
        """Initialize the Drive client.

        Args:
            credentials_file: Path to OAuth2 credentials file. If not provided,
                            will be loaded from config or environment.
            token_file: Path to store/load the token. If not provided,
                       will be loaded from config or environment.
            config: Optional TranscriberConfig instance. If not provided,
                   will be loaded from environment.
        """
        # Load config if not provided
        if config is None:
            from gtranscriber.config import TranscriberConfig

            config = TranscriberConfig()

        self.credentials_file = credentials_file or config.credentials
        self.token_file = token_file or config.token
        self._service: Resource | None = None

    @property
    def service(self) -> Resource:
        """Get the Drive API service, authenticating if needed."""
        if self._service is None:
            self._service = self._authenticate()
        return self._service

    def _authenticate(self) -> Resource:
        """Authenticate with Google Drive API.

        Returns:
            Drive API service resource.
        """
        creds = None
        token_path = Path(self.token_file)

        # Check permissions on sensitive files
        _check_file_permissions(token_path)
        _check_file_permissions(Path(self.credentials_file))

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to refresh OAuth token: {e}. "
                        f"Please run 'gtranscriber refresh-token' on a machine with a browser, "
                        f"then copy the updated token.json to this environment."
                    ) from e
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                try:
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    raise RuntimeError(
                        f"OAuth authentication requires a browser but none is available: {e}. "
                        f"Please run 'gtranscriber refresh-token' on a machine with a browser, "
                        f"then copy the updated token.json to this environment."
                    ) from e

            # Save the credentials for next run
            token_path.write_text(creds.to_json())
            # Set restrictive permissions on the token file
            if hasattr(os, "chmod"):
                with suppress(OSError):
                    os.chmod(str(token_path), 0o600)

        return build("drive", "v3", credentials=creds)

    def get_file_metadata(self, file_id: str) -> dict:
        """Get metadata for a file.

        Args:
            file_id: Google Drive file ID.

        Returns:
            File metadata dictionary.

        Raises:
            ValueError: If file_id format is invalid.
        """
        if not _validate_file_id(file_id):
            raise ValueError(f"Invalid file_id format: {file_id}")

        return (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType,parents,webContentLink,size,appProperties",
            )
            .execute()
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((IncompleteDownloadError, EmptyDownloadError)),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Download incomplete, retrying in {retry_state.next_action.sleep} seconds "
            f"(attempt {retry_state.attempt_number}/5)..."
        ),
    )
    def download_file(
        self,
        file_id: str,
        destination: str | Path,
        expected_size: int | None = None,
        file_name: str | None = None,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> Path:
        """Download a file from Google Drive with resumable support and validation.

        Args:
            file_id: Google Drive file ID.
            destination: Local path to save the file.
            expected_size: Expected file size in bytes for validation.
            file_name: Original file name (for error messages).
            progress: Optional Rich progress bar.
            task_id: Optional task ID for progress updates.

        Returns:
            Path to the downloaded file.

        Raises:
            EmptyDownloadError: If the downloaded file is empty.
            IncompleteDownloadError: If the downloaded file size doesn't match expected.
            OSError: If file system operations fail.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        display_name = file_name or file_id

        # Fetch expected size from API if not provided
        if expected_size is None:
            try:
                metadata = self.get_file_metadata(file_id)
                expected_size = int(metadata.get("size", 0))
                if file_name is None:
                    display_name = metadata.get("name", file_id)
            except Exception as e:
                logger.warning(f"Could not fetch file metadata for size validation: {e}")

        request = self.service.files().get_media(fileId=file_id)

        try:
            with io.FileIO(str(destination), "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if progress and task_id is not None and status:
                        progress.update(task_id, completed=round(status.progress() * 100))
        except OSError as e:
            with suppress(Exception):
                destination.unlink()
            logger.error(f"Failed to download file to {destination}: {e}")
            raise

        # Validate downloaded file
        actual_size = destination.stat().st_size

        if actual_size == 0:
            with suppress(Exception):
                destination.unlink()
            raise EmptyDownloadError(file_id, display_name, destination)

        if expected_size and actual_size != expected_size:
            # Allow small tolerance (0.1%) for potential metadata/encoding differences
            tolerance = expected_size * 0.001
            if abs(actual_size - expected_size) > tolerance:
                with suppress(Exception):
                    destination.unlink()
                raise IncompleteDownloadError(
                    file_id, display_name, expected_size, actual_size, destination
                )

        logger.debug(f"Download validated: {display_name} ({actual_size / (1024 * 1024):.2f} MB)")
        return destination

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def upload_file(
        self,
        file_path: str | Path,
        parent_folder_id: str,
        mime_type: str = "application/json",
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> dict:
        """Upload a file to Google Drive.

        Args:
            file_path: Path to the local file.
            parent_folder_id: ID of the parent folder in Drive.
            mime_type: MIME type of the file.
            progress: Optional Rich progress bar.
            task_id: Optional task ID for progress updates.

        Returns:
            Uploaded file metadata.
        """
        file_path = Path(file_path)
        file_metadata = {
            "name": file_path.name,
            "parents": [parent_folder_id],
        }

        media = MediaFileUpload(
            str(file_path),
            mimetype=mime_type,
            resumable=True,
        )

        file = (
            self.service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id,name,webViewLink",
            )
            .execute()
        )

        if progress and task_id is not None:
            progress.update(task_id, completed=100)

        return file

    def update_file_properties(
        self,
        file_id: str,
        properties: dict,
    ) -> dict:
        """Update appProperties on a file for state persistence.

        Args:
            file_id: Google Drive file ID.
            properties: Properties to set (e.g., x-transcription-status).

        Returns:
            Updated file metadata.
        """
        return (
            self.service.files()
            .update(
                fileId=file_id,
                body={"appProperties": properties},
            )
            .execute()
        )

    def list_media_files(
        self,
        folder_id: str | None = None,
        status_filter: str | None = None,
    ) -> list[dict]:
        """List audio/video files from Drive.

        Args:
            folder_id: Optional folder ID to search in.
            status_filter: Optional transcription status filter.

        Returns:
            List of file metadata dictionaries.
        """
        query_parts = [
            "(mimeType contains 'audio/' or mimeType contains 'video/')",
            "trashed = false",
        ]

        if folder_id:
            # Escape single quotes in the folder_id to prevent query syntax issues
            safe_folder_id = folder_id.replace("'", "\\'")
            query_parts.append(f"'{safe_folder_id}' in parents")

        if status_filter:
            # Escape single quotes in the filter value to prevent query syntax issues
            safe_status_filter = status_filter.replace("'", "\\'")
            query_parts.append(
                f"appProperties has {{ key='x-transcription-status' "
                f"and value='{safe_status_filter}' }}"
            )

        query = " and ".join(query_parts)

        results = (
            self.service.files()
            .list(
                q=query,
                fields="files(id,name,mimeType,parents,webContentLink,size,appProperties)",
                pageSize=100,
            )
            .execute()
        )

        return results.get("files", [])
