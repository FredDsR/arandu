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
from tenacity import retry, stop_after_attempt, wait_exponential

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
        credentials_file: str = "credentials.json",
        token_file: str = "token.json",
    ) -> None:
        """Initialize the Drive client.

        Args:
            credentials_file: Path to OAuth2 credentials file.
            token_file: Path to store/load the token.
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
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
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)

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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def download_file(
        self,
        file_id: str,
        destination: str | Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> Path:
        """Download a file from Google Drive with resumable support.

        Args:
            file_id: Google Drive file ID.
            destination: Local path to save the file.
            progress: Optional Rich progress bar.
            task_id: Optional task ID for progress updates.

        Returns:
            Path to the downloaded file.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        request = self.service.files().get_media(fileId=file_id)

        try:
            with io.FileIO(str(destination), "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if progress and task_id is not None and status:
                        # Update progress based on actual progress (0-100)
                        progress.update(task_id, completed=round(status.progress() * 100))
        except OSError as e:
            # Clean up partial file on failure
            with suppress(Exception):
                destination.unlink()
            logger.error(f"Failed to download file to {destination}: {e}")
            raise

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
