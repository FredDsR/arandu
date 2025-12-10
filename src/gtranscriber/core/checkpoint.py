"""Checkpoint management for batch transcription.

Provides a simple checkpoint system to track progress and enable resumption
of interrupted batch transcription jobs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointState(BaseModel):
    """State tracking for batch transcription checkpoint."""

    completed_files: set[str] = Field(
        default_factory=set, description="Set of completed file IDs"
    )
    failed_files: dict[str, str] = Field(
        default_factory=dict, description="Map of file ID to error message"
    )
    total_files: int = Field(0, description="Total number of files to process")
    started_at: datetime = Field(default_factory=datetime.now, description="Start timestamp")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class CheckpointManager:
    """Manages checkpoint state for batch transcription."""

    def __init__(self, checkpoint_file: Path) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to the checkpoint JSON file.
        """
        self.checkpoint_file = checkpoint_file
        self.state = self._load()

    def _load(self) -> CheckpointState:
        """Load checkpoint state from file.

        Returns:
            CheckpointState object, or a new one if file doesn't exist.
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found, starting fresh")
            return CheckpointState()

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

                # Convert completed_files from list to set if needed
                if "completed_files" in data and isinstance(data["completed_files"], list):
                    data["completed_files"] = set(data["completed_files"])

                return CheckpointState(**data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}, starting fresh")
            return CheckpointState()

    def save(self) -> None:
        """Save checkpoint state to file."""
        try:
            self.state.last_updated = datetime.now()

            # Convert set to list for JSON serialization
            data = self.state.model_dump()
            data["completed_files"] = list(data["completed_files"])
            data["started_at"] = self.state.started_at.isoformat()
            data["last_updated"] = self.state.last_updated.isoformat()

            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Checkpoint saved: {len(self.state.completed_files)} files completed")
        except Exception as e:
            logger.exception(f"Failed to save checkpoint: {e}")

    def mark_completed(self, file_id: str) -> None:
        """Mark a file as completed.

        Args:
            file_id: Google Drive file ID.
        """
        self.state.completed_files.add(file_id)
        # Remove from failed if it was there
        self.state.failed_files.pop(file_id, None)
        self.save()

    def mark_failed(self, file_id: str, error: str) -> None:
        """Mark a file as failed.

        Args:
            file_id: Google Drive file ID.
            error: Error message.
        """
        self.state.failed_files[file_id] = error
        self.save()

    def is_completed(self, file_id: str) -> bool:
        """Check if a file has been completed.

        Args:
            file_id: Google Drive file ID.

        Returns:
            True if file is already completed.
        """
        return file_id in self.state.completed_files

    def set_total_files(self, total: int) -> None:
        """Set the total number of files to process.

        Args:
            total: Total number of files.
        """
        self.state.total_files = total
        self.save()

    def get_progress(self) -> tuple[int, int]:
        """Get progress information.

        Returns:
            Tuple of (completed, total) files.
        """
        return len(self.state.completed_files), self.state.total_files

    def get_remaining_count(self) -> int:
        """Get count of remaining files to process.

        Returns:
            Number of files remaining.
        """
        return self.state.total_files - len(self.state.completed_files)
