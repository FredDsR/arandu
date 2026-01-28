"""Tests for checkpoint management."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gtranscriber.core.checkpoint import CheckpointManager, CheckpointState


class TestCheckpointState:
    """Tests for CheckpointState model."""

    def test_default_initialization(self) -> None:
        """Test default CheckpointState initialization."""
        state = CheckpointState()

        assert state.completed_files == set()
        assert state.failed_files == {}
        assert state.total_files == 0
        assert isinstance(state.started_at, datetime)
        assert isinstance(state.last_updated, datetime)

    def test_custom_initialization(self) -> None:
        """Test CheckpointState with custom values."""
        state = CheckpointState(
            completed_files={"file1", "file2"},
            failed_files={"file3": "Error message"},
            total_files=5,
        )

        assert state.completed_files == {"file1", "file2"}
        assert state.failed_files == {"file3": "Error message"}
        assert state.total_files == 5


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_initialization_no_checkpoint(self, tmp_path: Path) -> None:
        """Test initialization when no checkpoint file exists."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)

        assert manager.checkpoint_file == checkpoint_file
        assert isinstance(manager.state, CheckpointState)
        assert len(manager.state.completed_files) == 0

    def test_initialization_with_existing_checkpoint(self, tmp_path: Path) -> None:
        """Test initialization when checkpoint file exists."""
        checkpoint_file = tmp_path / "checkpoint.json"

        # Create a checkpoint file
        data = {
            "completed_files": ["file1", "file2"],
            "failed_files": {"file3": "Test error"},
            "total_files": 5,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        checkpoint_file.write_text(json.dumps(data))

        manager = CheckpointManager(checkpoint_file)

        assert manager.state.completed_files == {"file1", "file2"}
        assert manager.state.failed_files == {"file3": "Test error"}
        assert manager.state.total_files == 5

    def test_initialization_with_corrupted_checkpoint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test initialization when checkpoint file is corrupted."""
        checkpoint_file = tmp_path / "checkpoint.json"

        # Create an invalid JSON file
        checkpoint_file.write_text("invalid json content")

        manager = CheckpointManager(checkpoint_file)

        # Should create a fresh state
        assert len(manager.state.completed_files) == 0
        assert "corrupted" in caplog.text.lower() or "invalid" in caplog.text.lower()

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that save() creates parent directories if they don't exist."""
        checkpoint_file = tmp_path / "subdir" / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.save()

        assert checkpoint_file.parent.exists()
        assert checkpoint_file.exists()

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """Test saving and loading checkpoint state."""
        checkpoint_file = tmp_path / "checkpoint.json"

        # Create manager and add some data
        manager1 = CheckpointManager(checkpoint_file)
        manager1.state.completed_files.add("file1")
        manager1.state.completed_files.add("file2")
        manager1.state.failed_files["file3"] = "Error message"
        manager1.state.total_files = 10
        manager1.save()

        # Load in a new manager
        manager2 = CheckpointManager(checkpoint_file)

        assert manager2.state.completed_files == {"file1", "file2"}
        assert manager2.state.failed_files == {"file3": "Error message"}
        assert manager2.state.total_files == 10

    def test_mark_completed(self, tmp_path: Path) -> None:
        """Test marking a file as completed."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.mark_completed("file1")

        assert "file1" in manager.state.completed_files

    def test_mark_completed_removes_from_failed(self, tmp_path: Path) -> None:
        """Test that marking as completed removes from failed list."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.mark_failed("file1", "Some error")
        assert "file1" in manager.state.failed_files

        manager.mark_completed("file1")

        assert "file1" in manager.state.completed_files
        assert "file1" not in manager.state.failed_files

    def test_mark_failed(self, tmp_path: Path) -> None:
        """Test marking a file as failed."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.mark_failed("file1", "Test error message")

        assert manager.state.failed_files["file1"] == "Test error message"

    def test_is_completed(self, tmp_path: Path) -> None:
        """Test checking if a file is completed."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)

        assert manager.is_completed("file1") is False

        manager.mark_completed("file1")

        assert manager.is_completed("file1") is True

    def test_set_total_files(self, tmp_path: Path) -> None:
        """Test setting total files count."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.set_total_files(100)

        assert manager.state.total_files == 100

        # Verify it persists
        manager2 = CheckpointManager(checkpoint_file)
        assert manager2.state.total_files == 100

    def test_get_progress(self, tmp_path: Path) -> None:
        """Test getting progress information."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.set_total_files(10)
        manager.mark_completed("file1")
        manager.mark_completed("file2")

        completed, total = manager.get_progress()

        assert completed == 2
        assert total == 10

    def test_get_remaining_count(self, tmp_path: Path) -> None:
        """Test getting remaining files count."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.set_total_files(10)
        manager.mark_completed("file1")
        manager.mark_completed("file2")

        remaining = manager.get_remaining_count()

        assert remaining == 8

    def test_checkpoint_file_format(self, tmp_path: Path) -> None:
        """Test that checkpoint file is saved in correct format."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.state.completed_files.add("file1")
        manager.state.failed_files["file2"] = "Error"
        manager.state.total_files = 5
        manager.save()

        # Read the raw JSON
        with open(checkpoint_file) as f:
            data = json.load(f)

        assert "completed_files" in data
        assert isinstance(data["completed_files"], list)
        assert "file1" in data["completed_files"]
        assert "failed_files" in data
        assert data["failed_files"]["file2"] == "Error"
        assert data["total_files"] == 5
        assert "started_at" in data
        assert "last_updated" in data

    def test_last_updated_changes_on_save(self, tmp_path: Path) -> None:
        """Test that last_updated timestamp is updated on save."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        original_timestamp = manager.state.last_updated

        # Wait a tiny bit and save
        import time

        time.sleep(0.01)
        manager.save()

        assert manager.state.last_updated > original_timestamp

    def test_save_failure_handling(self, tmp_path: Path, mocker: pytest.fixture) -> None:
        """Test that save() handles errors gracefully."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)

        # Mock open to raise an exception
        mocker.patch("builtins.open", side_effect=PermissionError("No permission"))

        # Should not raise an exception
        manager.save()

    def test_completed_files_set_to_list_conversion(self, tmp_path: Path) -> None:
        """Test that completed_files set is properly converted to list for JSON."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.mark_completed("file1")
        manager.mark_completed("file2")
        manager.mark_completed("file3")

        # Read the JSON file directly
        with open(checkpoint_file) as f:
            data = json.load(f)

        # Should be a list in JSON
        assert isinstance(data["completed_files"], list)
        assert len(data["completed_files"]) == 3
        assert set(data["completed_files"]) == {"file1", "file2", "file3"}

    def test_multiple_marks_same_file(self, tmp_path: Path) -> None:
        """Test that marking the same file multiple times works correctly."""
        checkpoint_file = tmp_path / "checkpoint.json"

        manager = CheckpointManager(checkpoint_file)
        manager.mark_completed("file1")
        manager.mark_completed("file1")
        manager.mark_completed("file1")

        # Should only be in the set once
        assert len(manager.state.completed_files) == 1
        assert "file1" in manager.state.completed_files
