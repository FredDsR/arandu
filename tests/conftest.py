"""Shared pytest fixtures for Arandu tests.

This module provides common fixtures used across all test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture.

    Returns:
        Path object pointing to the temporary directory.
    """
    return tmp_path


@pytest.fixture
def mock_torch_cuda(mocker: MockerFixture) -> MagicMock:
    """Mock torch.cuda module for hardware detection tests.

    Args:
        mocker: Pytest-mock fixture for creating mocks.

    Returns:
        MagicMock object for torch.cuda.
    """
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = False
    return mock_cuda


@pytest.fixture
def mock_openai_client(mocker: MockerFixture) -> MagicMock:
    """Mock OpenAI client for LLM tests.

    Args:
        mocker: Pytest-mock fixture for creating mocks.

    Returns:
        MagicMock object for OpenAI client.
    """
    return mocker.patch("openai.OpenAI")


@pytest.fixture
def mock_subprocess_run(mocker: MockerFixture) -> MagicMock:
    """Mock subprocess.run for media processing tests.

    Args:
        mocker: Pytest-mock fixture for creating mocks.

    Returns:
        MagicMock object for subprocess.run.
    """
    return mocker.patch("subprocess.run")


@pytest.fixture
def sample_enriched_record_data() -> dict:
    """Provide sample EnrichedRecord data for tests.

    Returns:
        Dictionary with all required EnrichedRecord fields.
    """
    return {
        "gdrive_id": "test123",
        "name": "test.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["parent_folder_id"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024000,
        "duration_milliseconds": 60000,
        "transcription_text": "This is a test transcription. " * 20,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "openai/whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 30.5,
        "transcription_status": "completed",
    }
