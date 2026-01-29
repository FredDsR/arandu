"""Shared pytest fixtures for G-Transcriber tests.

This module provides common fixtures used across all test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
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
