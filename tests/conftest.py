"""Shared pytest fixtures for Arandu tests.

This module provides common fixtures used across all test modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

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


#: Default transcription text for the shared record builders.
#:
#: Long and sentence-shaped on purpose: the chunkers need a substantial body
#: before they emit more than one chunk, so tests that do not care about the
#: text still get one that exercises real chunk boundaries.
DEFAULT_TRANSCRIPTION_TEXT = (
    "O pescador contou que quando o rio sobe ele guarda o barco no barranco alto. "
    "Depois falou da prefeitura, do ciclone e da ajuda que veio da universidade. "
) * 20


class TranscriptionRecordPayloadBuilder(Protocol):
    """Callable that builds a minimal-but-valid ``EnrichedRecord`` payload."""

    def __call__(
        self,
        file_id: str = ...,
        text: str = ...,
        *,
        is_valid: bool | None = ...,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the payload dictionary."""
        ...


class TranscriptionRecordWriter(Protocol):
    """Callable that writes an ``EnrichedRecord`` JSON into a directory."""

    def __call__(
        self,
        directory: Path,
        file_id: str = ...,
        text: str = ...,
        *,
        is_valid: bool | None = ...,
        suffix: str = ...,
        **overrides: Any,
    ) -> Path:
        """Write the record and return its path."""
        ...


@pytest.fixture
def transcription_record_payload() -> TranscriptionRecordPayloadBuilder:
    """Build minimal-but-valid transcription-stage record payloads.

    The single source of truth for what an ``EnrichedRecord`` needs on disk.
    A new required field on the schema is fixed here, not in every test module
    that happens to need a transcription artifact.

    Keys use the canonical field names (``file_id``, ``web_content_link``);
    the schema's ``populate_by_name`` config accepts those alongside the
    Google Drive aliases.

    Returns:
        A callable ``(file_id, text, *, is_valid, **overrides) -> dict``.
        ``is_valid`` drives the ``validation`` payload the transcription judge
        writes: ``True``/``False`` stamp a passing/rejected verdict, ``None``
        (the default) leaves the record unjudged. Any other field is
        overridden by keyword.

    Examples:
        >>> payload = transcription_record_payload("src_a", "Texto.", is_valid=False)
        >>> payload["validation"]["passed"]
        False
    """

    def _build(
        file_id: str = "test123",
        text: str = DEFAULT_TRANSCRIPTION_TEXT,
        *,
        is_valid: bool | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file_id": file_id,
            "name": f"{file_id}.mp3",
            "mimeType": "audio/mpeg",
            "parents": ["folder"],
            "web_content_link": "https://drive.google.com/test",
            "size_bytes": 1024,
            "duration_milliseconds": 60000,
            "transcription_text": text,
            "detected_language": "pt",
            "language_probability": 0.95,
            "model_id": "openai/whisper-large-v3",
            "compute_device": "cpu",
            "processing_duration_sec": 10.0,
            "transcription_status": "completed",
        }
        if is_valid is not None:
            payload["validation"] = {
                "stage_results": {},
                "passed": is_valid,
                "rejected_at": None if is_valid else "heuristic_filter",
            }
        payload.update(overrides)
        return payload

    return _build


@pytest.fixture
def write_transcription_record(
    transcription_record_payload: TranscriptionRecordPayloadBuilder,
) -> TranscriptionRecordWriter:
    """Write transcription-stage record JSONs the way the stage does.

    The payload is dumped verbatim rather than round-tripped through the
    model, so tests can stage pre-normalization text (a leading space, say)
    that the schema validator would otherwise strip on construction.

    Args:
        transcription_record_payload: Builder for the payload dictionary.

    Returns:
        A callable ``(directory, file_id, text, *, is_valid, suffix,
        **overrides) -> Path``. ``directory`` is created if missing.
        ``suffix`` defaults to ``"_transcription"``, the stage's real filename
        convention; pass ``""`` for the legacy bare ``<file_id>.json`` form.
    """

    def _write(
        directory: Path,
        file_id: str = "test123",
        text: str = DEFAULT_TRANSCRIPTION_TEXT,
        *,
        is_valid: bool | None = None,
        suffix: str = "_transcription",
        **overrides: Any,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = transcription_record_payload(file_id, text, is_valid=is_valid, **overrides)
        path = directory / f"{file_id}{suffix}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write
