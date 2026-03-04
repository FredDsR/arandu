"""Tests for batch KG construction orchestrator."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime for tmp_path fixtures
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from arandu.config import KGConfig
from arandu.core.kg.schemas import KGConstructionResult
from arandu.schemas import KGMetadata

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _write_transcription(
    directory: Path,
    file_id: str = "test123",
    text: str = "Test transcription text.",
    is_valid: bool = True,
) -> Path:
    """Write a minimal transcription JSON file."""
    data = {
        "file_id": file_id,
        "name": f"{file_id}.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 10.0,
        "transcription_status": "completed",
        "is_valid": is_valid,
    }
    filepath = directory / f"{file_id}_transcription.json"
    filepath.write_text(json.dumps(data))
    return filepath


class TestLoadTranscriptionRecords:
    """Tests for _load_transcription_records."""

    def test_loads_valid_records(self, tmp_path: Path) -> None:
        """Test loading valid transcription records."""
        from arandu.core.kg.batch import _load_transcription_records

        _write_transcription(tmp_path, "id1")
        _write_transcription(tmp_path, "id2")

        records = _load_transcription_records(tmp_path)
        assert len(records) == 2
        ids = {r.file_id for r in records}
        assert ids == {"id1", "id2"}

    def test_skips_invalid_records(self, tmp_path: Path) -> None:
        """Test that is_valid=False records are skipped."""
        from arandu.core.kg.batch import _load_transcription_records

        _write_transcription(tmp_path, "valid", is_valid=True)
        _write_transcription(tmp_path, "invalid", is_valid=False)

        records = _load_transcription_records(tmp_path)
        assert len(records) == 1
        assert records[0].file_id == "valid"

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test loading from empty directory returns empty list."""
        from arandu.core.kg.batch import _load_transcription_records

        records = _load_transcription_records(tmp_path)
        assert records == []


class TestResolveTranscriptionDir:
    """Tests for _resolve_transcription_dir."""

    def test_returns_dir_with_transcription_files(self, tmp_path: Path) -> None:
        """Test fast path: dir already contains transcription files."""
        from arandu.core.kg.batch import _resolve_transcription_dir

        _write_transcription(tmp_path, "test")
        result = _resolve_transcription_dir(tmp_path)
        assert result == tmp_path

    def test_returns_input_dir_when_empty(self, tmp_path: Path) -> None:
        """Test fallback: returns input dir when no files found."""
        from arandu.core.kg.batch import _resolve_transcription_dir

        result = _resolve_transcription_dir(tmp_path)
        assert result == tmp_path


class TestRunBatchKGConstruction:
    """Tests for run_batch_kg_construction orchestrator."""

    def test_empty_input_completes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test orchestrator completes gracefully with no input records."""
        mocker.patch(
            "arandu.core.kg.batch.ResultsConfig",
            return_value=MagicMock(enable_versioning=False),
        )

        from arandu.core.kg.batch import run_batch_kg_construction

        config = KGConfig()
        output_dir = tmp_path / "output"

        # Should not raise
        run_batch_kg_construction(tmp_path, output_dir, config)

    def test_processes_records(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test orchestrator dispatches to constructor and tracks results."""
        mocker.patch(
            "arandu.core.kg.batch.ResultsConfig",
            return_value=MagicMock(enable_versioning=False),
        )

        # Write transcription files
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_transcription(input_dir, "id1", "Texto um.")
        _write_transcription(input_dir, "id2", "Texto dois.")

        output_dir = tmp_path / "output"

        # Mock the constructor
        mock_result = KGConstructionResult(
            graph_file=output_dir / "graph.graphml",
            metadata=KGMetadata(
                graph_id="test",
                source_documents=["id1", "id2"],
                model_id="test-model",
                provider="ollama",
            ),
            node_count=10,
            edge_count=5,
            source_record_ids=["id1", "id2"],
        )

        mock_constructor = MagicMock()
        mock_constructor.build_graph.return_value = mock_result
        mocker.patch(
            "arandu.core.kg.batch.create_kg_constructor",
            return_value=mock_constructor,
        )

        from arandu.core.kg.batch import run_batch_kg_construction

        config = KGConfig()
        run_batch_kg_construction(input_dir, output_dir, config)

        mock_constructor.build_graph.assert_called_once()
        call_args = mock_constructor.build_graph.call_args
        assert len(call_args[0][0]) == 2  # 2 records

    def test_constructor_failure_marks_failed(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test orchestrator handles constructor exceptions gracefully."""
        mocker.patch(
            "arandu.core.kg.batch.ResultsConfig",
            return_value=MagicMock(enable_versioning=False),
        )

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_transcription(input_dir, "id1")

        mock_constructor = MagicMock()
        mock_constructor.build_graph.side_effect = RuntimeError("LLM error")
        mocker.patch(
            "arandu.core.kg.batch.create_kg_constructor",
            return_value=mock_constructor,
        )

        from arandu.core.kg.batch import run_batch_kg_construction

        config = KGConfig()
        output_dir = tmp_path / "output"

        with pytest.raises(RuntimeError, match="LLM error"):
            run_batch_kg_construction(input_dir, output_dir, config)
