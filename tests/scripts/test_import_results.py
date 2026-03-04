"""Tests for the import_results script."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.exceptions import Exit as ClickExit

from arandu.schemas import RunStatus
from scripts.import_results import (
    _find_extracted_root,
    build_run_metadata,
    derive_run_id,
    extract_hardware_from_transcription,
    import_results,
    load_checkpoint_data,
    parse_zip_filename,
)


class TestParseZipFilename:
    """Tests for parse_zip_filename()."""

    def test_standard_format(self, tmp_path: Path) -> None:
        """Test parsing a standard partition-date zip filename."""
        partition, date_str = parse_zip_filename(Path("tupi-2026-01-24.zip"))
        assert partition == "tupi"
        assert date_str == "2026-01-24"

    def test_multi_word_partition(self) -> None:
        """Test parsing a multi-word partition name."""
        partition, date_str = parse_zip_filename(Path("grace-a100-2026-03-15.zip"))
        assert partition == "grace-a100"
        assert date_str == "2026-03-15"

    def test_no_date_pattern(self) -> None:
        """Test fallback when filename doesn't match the pattern."""
        partition, date_str = parse_zip_filename(Path("results_backup.zip"))
        assert partition == "results_backup"
        assert date_str is None


class TestDeriveRunId:
    """Tests for derive_run_id()."""

    def test_format(self) -> None:
        """Test that derive_run_id produces the expected format."""
        started_at = datetime(2026, 1, 24, 14, 30, 0, 123456, tzinfo=UTC)
        run_id = derive_run_id(started_at, "tupi")
        assert run_id == "20260124_143000_123456_slurm_tupi"

    def test_different_partitions(self) -> None:
        """Test that different partitions produce different IDs."""
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        id1 = derive_run_id(started_at, "tupi")
        id2 = derive_run_id(started_at, "grace")
        assert id1 != id2
        assert "tupi" in id1
        assert "grace" in id2


class TestLoadCheckpointData:
    """Tests for load_checkpoint_data()."""

    def test_valid_checkpoint(self, tmp_path: Path) -> None:
        """Test loading a valid checkpoint file."""
        checkpoint = {
            "completed_files": ["file1", "file2"],
            "total_files": 3,
            "started_at": "2026-01-24T14:30:00",
            "last_updated": "2026-01-24T15:00:00",
        }
        path = tmp_path / "checkpoint.json"
        path.write_text(json.dumps(checkpoint))

        data = load_checkpoint_data(path)
        assert data["total_files"] == 3
        assert len(data["completed_files"]) == 2

    def test_missing_fields(self, tmp_path: Path) -> None:
        """Test that missing required fields raise typer.Exit."""
        checkpoint = {"completed_files": []}
        path = tmp_path / "checkpoint.json"
        path.write_text(json.dumps(checkpoint))

        with pytest.raises(ClickExit):
            load_checkpoint_data(path)


class TestExtractHardwareFromTranscription:
    """Tests for extract_hardware_from_transcription()."""

    def test_extracts_fields(self, tmp_path: Path) -> None:
        """Test extracting model_id and compute_device."""
        data = {"model_id": "openai/whisper-large-v3", "compute_device": "cuda:0"}
        path = tmp_path / "test_transcription.json"
        path.write_text(json.dumps(data))

        model_id, device = extract_hardware_from_transcription(path)
        assert model_id == "openai/whisper-large-v3"
        assert device == "cuda:0"

    def test_missing_fields_default_to_unknown(self, tmp_path: Path) -> None:
        """Test that missing fields default to 'unknown'."""
        path = tmp_path / "test_transcription.json"
        path.write_text(json.dumps({"text": "hello"}))

        model_id, device = extract_hardware_from_transcription(path)
        assert model_id == "unknown"
        assert device == "unknown"


class TestBuildRunMetadata:
    """Tests for build_run_metadata()."""

    def test_completed_status(self, tmp_path: Path) -> None:
        """Test that zero failures produce COMPLETED status."""
        metadata = build_run_metadata(
            run_id="test",
            run_dir=tmp_path,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            partition="tupi",
            total_items=10,
            completed_items=10,
            failed_items=0,
            model_id="whisper",
            compute_device="cuda:0",
        )
        assert metadata.status == RunStatus.COMPLETED
        assert metadata.pipeline_id == "test"
        assert metadata.execution.is_slurm is True

    def test_failed_status(self, tmp_path: Path) -> None:
        """Test that failures produce FAILED status."""
        metadata = build_run_metadata(
            run_id="test",
            run_dir=tmp_path,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            partition="tupi",
            total_items=10,
            completed_items=8,
            failed_items=2,
            model_id="whisper",
            compute_device="cuda:0",
        )
        assert metadata.status == RunStatus.FAILED

    def test_cpu_device_type(self, tmp_path: Path) -> None:
        """Test that non-cuda devices are passed through."""
        metadata = build_run_metadata(
            run_id="test",
            run_dir=tmp_path,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            partition="tupi",
            total_items=1,
            completed_items=1,
            failed_items=0,
            model_id="whisper",
            compute_device="cpu",
        )
        assert metadata.hardware.device_type == "cpu"


class TestFindExtractedRoot:
    """Tests for _find_extracted_root()."""

    def test_single_top_level_dir(self, tmp_path: Path) -> None:
        """Test descending into a single top-level directory."""
        inner = tmp_path / "results"
        inner.mkdir()
        (inner / "file.json").touch()

        assert _find_extracted_root(tmp_path) == inner

    def test_flat_contents(self, tmp_path: Path) -> None:
        """Test returning tmp_path when contents are flat."""
        (tmp_path / "file1.json").touch()
        (tmp_path / "file2.json").touch()

        assert _find_extracted_root(tmp_path) == tmp_path


def _create_test_zip(zip_path: Path, *, with_subdir: bool = True) -> None:
    """Create a test zip archive with checkpoint and transcription files.

    Args:
        zip_path: Path where the zip will be created.
        with_subdir: If True, wrap contents in a subdirectory inside the zip.
    """
    checkpoint = {
        "completed_files": ["file1", "file2"],
        "total_files": 3,
        "started_at": "2026-01-24T14:30:00",
        "last_updated": "2026-01-24T15:00:00",
        "failed_files": {"file3": "timeout"},
    }
    transcription = {
        "model_id": "openai/whisper-large-v3-turbo",
        "compute_device": "cuda:0",
        "text": "sample transcription",
    }

    prefix = "results/" if with_subdir else ""

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{prefix}checkpoint.json", json.dumps(checkpoint))
        zf.writestr(f"{prefix}file1_transcription.json", json.dumps(transcription))
        zf.writestr(f"{prefix}file2_transcription.json", json.dumps(transcription))


class TestImportResults:
    """Tests for the import_results() end-to-end workflow."""

    def test_successful_import(self, tmp_path: Path) -> None:
        """Test importing a valid zip creates the expected directory structure."""
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        results_dir = tmp_path / "results"
        _create_test_zip(zip_path)

        import_results(zip_path, results_dir)

        # Derive the expected pipeline_id
        started_at = datetime(2026, 1, 24, 14, 30, 0, tzinfo=UTC)
        pipeline_id = derive_run_id(started_at, "tupi")

        # Verify ID-first directory structure
        pipeline_dir = results_dir.resolve() / pipeline_id
        step_dir = pipeline_dir / "transcription"
        outputs_dir = step_dir / "outputs"

        assert pipeline_dir.exists()
        assert step_dir.exists()
        assert outputs_dir.exists()

        # Verify files were copied
        assert (outputs_dir / "file1_transcription.json").exists()
        assert (outputs_dir / "file2_transcription.json").exists()
        assert (step_dir / "checkpoint.json").exists()
        assert (step_dir / "run_metadata.json").exists()
        assert (pipeline_dir / "pipeline.json").exists()

        # Verify pipeline.json content
        with open(pipeline_dir / "pipeline.json") as f:
            pipeline_meta = json.load(f)
        assert pipeline_meta["pipeline_id"] == pipeline_id
        assert "transcription" in pipeline_meta["steps_run"]

        # Verify index.json was created
        index_path = results_dir.resolve() / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index_data = json.load(f)
        assert len(index_data["runs"]) == 1
        assert index_data["runs"][0]["pipeline_id"] == pipeline_id

    def test_idempotent_import(self, tmp_path: Path) -> None:
        """Test that importing the same zip twice skips the second time."""
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        results_dir = tmp_path / "results"
        _create_test_zip(zip_path)

        # First import
        import_results(zip_path, results_dir)

        # Get outputs dir file count
        started_at = datetime(2026, 1, 24, 14, 30, 0, tzinfo=UTC)
        pipeline_id = derive_run_id(started_at, "tupi")
        outputs_dir = results_dir.resolve() / pipeline_id / "transcription" / "outputs"
        file_count = len(list(outputs_dir.iterdir()))

        # Second import should be a no-op
        import_results(zip_path, results_dir)

        # File count unchanged (wasn't re-created)
        assert len(list(outputs_dir.iterdir())) == file_count

    def test_flat_zip_contents(self, tmp_path: Path) -> None:
        """Test importing a zip without a wrapper subdirectory."""
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        results_dir = tmp_path / "results"
        _create_test_zip(zip_path, with_subdir=False)

        import_results(zip_path, results_dir)

        started_at = datetime(2026, 1, 24, 14, 30, 0, tzinfo=UTC)
        pipeline_id = derive_run_id(started_at, "tupi")
        outputs_dir = results_dir.resolve() / pipeline_id / "transcription" / "outputs"
        assert (outputs_dir / "file1_transcription.json").exists()

    def test_missing_zip_file(self, tmp_path: Path) -> None:
        """Test that a missing zip file raises typer.Exit."""
        with pytest.raises(ClickExit):
            import_results(tmp_path / "nonexistent.zip", tmp_path / "results")

    def test_invalid_zip_file(self, tmp_path: Path) -> None:
        """Test that a non-zip file raises typer.Exit."""
        bad_file = tmp_path / "not-a-zip.zip"
        bad_file.write_text("this is not a zip")

        with pytest.raises(ClickExit):
            import_results(bad_file, tmp_path / "results")

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        """Test that a zip without checkpoint.json raises typer.Exit."""
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1_transcription.json", json.dumps({"text": "hi"}))

        with pytest.raises(ClickExit):
            import_results(zip_path, tmp_path / "results")

    def test_no_transcription_files(self, tmp_path: Path) -> None:
        """Test that a zip with checkpoint but no transcriptions raises typer.Exit."""
        checkpoint = {
            "completed_files": [],
            "total_files": 0,
            "started_at": "2026-01-24T14:30:00",
            "last_updated": "2026-01-24T15:00:00",
        }
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("checkpoint.json", json.dumps(checkpoint))

        with pytest.raises(ClickExit):
            import_results(zip_path, tmp_path / "results")

    def test_run_metadata_content(self, tmp_path: Path) -> None:
        """Test that run_metadata.json contains correct imported values."""
        zip_path = tmp_path / "tupi-2026-01-24.zip"
        results_dir = tmp_path / "results"
        _create_test_zip(zip_path)

        import_results(zip_path, results_dir)

        started_at = datetime(2026, 1, 24, 14, 30, 0, tzinfo=UTC)
        pipeline_id = derive_run_id(started_at, "tupi")
        step_dir = results_dir.resolve() / pipeline_id / "transcription"

        with open(step_dir / "run_metadata.json") as f:
            meta = json.load(f)

        assert meta["pipeline_type"] == "transcription"
        assert meta["execution"]["is_slurm"] is True
        assert meta["execution"]["slurm_partition"] == "tupi"
        assert meta["config"]["config_values"]["model_id"] == "openai/whisper-large-v3-turbo"
        assert meta["total_items"] == 3
        assert meta["completed_items"] == 2
        assert meta["failed_items"] == 1
