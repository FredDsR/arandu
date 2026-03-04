"""Import externally produced results into the versioned results structure.

Usage:
    uv run python scripts/import_results.py tupi-2026-01-24.zip
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from arandu import __version__
from arandu.core.results_manager import ResultsManager
from arandu.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineMetadata,
    PipelineType,
    RunMetadata,
    RunStatus,
)
from arandu.utils.logger import print_error, print_info, print_success, print_warning

app = typer.Typer(help="Import externally produced results into the versioned structure.")

# Pattern: {partition}-{YYYY-MM-DD}.zip
ZIP_FILENAME_PATTERN = re.compile(r"^(.+)-(\d{4}-\d{2}-\d{2})\.zip$")


def parse_zip_filename(zip_path: Path) -> tuple[str, str | None]:
    """Parse partition name and optional date from a zip filename.

    Expected format: ``{partition}-{YYYY-MM-DD}.zip``.
    Falls back to the entire stem as the partition if the pattern doesn't match.

    Args:
        zip_path: Path to the zip file.

    Returns:
        Tuple of (partition, date_str or None).
    """
    match = ZIP_FILENAME_PATTERN.match(zip_path.name)
    if match:
        return match.group(1), match.group(2)
    return zip_path.stem, None


def derive_run_id(started_at: datetime, partition: str) -> str:
    """Derive a run ID from a checkpoint's started_at timestamp and partition name.

    Format: ``YYYYMMDD_HHMMSS_ffffff_slurm_{partition}``.

    Args:
        started_at: The checkpoint's started_at timestamp.
        partition: SLURM partition name (e.g., 'tupi').

    Returns:
        Run ID string.
    """
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_slurm_{partition}"


def load_checkpoint_data(path: Path) -> dict:
    """Load and validate checkpoint JSON data.

    Args:
        path: Path to checkpoint.json.

    Returns:
        Parsed checkpoint dictionary.

    Raises:
        typer.Exit: If required fields are missing.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["completed_files", "total_files", "started_at", "last_updated"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        print_error(f"Checkpoint missing required fields: {', '.join(missing)}")
        raise typer.Exit(code=1)

    return data


def extract_hardware_from_transcription(path: Path) -> tuple[str, str]:
    """Extract model_id and compute_device from a transcription JSON file.

    Args:
        path: Path to a ``*_transcription.json`` file.

    Returns:
        Tuple of (model_id, compute_device).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model_id", "unknown"), data.get("compute_device", "unknown")


def build_run_metadata(
    run_id: str,
    run_dir: Path,
    started_at: datetime,
    ended_at: datetime,
    partition: str,
    total_items: int,
    completed_items: int,
    failed_items: int,
    model_id: str,
    compute_device: str,
) -> RunMetadata:
    """Build RunMetadata for an imported run.

    Args:
        run_id: Unique run identifier.
        run_dir: Path to the run directory.
        started_at: Run start time.
        ended_at: Run end time.
        partition: SLURM partition name.
        total_items: Total items to process.
        completed_items: Successfully completed items.
        failed_items: Number of failed items.
        model_id: Model identifier from transcription output.
        compute_device: Compute device from transcription output.

    Returns:
        Populated RunMetadata instance.
    """
    has_failures = failed_items > 0
    status = RunStatus.FAILED if has_failures else RunStatus.COMPLETED

    execution = ExecutionEnvironment(
        is_slurm=True,
        is_local=False,
        slurm_partition=partition,
        hostname=f"slurm-{partition}",
        username="imported",
    )

    device_type = "cuda" if compute_device.startswith("cuda") else compute_device
    hardware = HardwareInfo(
        device_type=device_type,
        cpu_count=0,
        torch_version="unknown",
        python_version="unknown",
    )

    config_snapshot = ConfigSnapshot(
        config_type="ImportedRun",
        config_values={
            "model_id": model_id,
            "compute_device": compute_device,
            "source": "slurm_import",
        },
    )

    return RunMetadata(
        run_id=run_id,
        pipeline_id=run_id,
        pipeline_type=PipelineType.TRANSCRIPTION,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        execution=execution,
        hardware=hardware,
        config=config_snapshot,
        total_items=total_items,
        completed_items=completed_items,
        failed_items=failed_items,
        output_directory=str(run_dir),
        checkpoint_file=str(run_dir / "checkpoint.json"),
        arandu_version=__version__,
        input_source="slurm_import",
    )


def _find_extracted_root(tmp_path: Path) -> Path:
    """Find the root directory of extracted zip contents.

    Handles zips that contain a single top-level directory (descend into it)
    or flat contents (use tmp_path directly).

    Args:
        tmp_path: Path to the temporary extraction directory.

    Returns:
        Path to the directory containing the actual result files.
    """
    entries = list(tmp_path.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return tmp_path


def import_results(zip_path: Path, results_dir: Path) -> None:
    """Import results from a zip archive into the ID-first results structure.

    Creates ``results/{pipeline_id}/transcription/outputs/`` with pipeline.json.

    Args:
        zip_path: Path to the zip archive.
        results_dir: Base results directory (e.g., ./results).

    Raises:
        typer.Exit: On validation errors.
    """
    # 1. Validate zip file
    if not zip_path.exists():
        print_error(f"Zip file not found: {zip_path}")
        raise typer.Exit(code=1)

    if not zipfile.is_zipfile(zip_path):
        print_error(f"Not a valid zip file: {zip_path}")
        raise typer.Exit(code=1)

    # 2. Parse zip filename
    partition, date_str = parse_zip_filename(zip_path)
    print_info(f"Partition: {partition}" + (f", date: {date_str}" if date_str else ""))

    # 3. Extract to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print_info(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        # 4. Find extracted root
        extracted_root = _find_extracted_root(tmp_path)
        print_info(f"Extracted root: {extracted_root.name}")

        # 5. Load checkpoint
        checkpoint_path = extracted_root / "checkpoint.json"
        if not checkpoint_path.exists():
            print_error("checkpoint.json not found in archive")
            raise typer.Exit(code=1)

        checkpoint = load_checkpoint_data(checkpoint_path)
        started_at = datetime.fromisoformat(checkpoint["started_at"]).replace(tzinfo=UTC)
        last_updated = datetime.fromisoformat(checkpoint["last_updated"]).replace(tzinfo=UTC)
        total_files = checkpoint["total_files"]
        completed_count = len(checkpoint["completed_files"])
        failed_files = checkpoint.get("failed_files", {})
        failed_count = len(failed_files)

        print_info(
            f"Checkpoint: {completed_count} completed, {failed_count} failed, {total_files} total"
        )

        # 6. Derive pipeline_id (same as run_id)
        pipeline_id = derive_run_id(started_at, partition)
        print_info(f"Pipeline ID: {pipeline_id}")

        # 7. Idempotency check (ID-first layout)
        pipeline_dir = results_dir.resolve() / pipeline_id
        step_dir = pipeline_dir / PipelineType.TRANSCRIPTION.value
        if step_dir.exists():
            print_warning(f"Step directory already exists: {step_dir}")
            print_warning("Skipping import (idempotent).")
            return

        # 8. Glob transcription files
        transcription_files = list(extracted_root.glob("*_transcription.json"))
        if not transcription_files:
            print_error("No *_transcription.json files found in archive")
            raise typer.Exit(code=1)

        print_info(f"Found {len(transcription_files)} transcription files")

        # 9. Extract hardware info from first transcription file
        model_id, compute_device = extract_hardware_from_transcription(transcription_files[0])
        print_info(f"Model: {model_id}, device: {compute_device}")

        # 10. Create ID-first directory structure
        outputs_dir = step_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # 11. Copy transcription files into outputs/
        for src_file in transcription_files:
            shutil.copy2(src_file, outputs_dir / src_file.name)

        # 12. Copy checkpoint.json into step directory
        shutil.copy2(checkpoint_path, step_dir / "checkpoint.json")

        # 13. Build and save RunMetadata
        metadata = build_run_metadata(
            run_id=pipeline_id,
            run_dir=step_dir,
            started_at=started_at,
            ended_at=last_updated,
            partition=partition,
            total_items=total_files,
            completed_items=completed_count,
            failed_items=failed_count,
            model_id=model_id,
            compute_device=compute_device,
        )
        metadata.save(step_dir / "run_metadata.json")

        # 14. Create pipeline.json
        pipeline_meta = PipelineMetadata(
            pipeline_id=pipeline_id,
            created_at=started_at,
            steps_run=[PipelineType.TRANSCRIPTION.value],
        )
        pipeline_meta.save(pipeline_dir / "pipeline.json")

        # 15. Register in global index via ResultsManager
        manager = ResultsManager(results_dir, PipelineType.TRANSCRIPTION, pipeline_id=pipeline_id)
        manager.register_external_run(metadata)

    print_success(f"Imported {len(transcription_files)} files into {step_dir}")


@app.command()
def main(
    zip_file: Annotated[Path, typer.Argument(help="Path to the zip archive to import.")],
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", "-r", help="Base results directory."),
    ] = Path("./results"),
) -> None:
    """Import SLURM transcription results from a zip archive."""
    import_results(zip_file, results_dir)


if __name__ == "__main__":
    app()
