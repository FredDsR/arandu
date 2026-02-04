"""Results versioning manager for G-Transcriber.

Manages versioned result directories, run metadata, symlinks, and index tracking
for all pipeline outputs.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gtranscriber import __version__
from gtranscriber.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineType,
    RunMetadata,
    RunStatus,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages versioned result directories and run metadata.

    Creates timestamped run directories with format:
    results/{pipeline}/{YYYYMMDD}_{HHMMSS}_{context}/

    Where context is 'slurm_{partition}_{job_id}' for SLURM jobs or 'local' for
    local execution.
    """

    def __init__(self, base_results_dir: Path, pipeline_type: PipelineType) -> None:
        """Initialize ResultsManager.

        Args:
            base_results_dir: Base directory for all results (e.g., ./results).
            pipeline_type: Type of pipeline being executed.
        """
        self.base_dir = Path(base_results_dir).resolve()
        self.pipeline_type = pipeline_type
        self._run_dir: Path | None = None
        self._metadata: RunMetadata | None = None

    @property
    def run_dir(self) -> Path:
        """Get the current run directory.

        Returns:
            Path to the run directory.

        Raises:
            RuntimeError: If create_run() has not been called.
        """
        if self._run_dir is None:
            raise RuntimeError("create_run() must be called before accessing run_dir")
        return self._run_dir

    @property
    def outputs_dir(self) -> Path:
        """Get the outputs subdirectory for the current run.

        Returns:
            Path to the outputs directory.
        """
        return self.run_dir / "outputs"

    @property
    def metadata(self) -> RunMetadata:
        """Get the current run metadata.

        Returns:
            RunMetadata for the current run.

        Raises:
            RuntimeError: If create_run() has not been called.
        """
        if self._metadata is None:
            raise RuntimeError("create_run() must be called before accessing metadata")
        return self._metadata

    def _generate_run_id(self, execution: ExecutionEnvironment) -> str:
        """Generate a unique run ID based on timestamp and execution context.

        Format: YYYYMMDD_HHMMSS_context
        Where context is 'slurm_{partition}_{job_id}' or 'local'.

        Args:
            execution: ExecutionEnvironment with SLURM or local context.

        Returns:
            Unique run ID string.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if execution.is_slurm and execution.slurm_partition and execution.slurm_job_id:
            context = f"slurm_{execution.slurm_partition}_{execution.slurm_job_id}"
        else:
            context = "local"

        return f"{timestamp}_{context}"

    def _get_pipeline_dir(self) -> Path:
        """Get the directory for the current pipeline type.

        Returns:
            Path to the pipeline directory.
        """
        return self.base_dir / self.pipeline_type.value

    def create_run(
        self, config: BaseSettings | BaseModel, input_source: str | None = None
    ) -> RunMetadata:
        """Create a new versioned run.

        Creates the run directory structure and initializes metadata.

        Args:
            config: Pipeline configuration to snapshot.
            input_source: Optional input source description (catalog path, directory).

        Returns:
            RunMetadata for the new run.
        """
        # 1. Detect execution environment
        execution = ExecutionEnvironment.detect()

        # 2. Capture hardware info
        hardware = HardwareInfo.capture()

        # 3. Snapshot config
        config_snapshot = ConfigSnapshot.from_config(config)

        # 4. Generate run_id and create directories
        run_id = self._generate_run_id(execution)
        pipeline_dir = self._get_pipeline_dir()
        self._run_dir = pipeline_dir / run_id

        # Create directory structure
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 5. Create and save initial metadata
        self._metadata = RunMetadata(
            run_id=run_id,
            pipeline_type=self.pipeline_type,
            started_at=datetime.now(),
            status=RunStatus.IN_PROGRESS,
            execution=execution,
            hardware=hardware,
            config=config_snapshot,
            output_directory=str(self._run_dir),
            checkpoint_file=str(self._run_dir / "checkpoint.json"),
            gtranscriber_version=__version__,
            input_source=input_source,
        )

        # Save initial metadata
        metadata_path = self._run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

        logger.info(f"Created run: {run_id} at {self._run_dir}")
        return self._metadata

    def update_progress(self, completed: int, failed: int, total: int) -> None:
        """Update progress tracking for the current run.

        Args:
            completed: Number of successfully completed items.
            failed: Number of failed items.
            total: Total number of items to process.
        """
        if self._metadata is None:
            raise RuntimeError("create_run() must be called before update_progress()")

        self._metadata.completed_items = completed
        self._metadata.failed_items = failed
        self._metadata.total_items = total

        # Save updated metadata
        metadata_path = self.run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

    def complete_run(self, success: bool, error: str | None = None) -> None:
        """Mark the run as completed.

        Updates status, sets end time, updates symlinks, and updates index.

        Args:
            success: Whether the run completed successfully.
            error: Error message if the run failed.
        """
        if self._metadata is None:
            raise RuntimeError("create_run() must be called before complete_run()")

        # 1. Set ended_at and status
        self._metadata.ended_at = datetime.now()
        self._metadata.status = RunStatus.COMPLETED if success else RunStatus.FAILED
        if error:
            self._metadata.error_message = error

        # Save final metadata
        metadata_path = self.run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

        # 2. Update latest symlink
        self._update_latest_symlink()

        # 3. Update index.json
        self._update_index()

        logger.info(
            f"Run {self._metadata.run_id} completed with status: {self._metadata.status.value}"
        )

    def _update_latest_symlink(self) -> None:
        """Update the 'latest' symlink for the current pipeline type."""
        latest_dir = self.base_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = latest_dir / self.pipeline_type.value

        # Calculate relative path to the run directory
        try:
            # Get relative path from latest/ to the actual run directory
            # e.g., from results/latest to results/transcription/20260204_150000_local
            relative_target = os.path.relpath(self.run_dir, latest_dir)

            # Remove existing symlink if it exists
            if symlink_path.is_symlink():
                symlink_path.unlink()
            elif symlink_path.exists():
                # If it's a regular file/dir (shouldn't happen), remove it
                symlink_path.unlink()

            # Create new symlink with relative path for portability
            symlink_path.symlink_to(relative_target)
            logger.debug(f"Updated latest symlink: {symlink_path} -> {relative_target}")

        except OSError as e:
            logger.warning(f"Failed to update latest symlink: {e}")

    def _update_index(self) -> None:
        """Update the global index.json with this run's information.

        Uses file locking for thread-safe updates.
        """
        if self._metadata is None:
            return

        index_path = self.base_dir / "index.json"

        # Create run entry for index
        run_entry = {
            "run_id": self._metadata.run_id,
            "pipeline_type": self._metadata.pipeline_type.value,
            "status": self._metadata.status.value,
            "started_at": self._metadata.started_at.isoformat(),
            "ended_at": self._metadata.ended_at.isoformat() if self._metadata.ended_at else None,
            "duration_seconds": self._metadata.duration_seconds,
            "total_items": self._metadata.total_items,
            "completed_items": self._metadata.completed_items,
            "failed_items": self._metadata.failed_items,
            "success_rate": self._metadata.success_rate,
            "output_directory": self._metadata.output_directory,
        }

        try:
            # Ensure parent directory exists
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # Use file locking for thread-safe access
            mode = "r+" if index_path.exists() else "w"
            with open(index_path, mode, encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Read existing index if it exists
                    if mode == "r+":
                        f.seek(0)
                        try:
                            index_data = json.load(f)
                        except json.JSONDecodeError:
                            index_data = {"runs": []}
                    else:
                        index_data = {"runs": []}

                    # Remove any existing entry for this run_id and pipeline_type combo
                    index_data["runs"] = [
                        r
                        for r in index_data["runs"]
                        if not (
                            r.get("run_id") == self._metadata.run_id
                            and r.get("pipeline_type") == self._metadata.pipeline_type.value
                        )
                    ]

                    # Add new entry at the beginning
                    index_data["runs"].insert(0, run_entry)
                    index_data["last_updated"] = datetime.now().isoformat()

                    # Write updated index
                    f.seek(0)
                    f.truncate()
                    json.dump(index_data, f, indent=2)

                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(f"Updated index.json with run {self._metadata.run_id}")

        except OSError as e:
            logger.warning(f"Failed to update index.json: {e}")

    @classmethod
    def get_latest_run(cls, base_dir: Path, pipeline: PipelineType) -> RunMetadata | None:
        """Get the metadata for the latest run of a pipeline type.

        Args:
            base_dir: Base results directory.
            pipeline: Pipeline type to look up.

        Returns:
            RunMetadata for the latest run, or None if no runs exist.
        """
        latest_symlink = Path(base_dir) / "latest" / pipeline.value

        if latest_symlink.is_symlink():
            target = latest_symlink.resolve()
            metadata_path = target / "run_metadata.json"
            if metadata_path.exists():
                return RunMetadata.load(metadata_path)

        # Fallback: find the most recent run directory
        pipeline_dir = Path(base_dir) / pipeline.value
        if not pipeline_dir.exists():
            return None

        run_dirs = sorted(pipeline_dir.iterdir(), reverse=True)
        for run_dir in run_dirs:
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                return RunMetadata.load(metadata_path)

        return None

    @classmethod
    def list_runs(cls, base_dir: Path, pipeline: PipelineType | None = None) -> list[dict]:
        """List all runs, optionally filtered by pipeline type.

        Args:
            base_dir: Base results directory.
            pipeline: Optional pipeline type to filter by.

        Returns:
            List of run summary dictionaries.
        """
        index_path = Path(base_dir) / "index.json"

        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    index_data = json.load(f)

                runs = index_data.get("runs", [])

                # Filter by pipeline type if specified
                if pipeline is not None:
                    runs = [r for r in runs if r.get("pipeline_type") == pipeline.value]

                return runs

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read index.json: {e}")

        # Fallback: scan directories
        return cls._scan_runs(base_dir, pipeline)

    @classmethod
    def _scan_runs(cls, base_dir: Path, pipeline: PipelineType | None = None) -> list[dict]:
        """Scan directories to build run list (fallback when index is unavailable).

        Args:
            base_dir: Base results directory.
            pipeline: Optional pipeline type to filter by.

        Returns:
            List of run summary dictionaries.
        """
        runs: list[dict] = []
        base_path = Path(base_dir)

        if not base_path.exists():
            return runs

        pipelines = [pipeline] if pipeline else list(PipelineType)

        for p in pipelines:
            pipeline_dir = base_path / p.value
            if not pipeline_dir.exists():
                continue

            for run_dir in pipeline_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if run_dir.name == "latest":
                    continue

                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = RunMetadata.load(metadata_path)
                        runs.append(
                            {
                                "run_id": metadata.run_id,
                                "pipeline_type": metadata.pipeline_type.value,
                                "status": metadata.status.value,
                                "started_at": metadata.started_at.isoformat(),
                                "ended_at": (
                                    metadata.ended_at.isoformat() if metadata.ended_at else None
                                ),
                                "duration_seconds": metadata.duration_seconds,
                                "total_items": metadata.total_items,
                                "completed_items": metadata.completed_items,
                                "failed_items": metadata.failed_items,
                                "success_rate": metadata.success_rate,
                                "output_directory": metadata.output_directory,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {run_dir}: {e}")

        # Sort by started_at descending
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs
