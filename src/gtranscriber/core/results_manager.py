"""Results versioning manager for G-Transcriber.

Manages versioned result directories using a pipeline-ID-first layout where all
steps share one pipeline ID under ``results/{pipeline_id}/{step}/``.
"""

from __future__ import annotations

import fcntl  # Unix-only: this project targets Linux/SLURM environments
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gtranscriber import __version__
from gtranscriber.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineMetadata,
    PipelineType,
    ReplicationInfo,
    RunMetadata,
    RunStatus,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages versioned result directories using an ID-first layout.

    Directory structure::

        results/
          {pipeline_id}/
            pipeline.json
            transcription/
              run_metadata.json
              checkpoint.json
              outputs/*.json
            qa/
              run_metadata.json
              qa_checkpoint.json
              outputs/*.json
          index.json
    """

    def __init__(
        self,
        base_results_dir: Path,
        pipeline_type: PipelineType,
        pipeline_id: str | None = None,
    ) -> None:
        """Initialize ResultsManager.

        Args:
            base_results_dir: Base directory for all results (e.g., ./results).
            pipeline_type: Type of pipeline being executed.
            pipeline_id: Explicit pipeline ID. Auto-generated if None.
        """
        self.base_dir = Path(base_results_dir).resolve()
        self.pipeline_type = pipeline_type
        self._pipeline_id = pipeline_id
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

    def _generate_pipeline_id(self, execution: ExecutionEnvironment) -> str:
        """Generate a unique pipeline ID based on timestamp and execution context.

        Format: YYYYMMDD_HHMMSS_ffffff_context
        Where context is 'slurm_{partition}_{job_id}' or 'local'.

        Args:
            execution: ExecutionEnvironment with SLURM or local context.

        Returns:
            Unique pipeline ID string.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        if execution.is_slurm and execution.slurm_partition and execution.slurm_job_id:
            context = f"slurm_{execution.slurm_partition}_{execution.slurm_job_id}"
        else:
            context = "local"

        return f"{timestamp}_{context}"

    def create_run(
        self,
        config: BaseSettings | BaseModel,
        input_source: str | None = None,
        checkpoint_filename: str = "checkpoint.json",
    ) -> RunMetadata:
        """Create a new versioned run under the pipeline ID.

        Creates ``results/{pipeline_id}/{step}/outputs/`` and writes both
        ``pipeline.json`` (pipeline-level) and ``run_metadata.json`` (step-level).

        Args:
            config: Pipeline configuration to snapshot.
            input_source: Optional input source description (catalog path, directory).
            checkpoint_filename: Name of the checkpoint file within the run directory.

        Returns:
            RunMetadata for the new run.
        """
        # 1. Detect execution environment
        execution = ExecutionEnvironment.detect()

        # 2. Capture hardware info
        hardware = HardwareInfo.capture()

        # 3. Snapshot config
        config_snapshot = ConfigSnapshot.from_config(config)

        # 4. Determine pipeline_id
        pipeline_id = self._pipeline_id or self._generate_pipeline_id(execution)
        self._pipeline_id = pipeline_id

        # 5. Create step directory (overwrite if exists)
        step = self.pipeline_type.value
        pipeline_dir = self.base_dir / pipeline_id
        self._run_dir = pipeline_dir / step

        if self._run_dir.exists():
            shutil.rmtree(self._run_dir)

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 6. Create/update pipeline.json
        pipeline_json_path = pipeline_dir / "pipeline.json"
        if pipeline_json_path.exists():
            pipeline_meta = PipelineMetadata.load(pipeline_json_path)
            if step not in pipeline_meta.steps_run:
                pipeline_meta.steps_run.append(step)
        else:
            pipeline_meta = PipelineMetadata(
                pipeline_id=pipeline_id,
                steps_run=[step],
            )
        pipeline_meta.save(pipeline_json_path)

        # 7. Create and save initial run metadata
        run_id = pipeline_id  # Use pipeline_id as the run_id for simplicity
        self._metadata = RunMetadata(
            run_id=run_id,
            pipeline_id=pipeline_id,
            pipeline_type=self.pipeline_type,
            started_at=datetime.now(UTC),
            status=RunStatus.IN_PROGRESS,
            execution=execution,
            hardware=hardware,
            config=config_snapshot,
            output_directory=str(self._run_dir),
            checkpoint_file=str(self._run_dir / checkpoint_filename),
            gtranscriber_version=__version__,
            input_source=input_source,
        )

        metadata_path = self._run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

        logger.info(f"Created run: {pipeline_id}/{step} at {self._run_dir}")
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

        metadata_path = self.run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

    def complete_run(self, success: bool, error: str | None = None) -> None:
        """Mark the run as completed.

        Updates status, sets end time, and updates index.

        Args:
            success: Whether the run completed successfully.
            error: Error message if the run failed.
        """
        if self._metadata is None:
            raise RuntimeError("create_run() must be called before complete_run()")

        self._metadata.ended_at = datetime.now(UTC)
        self._metadata.status = RunStatus.COMPLETED if success else RunStatus.FAILED
        if error:
            self._metadata.error_message = error

        metadata_path = self.run_dir / "run_metadata.json"
        self._metadata.save(metadata_path)

        self._update_index()

        logger.info(
            f"Run {self._metadata.run_id}/{self.pipeline_type.value} "
            f"completed with status: {self._metadata.status.value}"
        )

    def register_external_run(self, run_metadata: RunMetadata) -> None:
        """Register a pre-built external run in the index.

        Use this when run directories and metadata were constructed externally
        (e.g., from an imported SLURM result) and only need to be registered
        in the global index.

        Args:
            run_metadata: Pre-built RunMetadata instance. Its ``output_directory``
                must point to an existing step directory.

        Raises:
            ValueError: If the step directory does not exist.
        """
        step_dir = Path(run_metadata.output_directory)
        if not step_dir.exists():
            raise ValueError(f"Step directory not found: {step_dir}")

        self._run_dir = step_dir
        self._metadata = run_metadata
        self._update_index()

        logger.info(
            f"Registered external run: {run_metadata.pipeline_id}/"
            f"{run_metadata.pipeline_type.value}"
        )

    def _update_index(self) -> None:
        """Update the global index.json with this run's information.

        Uses file locking for thread-safe updates.
        """
        if self._metadata is None:
            return

        index_path = self.base_dir / "index.json"

        run_entry = {
            "run_id": self._metadata.run_id,
            "pipeline_id": self._metadata.pipeline_id,
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
            index_path.parent.mkdir(parents=True, exist_ok=True)

            with open(index_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    content = f.read()
                    if content.strip():
                        try:
                            index_data = json.loads(content)
                        except json.JSONDecodeError:
                            index_data = {"runs": []}
                    else:
                        index_data = {"runs": []}

                    # Remove any existing entry for this pipeline_id + pipeline_type
                    index_data["runs"] = [
                        r
                        for r in index_data["runs"]
                        if not (
                            r.get("pipeline_id") == self._metadata.pipeline_id
                            and r.get("pipeline_type") == self._metadata.pipeline_type.value
                        )
                    ]

                    index_data["runs"].insert(0, run_entry)
                    index_data["last_updated"] = datetime.now(UTC).isoformat()

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

        Scans ``results/*/`` directories for ``{step}/run_metadata.json`` and
        returns the most recent by ``started_at``.

        Args:
            base_dir: Base results directory.
            pipeline: Pipeline type to look up.

        Returns:
            RunMetadata for the latest run, or None if no runs exist.
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return None

        step = pipeline.value
        latest_metadata: RunMetadata | None = None

        for pipeline_dir in base_path.iterdir():
            if not pipeline_dir.is_dir():
                continue
            metadata_path = pipeline_dir / step / "run_metadata.json"
            if metadata_path.exists():
                try:
                    metadata = RunMetadata.load(metadata_path)
                    if latest_metadata is None or metadata.started_at > latest_metadata.started_at:
                        latest_metadata = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        return latest_metadata

    @classmethod
    def resolve_outputs(cls, base_dir: Path, pipeline_id: str, step: PipelineType) -> Path | None:
        """Resolve the outputs directory for a specific pipeline ID and step.

        Direct lookup at ``results/{pipeline_id}/{step}/outputs/``.

        Args:
            base_dir: Base results directory.
            pipeline_id: Pipeline identifier.
            step: Pipeline type/step to look up.

        Returns:
            Path to the outputs directory if found, or None.
        """
        outputs = Path(base_dir) / pipeline_id / step.value / "outputs"
        if outputs.is_dir():
            return outputs
        return None

    @classmethod
    def resolve_latest_outputs(cls, base_dir: Path, pipeline: PipelineType) -> Path | None:
        """Resolve the outputs directory for the latest run of a pipeline type.

        Scans all pipeline dirs for the most recent one with that step.

        Args:
            base_dir: Base results directory.
            pipeline: Pipeline type to look up.

        Returns:
            Path to the outputs directory if found, or None.
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return None

        step = pipeline.value
        best_outputs: Path | None = None
        best_time: datetime | None = None

        for pipeline_dir in base_path.iterdir():
            if not pipeline_dir.is_dir():
                continue
            step_dir = pipeline_dir / step
            outputs = step_dir / "outputs"
            if not outputs.is_dir():
                continue

            metadata_path = step_dir / "run_metadata.json"
            if metadata_path.exists():
                try:
                    metadata = RunMetadata.load(metadata_path)
                    if best_time is None or metadata.started_at > best_time:
                        best_time = metadata.started_at
                        best_outputs = outputs
                except Exception:
                    # If metadata unreadable, still consider the directory
                    if best_outputs is None:
                        best_outputs = outputs
            elif best_outputs is None:
                best_outputs = outputs

        return best_outputs

    @classmethod
    def get_latest_pipeline_id(cls, base_dir: Path) -> str | None:
        """Get the most recent pipeline ID by scanning pipeline.json files.

        Args:
            base_dir: Base results directory.

        Returns:
            Pipeline ID string, or None if no pipelines found.
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return None

        best_id: str | None = None
        best_time: datetime | None = None

        for pipeline_dir in base_path.iterdir():
            if not pipeline_dir.is_dir():
                continue
            pipeline_json = pipeline_dir / "pipeline.json"
            if pipeline_json.exists():
                try:
                    meta = PipelineMetadata.load(pipeline_json)
                    if best_time is None or meta.created_at > best_time:
                        best_time = meta.created_at
                        best_id = meta.pipeline_id
                except Exception as e:
                    logger.warning(f"Failed to load pipeline.json from {pipeline_dir}: {e}")

        return best_id

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

                if pipeline is not None:
                    runs = [r for r in runs if r.get("pipeline_type") == pipeline.value]

                return runs

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read index.json: {e}")

        return cls._scan_runs(base_dir, pipeline)

    @classmethod
    def _scan_runs(cls, base_dir: Path, pipeline: PipelineType | None = None) -> list[dict]:
        """Scan directories to build run list (fallback when index is unavailable).

        Scans ``results/*/{step}/run_metadata.json`` for the ID-first layout.

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

        for pipeline_dir in base_path.iterdir():
            if not pipeline_dir.is_dir():
                continue
            # Skip non-pipeline directories (index.json, etc.)
            if not (pipeline_dir / "pipeline.json").exists():
                continue

            for p in pipelines:
                metadata_path = pipeline_dir / p.value / "run_metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    metadata = RunMetadata.load(metadata_path)
                    runs.append(
                        {
                            "run_id": metadata.run_id,
                            "pipeline_id": metadata.pipeline_id,
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
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    @staticmethod
    def _generate_new_pipeline_id() -> str:
        """Generate a unique pipeline ID based on timestamp and execution context.

        Returns:
            Unique pipeline ID string.
        """
        execution = ExecutionEnvironment.detect()
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        if execution.is_slurm and execution.slurm_partition and execution.slurm_job_id:
            context = f"slurm_{execution.slurm_partition}_{execution.slurm_job_id}"
        else:
            context = "local"

        return f"{timestamp}_{context}"

    @classmethod
    def replicate_pipeline(
        cls,
        base_dir: Path,
        source_pipeline_id: str,
        target_pipeline_id: str | None = None,
    ) -> str:
        """Replicate (clone) a pipeline run to a new ID without re-executing.

        Copies the entire pipeline directory tree and rewrites metadata with the
        new pipeline ID. Enables experimentation workflows where users clone a
        completed run and re-run only specific steps.

        Args:
            base_dir: Base results directory.
            source_pipeline_id: Pipeline ID to replicate from.
            target_pipeline_id: New pipeline ID. Auto-generated if None.

        Returns:
            The new pipeline ID.

        Raises:
            ValueError: If source pipeline not found or target already exists.
        """
        base_dir = Path(base_dir).resolve()
        source_dir = base_dir / source_pipeline_id

        # 1. Validate source exists
        source_pipeline_json = source_dir / "pipeline.json"
        if not source_pipeline_json.exists():
            raise ValueError(
                f"Source pipeline not found: {source_pipeline_id} "
                f"(no pipeline.json at {source_pipeline_json})"
            )

        # 2. Generate or validate target ID
        if target_pipeline_id is None:
            target_pipeline_id = cls._generate_new_pipeline_id()

        target_dir = base_dir / target_pipeline_id
        if target_dir.exists():
            raise ValueError(f"Target pipeline already exists: {target_pipeline_id}")

        # 3. Copy directory tree
        shutil.copytree(source_dir, target_dir)

        # 4. Rewrite pipeline.json
        pipeline_meta = PipelineMetadata.load(target_dir / "pipeline.json")
        pipeline_meta = PipelineMetadata(
            pipeline_id=target_pipeline_id,
            created_at=datetime.now(UTC),
            steps_run=pipeline_meta.steps_run,
            replicated_from=ReplicationInfo(source_pipeline_id=source_pipeline_id),
        )
        pipeline_meta.save(target_dir / "pipeline.json")

        # 5. Rewrite each step's run_metadata.json and register in index
        for step_name in pipeline_meta.steps_run:
            step_dir = target_dir / step_name
            metadata_path = step_dir / "run_metadata.json"
            if not metadata_path.exists():
                continue

            run_meta = RunMetadata.load(metadata_path)

            # Update identity and paths
            run_meta.pipeline_id = target_pipeline_id
            run_meta.run_id = target_pipeline_id
            run_meta.output_directory = str(step_dir)
            if run_meta.checkpoint_file is not None:
                checkpoint_name = Path(run_meta.checkpoint_file).name
                run_meta.checkpoint_file = str(step_dir / checkpoint_name)

            run_meta.save(metadata_path)

            # Register in index
            try:
                pipeline_type = PipelineType(step_name)
                manager = cls(base_dir, pipeline_type, pipeline_id=target_pipeline_id)
                manager.register_external_run(run_meta)
            except ValueError:
                logger.warning(f"Unknown pipeline type during replication: {step_name}")

        logger.info(f"Replicated pipeline {source_pipeline_id} -> {target_pipeline_id}")
        return target_pipeline_id
