"""Cross-domain Pydantic schemas shared across pipeline domains."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from arandu.shared.judge.schemas import (
    JudgeResultMixin,
)

# Sentinel for "key not present" — distinguishes a missing key from an
# explicit ``None`` value during legacy-payload migration.
_MISSING = object()


class InputRecord(BaseModel):
    """Schema for input records from file metadata.

    Validates the existence of critical fields like file_id, mimeType and parents
    before processing.
    """

    file_id: str = Field(..., alias="gdrive_id", description="Unique file identifier")
    name: str = Field(..., description="File name")
    mimeType: str = Field(..., description="MIME type of the file")
    parents: list[str] = Field(..., description="List of parent folder IDs")
    web_content_link: str = Field(..., alias="webContentLink", description="Direct download link")
    size_bytes: int | None = Field(None, description="File size in bytes")
    duration_milliseconds: int | None = Field(
        None, description="Media duration in milliseconds (for audio/video files)"
    )

    model_config = {"populate_by_name": True}

    @field_validator("parents", mode="before")
    @classmethod
    def parse_parents(cls, v: str | list[str]) -> list[str]:
        """Parse parents field from string or list format."""
        if isinstance(v, str):
            try:
                # Handle single-quoted JSON strings
                return json.loads(v.replace("'", '"'))
            except json.JSONDecodeError:
                return []
        if isinstance(v, list):
            return v
        return []

    @field_validator("size_bytes", mode="before")
    @classmethod
    def parse_size_bytes(cls, v: str | int | None) -> int | None:
        """Parse size_bytes from string or int format."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return None
        return None


class TranscriptionSegment(BaseModel):
    """Schema for a transcription segment with timestamp information."""

    text: str = Field(..., description="Transcribed text for this segment")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class SourceMetadata(BaseModel):
    """Metadata extracted from catalog source information.

    Provides provenance context about the interview: who was recorded,
    who conducted the recording, where and when it happened, and any
    sequence information for multi-part recordings.
    """

    participant_name: str | None = Field(None, description="Name of the interviewee/participant")
    researcher_name: str | None = Field(None, description="Name of the researcher/interviewer")
    location: str | None = Field(None, description="Recording location (e.g., 'Barra de Pelotas')")
    recording_date: str | None = Field(
        None, description="Recording date extracted from filename (original format preserved)"
    )
    sequence_number: int | None = Field(
        None, ge=1, description="Sequence number for multi-part recordings"
    )
    sequence_label: str | None = Field(
        None, description="Sequence label (e.g., 'Parte I', 'Parte II')"
    )
    event_context: str | None = Field(
        None, description="Event context from folder path (e.g., 'Audiência Câmara de Vereadores')"
    )
    source_gdrive_path: str | None = Field(
        None, description="Original Google Drive path for traceability"
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence of metadata extraction (0.0-1.0)"
    )
    custom_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Extensible key-value metadata for catalog-specific fields",
    )


class EnrichedRecord(InputRecord, JudgeResultMixin):
    """Schema for output records containing transcription results and metadata.

    This schema defines the format of the final JSON file that will be saved
    to Google Drive alongside the original media file.
    """

    transcription_text: str = Field(..., description="Full transcription text")
    detected_language: str = Field(..., description="Detected language code")
    language_probability: float = Field(..., description="Confidence score for detected language")
    model_id: str = Field(..., description="Hugging Face model ID used for transcription")
    compute_device: str = Field(..., description="Device used for computation (cpu/cuda/mps)")
    processing_duration_sec: float = Field(..., description="Processing time in seconds")
    transcription_status: str = Field(..., description="Status of transcription process")
    created_at_enrichment: datetime = Field(
        default_factory=datetime.now, description="Timestamp of enrichment"
    )
    segments: list[TranscriptionSegment] | None = Field(
        None, description="Detailed timestamp segments"
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_quality_field(cls, data: object) -> object:
        """Adapt records that still carry the retired ``transcription_quality`` key.

        Two pre-mixin shapes can show up on disk:

        - The very old weighted-score struct (``overall_score``,
          ``script_match_score``, …) — drop it; those scores weren't
          consumed anyway and the record becomes re-judgeable.
        - The interim ``JudgePipelineResult`` payload that was briefly
          stored under ``transcription_quality`` — rename it to
          ``validation`` so the mixin picks it up.
        """
        if not isinstance(data, dict):
            return data
        tq = data.get("transcription_quality", _MISSING)
        if tq is _MISSING:
            return data
        new_data = {k: v for k, v in data.items() if k != "transcription_quality"}
        if isinstance(tq, dict) and "stage_results" in tq:
            new_data.setdefault("validation", tq)
        # Anything else (legacy weighted-score struct, or None) → drop the key.
        return new_data

    source_metadata: SourceMetadata | None = Field(
        default=None,
        description="Extracted metadata from source catalog (None = not yet extracted)",
    )

    def ensure_language_metadata(self) -> None:
        """Ensure metadata compatibility for AutoSchemaKG language routing.

        AutoSchemaKG uses a 'lang' field in metadata for language-specific prompts.
        This method can be called to ensure compatibility if needed.
        """
        # Note: This is a placeholder for future metadata handling
        # The detected_language field already provides the language code
        pass


# Bloom's Taxonomy levels for cognitive scaffolding
BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]


# =============================================================================
# Results Versioning Schemas
# =============================================================================


def _utc_now() -> datetime:
    """Return the current UTC datetime for consistent timestamp capture."""
    return datetime.now(UTC)


class PipelineType(StrEnum):
    """Enum representing the different pipeline types."""

    TRANSCRIPTION = "transcription"
    QA = "qa"
    CEP = "cep"
    KG = "kg"
    EVALUATION = "evaluation"


class ReplicationInfo(BaseModel):
    """Provenance info for a replicated pipeline."""

    source_pipeline_id: str = Field(
        ...,
        description="Pipeline ID this was replicated from",
    )
    replicated_at: datetime = Field(
        default_factory=_utc_now,
        description="When the replication occurred (UTC)",
    )


class PipelineMetadata(BaseModel):
    """Metadata for a pipeline run group sharing a single pipeline ID.

    Stored as ``pipeline.json`` at ``results/{pipeline_id}/``.
    """

    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    created_at: datetime = Field(
        default_factory=_utc_now, description="When the pipeline was first created (UTC)"
    )
    steps_run: list[str] = Field(
        default_factory=list, description="Pipeline steps executed (e.g. ['transcription', 'qa'])"
    )
    schema_version: str = Field(default="2.0", description="Schema version for compatibility")
    replicated_from: ReplicationInfo | None = Field(
        default=None,
        description="Provenance info if this pipeline was replicated from another",
    )

    def save(self, path: str | Path) -> None:
        """Save pipeline metadata to JSON file.

        Args:
            path: Path to save the metadata file.
        """
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PipelineMetadata:
        """Load pipeline metadata from JSON file.

        Args:
            path: Path to the metadata file.

        Returns:
            PipelineMetadata instance.
        """
        return cls.model_validate_json(Path(path).read_text())


class RunStatus(StrEnum):
    """Enum representing the status of a pipeline run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionEnvironment(BaseModel):
    """Captures execution environment information including SLURM detection."""

    is_slurm: bool = Field(default=False, description="Whether running in SLURM environment")
    is_local: bool = Field(default=True, description="Whether running locally")
    slurm_job_id: str | None = Field(default=None, description="SLURM job ID")
    slurm_partition: str | None = Field(default=None, description="SLURM partition name")
    slurm_node: str | None = Field(default=None, description="SLURM node hostname")
    hostname: str = Field(..., description="System hostname")
    username: str = Field(..., description="Current username")

    @classmethod
    def detect(cls) -> ExecutionEnvironment:
        """Detect execution environment from environment variables.

        Returns:
            ExecutionEnvironment instance with detected values.
        """
        import getpass
        import os
        import socket

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_partition = os.environ.get("SLURM_JOB_PARTITION")
        slurm_node = os.environ.get("SLURMD_NODENAME")
        is_slurm = slurm_job_id is not None

        return cls(
            is_slurm=is_slurm,
            is_local=not is_slurm,
            slurm_job_id=slurm_job_id,
            slurm_partition=slurm_partition,
            slurm_node=slurm_node,
            hostname=socket.gethostname(),
            username=getpass.getuser(),
        )


class HardwareInfo(BaseModel):
    """Captures hardware information for reproducibility."""

    device_type: str = Field(..., description="Device type: cpu, cuda, mps")
    gpu_name: str | None = Field(default=None, description="GPU name if available")
    gpu_memory_gb: float | None = Field(default=None, description="GPU memory in GB")
    cuda_version: str | None = Field(default=None, description="CUDA version if available")
    cpu_count: int = Field(..., description="Number of CPU cores")
    torch_version: str = Field(..., description="PyTorch version")
    python_version: str = Field(..., description="Python version")

    @classmethod
    def capture(cls) -> HardwareInfo:
        """Capture current hardware information.

        Returns:
            HardwareInfo instance with current hardware details.
        """
        import os
        import sys

        import torch

        device_type = "cpu"
        gpu_name = None
        gpu_memory_gb = None
        cuda_version = None

        if torch.cuda.is_available():
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = round(props.total_memory / (1024**3), 2)
            cuda_version = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"

        return cls(
            device_type=device_type,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=cuda_version,
            cpu_count=os.cpu_count() or 1,
            torch_version=torch.__version__,
            python_version=sys.version.split()[0],
        )


class ConfigSnapshot(BaseModel):
    """Captures configuration at the time of run execution."""

    config_type: str = Field(..., description="Configuration class name")
    config_values: dict = Field(..., description="Configuration values as dictionary")
    environment_variables: dict[str, str] = Field(
        default_factory=dict, description="Relevant environment variables"
    )

    @classmethod
    def from_config(cls, config: BaseModel, env_prefix: str = "ARANDU_") -> ConfigSnapshot:
        """Create snapshot from a Pydantic BaseSettings/BaseModel.

        Args:
            config: The configuration object to snapshot.
            env_prefix: Prefix for environment variables to capture.

        Returns:
            ConfigSnapshot instance.
        """
        import os

        # Capture relevant environment variables
        env_vars = {key: value for key, value in os.environ.items() if key.startswith(env_prefix)}

        return cls(
            config_type=config.__class__.__name__,
            config_values=config.model_dump(mode="json"),
            environment_variables=env_vars,
        )


class RunMetadata(BaseModel):
    """Complete metadata for a pipeline run.

    Tracks identity, timing, status, execution context, and progress for
    a versioned pipeline run.
    """

    model_config = {"populate_by_name": True}

    # Identity
    run_id: str = Field(..., description="Unique run identifier (YYYYMMDD_HHMMSS_context)")
    pipeline_id: str | None = Field(default=None, description="Pipeline ID grouping related steps")
    pipeline_type: PipelineType = Field(..., description="Type of pipeline executed")

    # Timing
    started_at: datetime = Field(default_factory=_utc_now, description="Run start time (UTC)")
    ended_at: datetime | None = Field(default=None, description="Run end time (UTC)")

    # Status
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current run status")
    error_message: str | None = Field(default=None, description="Error message if failed")

    # Context
    execution: ExecutionEnvironment = Field(..., description="Execution environment details")
    hardware: HardwareInfo = Field(..., description="Hardware information")
    config: ConfigSnapshot = Field(..., description="Configuration snapshot")

    # Progress
    total_items: int = Field(default=0, description="Total items to process")
    completed_items: int = Field(default=0, description="Successfully completed items")
    failed_items: int = Field(default=0, description="Failed items")

    # Paths
    output_directory: str = Field(..., description="Path to run output directory")
    checkpoint_file: str | None = Field(default=None, description="Path to checkpoint file")

    # Version info
    arandu_version: str = Field(..., alias="gtranscriber_version", description="Arandu version")
    schema_version: str = Field(default="1.0", description="Schema version for compatibility")

    # Optional input source
    input_source: str | None = Field(default=None, description="Input source (catalog, directory)")

    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        """Compute run duration in seconds."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @computed_field
    @property
    def success_rate(self) -> float | None:
        """Compute success rate as percentage."""
        total = self.completed_items + self.failed_items
        if total == 0:
            return None
        return round(self.completed_items / total * 100, 2)

    def save(self, path: str | Path) -> None:
        """Save run metadata to JSON file.

        Args:
            path: Path to save the metadata file.
        """
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RunMetadata:
        """Load run metadata from JSON file.

        Args:
            path: Path to the metadata file.

        Returns:
            RunMetadata instance.
        """
        return cls.model_validate_json(Path(path).read_text())
