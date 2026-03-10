"""Shared test helpers for report tests."""

from __future__ import annotations

from typing import Any

from arandu.shared.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineType,
    RunMetadata,
)


def make_run_metadata(
    pipeline_type: PipelineType = PipelineType.TRANSCRIPTION,
    config_values: dict[str, Any] | None = None,
    pipeline_id: str | None = None,
    output_directory: str = "/tmp/test",
) -> RunMetadata:
    """Create a minimal RunMetadata for testing.

    Args:
        pipeline_type: Pipeline type for this run.
        config_values: Configuration values to snapshot.
        pipeline_id: Optional pipeline ID.
        output_directory: Output directory path.

    Returns:
        RunMetadata instance with minimal required fields.
    """
    execution = ExecutionEnvironment(hostname="test-host", username="test-user")
    hardware = HardwareInfo(
        device_type="cpu",
        cpu_count=4,
        torch_version="2.0.0",
        python_version="3.13.0",
    )
    config = ConfigSnapshot(
        config_type="TestConfig",
        config_values=config_values or {},
    )
    return RunMetadata(
        run_id="20240515_120000_test",
        pipeline_id=pipeline_id,
        pipeline_type=pipeline_type,
        execution=execution,
        hardware=hardware,
        config=config,
        output_directory=output_directory,
        arandu_version="0.1.0",
    )
