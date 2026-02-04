"""Tests for results versioning manager."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from gtranscriber.core.results_manager import ResultsManager
from gtranscriber.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineType,
    RunMetadata,
    RunStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def mock_torch_cpu(mocker: MockerFixture) -> MagicMock:
    """Create a mock torch module configured for CPU-only environment."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.__version__ = "2.0.0"
    mocker.patch.dict("sys.modules", {"torch": mock_torch})
    return mock_torch


class TestExecutionEnvironment:
    """Tests for ExecutionEnvironment model."""

    def test_detect_local_environment(self, mocker: MockerFixture) -> None:
        """Test detection of local execution environment."""
        # Clear SLURM environment variables
        mocker.patch.dict(os.environ, {}, clear=True)

        env = ExecutionEnvironment.detect()

        assert env.is_local is True
        assert env.is_slurm is False
        assert env.slurm_job_id is None
        assert env.slurm_partition is None
        assert env.slurm_node is None
        assert env.hostname is not None
        assert env.username is not None

    def test_detect_slurm_environment(self, mocker: MockerFixture) -> None:
        """Test detection of SLURM execution environment."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_PARTITION": "gpu",
            "SLURMD_NODENAME": "node001",
        }
        mocker.patch.dict(os.environ, slurm_env, clear=False)

        env = ExecutionEnvironment.detect()

        assert env.is_slurm is True
        assert env.is_local is False
        assert env.slurm_job_id == "12345"
        assert env.slurm_partition == "gpu"
        assert env.slurm_node == "node001"


class TestHardwareInfo:
    """Tests for HardwareInfo model."""

    def test_capture_cpu_only(self, mocker: MockerFixture) -> None:
        """Test hardware capture on CPU-only system."""
        # Mock torch module which is imported inside the capture() method
        mock_torch = mocker.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.__version__ = "2.0.0"
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        hw = HardwareInfo.capture()

        assert hw.device_type == "cpu"
        assert hw.gpu_name is None
        assert hw.gpu_memory_gb is None
        assert hw.cuda_version is None
        assert hw.cpu_count >= 1
        assert hw.torch_version == "2.0.0"
        assert hw.python_version is not None

    def test_capture_with_cuda(self, mocker: MockerFixture) -> None:
        """Test hardware capture on system with CUDA GPU."""
        mock_torch = mocker.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_props = mocker.MagicMock()
        mock_props.total_memory = 40 * 1024**3  # 40 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.version.cuda = "12.1"
        mock_torch.__version__ = "2.0.0"
        mocker.patch.dict("sys.modules", {"torch": mock_torch})

        hw = HardwareInfo.capture()

        assert hw.device_type == "cuda"
        assert hw.gpu_name == "NVIDIA A100"
        assert hw.gpu_memory_gb == 40.0
        assert hw.cuda_version == "12.1"


class TestConfigSnapshot:
    """Tests for ConfigSnapshot model."""

    def test_from_config(self, mocker: MockerFixture) -> None:
        """Test creating snapshot from config."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test-model"
            workers: int = 4

        # Set up environment variables
        mocker.patch.dict(
            os.environ,
            {"GTRANSCRIBER_MODEL_ID": "env-model", "OTHER_VAR": "ignored"},
            clear=False,
        )

        config = TestConfig()
        snapshot = ConfigSnapshot.from_config(config, env_prefix="GTRANSCRIBER_")

        assert snapshot.config_type == "TestConfig"
        assert snapshot.config_values["model_id"] == "test-model"
        assert snapshot.config_values["workers"] == 4
        assert "GTRANSCRIBER_MODEL_ID" in snapshot.environment_variables
        assert "OTHER_VAR" not in snapshot.environment_variables


class TestRunMetadata:
    """Tests for RunMetadata model."""

    def test_creation(self) -> None:
        """Test RunMetadata creation with required fields."""
        metadata = RunMetadata(
            run_id="20260204_143052_local",
            pipeline_type=PipelineType.TRANSCRIPTION,
            execution=ExecutionEnvironment(
                hostname="testhost",
                username="testuser",
            ),
            hardware=HardwareInfo(
                device_type="cpu",
                cpu_count=8,
                torch_version="2.0.0",
                python_version="3.13.0",
            ),
            config=ConfigSnapshot(
                config_type="TestConfig",
                config_values={},
            ),
            output_directory="/path/to/output",
            gtranscriber_version="0.1.0",
        )

        assert metadata.run_id == "20260204_143052_local"
        assert metadata.pipeline_type == PipelineType.TRANSCRIPTION
        assert metadata.status == RunStatus.PENDING
        assert metadata.duration_seconds is None

    def test_duration_calculation(self) -> None:
        """Test duration_seconds computed field."""
        started = datetime(2026, 2, 4, 14, 30, 0)
        ended = datetime(2026, 2, 4, 14, 35, 30)

        metadata = RunMetadata(
            run_id="test_run",
            pipeline_type=PipelineType.QA,
            started_at=started,
            ended_at=ended,
            execution=ExecutionEnvironment(hostname="test", username="test"),
            hardware=HardwareInfo(
                device_type="cpu", cpu_count=1, torch_version="2.0", python_version="3.13"
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory="/test",
            gtranscriber_version="0.1.0",
        )

        assert metadata.duration_seconds == 330.0  # 5 minutes 30 seconds

    def test_success_rate_calculation(self) -> None:
        """Test success_rate computed field."""
        metadata = RunMetadata(
            run_id="test_run",
            pipeline_type=PipelineType.TRANSCRIPTION,
            completed_items=8,
            failed_items=2,
            total_items=10,
            execution=ExecutionEnvironment(hostname="test", username="test"),
            hardware=HardwareInfo(
                device_type="cpu", cpu_count=1, torch_version="2.0", python_version="3.13"
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory="/test",
            gtranscriber_version="0.1.0",
        )

        assert metadata.success_rate == 80.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading metadata."""
        metadata = RunMetadata(
            run_id="test_save_load",
            pipeline_type=PipelineType.KG,
            execution=ExecutionEnvironment(hostname="test", username="test"),
            hardware=HardwareInfo(
                device_type="cpu", cpu_count=1, torch_version="2.0", python_version="3.13"
            ),
            config=ConfigSnapshot(config_type="Test", config_values={"key": "value"}),
            output_directory="/test",
            gtranscriber_version="0.1.0",
        )

        save_path = tmp_path / "metadata.json"
        metadata.save(save_path)

        assert save_path.exists()

        loaded = RunMetadata.load(save_path)
        assert loaded.run_id == "test_save_load"
        assert loaded.pipeline_type == PipelineType.KG
        assert loaded.config.config_values["key"] == "value"


class TestResultsManager:
    """Tests for ResultsManager class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test ResultsManager initialization."""
        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)

        assert manager.base_dir == tmp_path.resolve()
        assert manager.pipeline_type == PipelineType.TRANSCRIPTION

    def test_run_dir_before_create(self, tmp_path: Path) -> None:
        """Test that accessing run_dir before create_run raises error."""
        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)

        with pytest.raises(RuntimeError, match=r"create_run.*must be called"):
            _ = manager.run_dir

    def test_create_run_local(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test creating a run in local environment."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        # Mock environment as local
        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        config = TestConfig()
        metadata = manager.create_run(config, input_source="catalog.csv")

        # Verify run directory created
        assert manager.run_dir.exists()
        assert manager.outputs_dir.exists()

        # Verify metadata
        assert metadata.status == RunStatus.IN_PROGRESS
        assert metadata.execution.is_local is True
        assert "local" in metadata.run_id
        assert metadata.input_source == "catalog.csv"

        # Verify metadata file saved
        metadata_path = manager.run_dir / "run_metadata.json"
        assert metadata_path.exists()

    def test_create_run_slurm(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test creating a run in SLURM environment."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        # Mock SLURM environment
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_PARTITION": "grace",
            "SLURMD_NODENAME": "node001",
        }
        mocker.patch.dict(os.environ, slurm_env, clear=False)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata = manager.create_run(TestConfig())

        # Verify SLURM context in run_id
        assert "slurm_grace_12345" in metadata.run_id
        assert metadata.execution.is_slurm is True
        assert metadata.execution.slurm_partition == "grace"

    def test_update_progress(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test updating progress during a run."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.QA)
        manager.create_run(TestConfig())

        manager.update_progress(completed=5, failed=1, total=10)

        assert manager.metadata.completed_items == 5
        assert manager.metadata.failed_items == 1
        assert manager.metadata.total_items == 10

        # Verify metadata file updated
        metadata_path = manager.run_dir / "run_metadata.json"
        loaded = RunMetadata.load(metadata_path)
        assert loaded.completed_items == 5

    def test_complete_run_success(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test completing a run successfully."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        manager.create_run(TestConfig())
        manager.update_progress(completed=10, failed=0, total=10)
        manager.complete_run(success=True)

        # Verify status
        assert manager.metadata.status == RunStatus.COMPLETED
        assert manager.metadata.ended_at is not None
        assert manager.metadata.error_message is None

        # Verify symlink created
        latest_symlink = tmp_path / "latest" / "transcription"
        assert latest_symlink.is_symlink()

        # Verify index updated
        index_path = tmp_path / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index_data = json.load(f)
        assert len(index_data["runs"]) == 1
        assert index_data["runs"][0]["status"] == "completed"

    def test_complete_run_failure(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test completing a run with failure."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        manager.create_run(TestConfig())
        manager.complete_run(success=False, error="Out of memory")

        assert manager.metadata.status == RunStatus.FAILED
        assert manager.metadata.error_message == "Out of memory"

    def test_get_latest_run(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test getting the latest run for a pipeline."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        original_metadata = manager.create_run(TestConfig())
        manager.complete_run(success=True)

        # Get latest run
        latest = ResultsManager.get_latest_run(tmp_path, PipelineType.TRANSCRIPTION)

        assert latest is not None
        assert latest.run_id == original_metadata.run_id

    def test_get_latest_run_no_runs(self, tmp_path: Path) -> None:
        """Test getting latest run when no runs exist."""
        latest = ResultsManager.get_latest_run(tmp_path, PipelineType.TRANSCRIPTION)
        assert latest is None

    def test_list_runs(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test listing runs."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        # Create two runs
        manager1 = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        manager1.create_run(TestConfig())
        manager1.complete_run(success=True)

        # Add small delay to ensure different timestamps
        import time

        time.sleep(0.01)

        manager2 = ResultsManager(tmp_path, PipelineType.QA)
        manager2.create_run(TestConfig())
        manager2.complete_run(success=True)

        # List all runs
        all_runs = ResultsManager.list_runs(tmp_path)
        assert len(all_runs) == 2

        # List transcription runs only
        transcription_runs = ResultsManager.list_runs(tmp_path, PipelineType.TRANSCRIPTION)
        assert len(transcription_runs) == 1
        assert transcription_runs[0]["pipeline_type"] == "transcription"

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        """Test listing runs when none exist."""
        runs = ResultsManager.list_runs(tmp_path)
        assert runs == []

    def test_symlink_update_replaces_existing(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that updating symlink replaces existing one."""
        import time

        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        # Create first run
        manager1 = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        manager1.create_run(TestConfig())
        manager1.complete_run(success=True)

        latest_symlink = tmp_path / "latest" / "transcription"
        first_target = latest_symlink.resolve()

        # Wait to ensure different timestamp
        time.sleep(1.1)

        # Create second run
        manager2 = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata2 = manager2.create_run(TestConfig())
        manager2.complete_run(success=True)

        second_target = latest_symlink.resolve()

        # Symlink should point to second run
        assert first_target != second_target
        assert metadata2.run_id in str(second_target)


class TestPipelineType:
    """Tests for PipelineType enum."""

    def test_all_pipeline_types(self) -> None:
        """Test all pipeline types are defined."""
        expected = {"transcription", "qa", "cep", "kg", "evaluation"}
        actual = {p.value for p in PipelineType}
        assert actual == expected


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_all_status_values(self) -> None:
        """Test all status values are defined."""
        expected = {"pending", "in_progress", "completed", "failed", "cancelled"}
        actual = {s.value for s in RunStatus}
        assert actual == expected
