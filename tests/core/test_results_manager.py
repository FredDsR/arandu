"""Tests for results versioning manager (ID-first layout)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from gtranscriber.core.results_manager import ResultsManager
from gtranscriber.schemas import (
    ConfigSnapshot,
    ExecutionEnvironment,
    HardwareInfo,
    PipelineMetadata,
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

    def test_pipeline_id_field(self) -> None:
        """Test that pipeline_id field is supported."""
        metadata = RunMetadata(
            run_id="test-run",
            pipeline_id="test-pipeline",
            pipeline_type=PipelineType.TRANSCRIPTION,
            execution=ExecutionEnvironment(hostname="test", username="test"),
            hardware=HardwareInfo(
                device_type="cpu", cpu_count=1, torch_version="2.0", python_version="3.13"
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory="/test",
            gtranscriber_version="0.1.0",
        )

        assert metadata.pipeline_id == "test-pipeline"

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

        assert metadata.duration_seconds == 330.0

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


class TestPipelineMetadata:
    """Tests for PipelineMetadata model."""

    def test_creation(self) -> None:
        """Test PipelineMetadata creation."""
        meta = PipelineMetadata(
            pipeline_id="test-id",
            steps_run=["transcription"],
        )

        assert meta.pipeline_id == "test-id"
        assert meta.steps_run == ["transcription"]
        assert meta.schema_version == "2.0"
        assert meta.created_at is not None

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading pipeline metadata."""
        meta = PipelineMetadata(
            pipeline_id="test-id",
            steps_run=["transcription", "qa"],
        )

        save_path = tmp_path / "pipeline.json"
        meta.save(save_path)

        assert save_path.exists()

        loaded = PipelineMetadata.load(save_path)
        assert loaded.pipeline_id == "test-id"
        assert loaded.steps_run == ["transcription", "qa"]


class TestResultsManager:
    """Tests for ResultsManager class (ID-first layout)."""

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

    def test_create_run_id_first_layout(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that create_run creates results/{id}/{step}/outputs/ layout."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata = manager.create_run(TestConfig(), input_source="catalog.csv")

        # Verify ID-first directory structure
        pipeline_id = metadata.pipeline_id
        assert pipeline_id is not None
        assert (tmp_path / pipeline_id / "transcription").exists()
        assert (tmp_path / pipeline_id / "transcription" / "outputs").exists()

        # Verify metadata
        assert metadata.status == RunStatus.IN_PROGRESS
        assert metadata.execution.is_local is True
        assert metadata.input_source == "catalog.csv"

        # Verify run_metadata.json saved in step dir
        metadata_path = tmp_path / pipeline_id / "transcription" / "run_metadata.json"
        assert metadata_path.exists()

    def test_create_run_creates_pipeline_json(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that create_run creates pipeline.json with correct content."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata = manager.create_run(TestConfig())

        pipeline_json = tmp_path / metadata.pipeline_id / "pipeline.json"
        assert pipeline_json.exists()

        pipeline_meta = PipelineMetadata.load(pipeline_json)
        assert pipeline_meta.pipeline_id == metadata.pipeline_id
        assert "transcription" in pipeline_meta.steps_run
        assert pipeline_meta.schema_version == "2.0"

    def test_create_run_with_explicit_id(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that user-provided pipeline_id is used."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION, pipeline_id="my-run")
        metadata = manager.create_run(TestConfig())

        assert metadata.pipeline_id == "my-run"
        assert (tmp_path / "my-run" / "transcription" / "outputs").exists()

    def test_resume_preserves_existing_files(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that re-running a step with the same ID preserves existing files."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        # First run — create outputs and a checkpoint file
        manager1 = ResultsManager(tmp_path, PipelineType.QA, pipeline_id="test-run")
        manager1.create_run(TestConfig())
        (manager1.outputs_dir / "output_001.json").write_text('{"done": true}')
        (manager1.run_dir / "checkpoint.json").write_text('{"last_index": 5}')

        # Second run with same ID (simulating job resumption)
        manager2 = ResultsManager(tmp_path, PipelineType.QA, pipeline_id="test-run")
        manager2.create_run(TestConfig())

        # Existing output and checkpoint files survive
        assert (manager2.outputs_dir / "output_001.json").exists()
        assert (manager2.run_dir / "checkpoint.json").exists()
        assert json.loads((manager2.run_dir / "checkpoint.json").read_text()) == {"last_index": 5}

        # run_metadata.json is still refreshed
        metadata_path = manager2.run_dir / "run_metadata.json"
        assert metadata_path.exists()
        loaded = RunMetadata.load(metadata_path)
        assert loaded.status == RunStatus.IN_PROGRESS

    def test_complete_run_no_symlinks(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that completing a run does NOT create latest/ symlinks."""
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

        # Verify NO latest/ symlink
        assert not (tmp_path / "latest").exists()

        # Verify index updated
        index_path = tmp_path / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index_data = json.load(f)
        assert len(index_data["runs"]) == 1
        assert index_data["runs"][0]["status"] == "completed"
        assert index_data["runs"][0]["pipeline_id"] is not None

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

    def test_resolve_outputs_by_id(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test resolving outputs by pipeline ID and step."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION, pipeline_id="test-run")
        manager.create_run(TestConfig())

        resolved = ResultsManager.resolve_outputs(tmp_path, "test-run", PipelineType.TRANSCRIPTION)
        assert resolved is not None
        assert resolved == tmp_path / "test-run" / "transcription" / "outputs"

    def test_resolve_outputs_nonexistent(self, tmp_path: Path) -> None:
        """Test resolving outputs for nonexistent pipeline ID."""
        resolved = ResultsManager.resolve_outputs(tmp_path, "nonexistent", PipelineType.QA)
        assert resolved is None

    def test_resolve_latest_outputs(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test resolving the latest outputs by scanning pipeline dirs."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata = manager.create_run(TestConfig())
        manager.complete_run(success=True)

        resolved = ResultsManager.resolve_latest_outputs(tmp_path, PipelineType.TRANSCRIPTION)
        assert resolved is not None
        assert resolved == tmp_path / metadata.pipeline_id / "transcription" / "outputs"

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

        latest = ResultsManager.get_latest_run(tmp_path, PipelineType.TRANSCRIPTION)

        assert latest is not None
        assert latest.run_id == original_metadata.run_id

    def test_get_latest_run_no_runs(self, tmp_path: Path) -> None:
        """Test getting latest run when no runs exist."""
        latest = ResultsManager.get_latest_run(tmp_path, PipelineType.TRANSCRIPTION)
        assert latest is None

    def test_pipeline_steps_tracked(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test that running multiple steps under the same ID updates pipeline.json."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        # Run transcription
        mgr1 = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION, pipeline_id="multi-step")
        mgr1.create_run(TestConfig())
        mgr1.complete_run(success=True)

        # Run QA under same pipeline ID
        mgr2 = ResultsManager(tmp_path, PipelineType.QA, pipeline_id="multi-step")
        mgr2.create_run(TestConfig())
        mgr2.complete_run(success=True)

        # pipeline.json should track both steps
        pipeline_json = tmp_path / "multi-step" / "pipeline.json"
        pipeline_meta = PipelineMetadata.load(pipeline_json)
        assert "transcription" in pipeline_meta.steps_run
        assert "qa" in pipeline_meta.steps_run

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

    def test_create_run_slurm(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test creating a run in SLURM environment."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_PARTITION": "grace",
            "SLURMD_NODENAME": "node001",
        }
        mocker.patch.dict(os.environ, slurm_env, clear=False)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        metadata = manager.create_run(TestConfig())

        # Verify SLURM context in pipeline_id
        assert "slurm_grace_12345" in metadata.pipeline_id
        assert metadata.execution.is_slurm is True
        assert metadata.execution.slurm_partition == "grace"

    def test_get_latest_pipeline_id(
        self, tmp_path: Path, mocker: MockerFixture, mock_torch_cpu: MagicMock
    ) -> None:
        """Test getting the most recent pipeline ID."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            model_id: str = "test"

        mocker.patch.dict(os.environ, {}, clear=True)

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION, pipeline_id="first-run")
        manager.create_run(TestConfig())
        manager.complete_run(success=True)

        latest_id = ResultsManager.get_latest_pipeline_id(tmp_path)
        assert latest_id == "first-run"

    def test_get_latest_pipeline_id_empty(self, tmp_path: Path) -> None:
        """Test getting latest pipeline ID when none exist."""
        latest_id = ResultsManager.get_latest_pipeline_id(tmp_path)
        assert latest_id is None


class TestRegisterExternalRun:
    """Tests for register_external_run()."""

    def test_register_external_run_updates_index(self, tmp_path: Path) -> None:
        """Test that register_external_run adds entry to index.json."""
        pipeline_id = "imported-run-001"
        step_dir = tmp_path / pipeline_id / "transcription"
        step_dir.mkdir(parents=True)
        (step_dir / "outputs").mkdir()

        metadata = RunMetadata(
            run_id=pipeline_id,
            pipeline_id=pipeline_id,
            pipeline_type=PipelineType.TRANSCRIPTION,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            status=RunStatus.COMPLETED,
            execution=ExecutionEnvironment(
                is_slurm=True, is_local=False, hostname="node", username="user"
            ),
            hardware=HardwareInfo(
                device_type="cuda",
                cpu_count=1,
                torch_version="2.0",
                python_version="3.13",
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory=str(step_dir),
            checkpoint_file=str(step_dir / "checkpoint.json"),
            gtranscriber_version="0.1.0",
        )

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION, pipeline_id=pipeline_id)
        manager.register_external_run(metadata)

        # Verify index.json was created
        index_path = tmp_path / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index_data = json.load(f)

        assert len(index_data["runs"]) == 1
        assert index_data["runs"][0]["pipeline_id"] == pipeline_id
        assert index_data["runs"][0]["status"] == "completed"

    def test_register_external_run_missing_dir_raises(self, tmp_path: Path) -> None:
        """Test that register_external_run raises ValueError for missing dir."""
        metadata = RunMetadata(
            run_id="missing",
            pipeline_id="missing",
            pipeline_type=PipelineType.TRANSCRIPTION,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status=RunStatus.COMPLETED,
            execution=ExecutionEnvironment(
                is_slurm=False, is_local=True, hostname="h", username="u"
            ),
            hardware=HardwareInfo(
                device_type="cpu",
                cpu_count=1,
                torch_version="2.0",
                python_version="3.13",
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory=str(tmp_path / "nonexistent" / "transcription"),
            checkpoint_file="checkpoint.json",
            gtranscriber_version="0.1.0",
        )

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        with pytest.raises(ValueError, match="Step directory not found"):
            manager.register_external_run(metadata)

    def test_register_sets_run_dir_and_metadata(self, tmp_path: Path) -> None:
        """Test that register_external_run sets run_dir and metadata properties."""
        pipeline_id = "ext-run"
        step_dir = tmp_path / pipeline_id / "transcription"
        step_dir.mkdir(parents=True)

        metadata = RunMetadata(
            run_id=pipeline_id,
            pipeline_id=pipeline_id,
            pipeline_type=PipelineType.TRANSCRIPTION,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status=RunStatus.COMPLETED,
            execution=ExecutionEnvironment(
                is_slurm=False, is_local=True, hostname="h", username="u"
            ),
            hardware=HardwareInfo(
                device_type="cpu",
                cpu_count=1,
                torch_version="2.0",
                python_version="3.13",
            ),
            config=ConfigSnapshot(config_type="Test", config_values={}),
            output_directory=str(step_dir),
            checkpoint_file=str(step_dir / "checkpoint.json"),
            gtranscriber_version="0.1.0",
        )

        manager = ResultsManager(tmp_path, PipelineType.TRANSCRIPTION)
        manager.register_external_run(metadata)

        assert manager.run_dir == step_dir
        assert manager.metadata.pipeline_id == pipeline_id


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


def _make_source_pipeline(
    base_dir: Path,
    pipeline_id: str = "source-pipeline",
    steps: list[str] | None = None,
) -> Path:
    """Create a realistic source pipeline directory for replication tests.

    Args:
        base_dir: Base results directory.
        pipeline_id: Pipeline ID to use.
        steps: Pipeline steps to create. Defaults to ["transcription", "cep"].

    Returns:
        Path to the pipeline directory.
    """
    if steps is None:
        steps = ["transcription", "cep"]

    pipeline_dir = base_dir / pipeline_id
    pipeline_dir.mkdir(parents=True)

    # Write pipeline.json
    PipelineMetadata(
        pipeline_id=pipeline_id,
        steps_run=steps,
    ).save(pipeline_dir / "pipeline.json")

    # Write run_metadata.json + dummy outputs for each step
    for step in steps:
        step_dir = pipeline_dir / step
        outputs_dir = step_dir / "outputs"
        outputs_dir.mkdir(parents=True)

        metadata = RunMetadata(
            run_id=pipeline_id,
            pipeline_id=pipeline_id,
            pipeline_type=PipelineType(step),
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            status=RunStatus.COMPLETED,
            total_items=5,
            completed_items=5,
            failed_items=0,
            execution=ExecutionEnvironment(
                is_slurm=False, is_local=True, hostname="h", username="u"
            ),
            hardware=HardwareInfo(
                device_type="cpu",
                cpu_count=4,
                torch_version="2.0",
                python_version="3.13",
            ),
            config=ConfigSnapshot(config_type="Test", config_values={"key": "val"}),
            output_directory=str(step_dir),
            checkpoint_file=str(step_dir / "checkpoint.json"),
            gtranscriber_version="0.1.0",
        )
        metadata.save(step_dir / "run_metadata.json")

        # Dummy output file
        (outputs_dir / "doc1.json").write_text('{"data": "hello"}')

    return pipeline_dir


class TestReplicatePipeline:
    """Tests for ResultsManager.replicate_pipeline."""

    def test_replicate_pipeline_copies_outputs(self, tmp_path: Path) -> None:
        """Test full replication copies directory structure and output files."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(tmp_path, "src-pipe")

        target_dir = tmp_path / new_id
        assert target_dir.exists()
        assert (target_dir / "pipeline.json").exists()
        assert (target_dir / "transcription" / "outputs" / "doc1.json").exists()
        assert (target_dir / "cep" / "outputs" / "doc1.json").exists()

    def test_replicate_pipeline_explicit_id(self, tmp_path: Path) -> None:
        """Test user-provided target ID is used."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(
            tmp_path, "src-pipe", target_pipeline_id="my-clone"
        )

        assert new_id == "my-clone"
        assert (tmp_path / "my-clone" / "pipeline.json").exists()

    def test_replicate_pipeline_source_not_found(self, tmp_path: Path) -> None:
        """Test ValueError when source pipeline does not exist."""
        with pytest.raises(ValueError, match="Source pipeline not found"):
            ResultsManager.replicate_pipeline(tmp_path, "nonexistent")

    def test_replicate_pipeline_target_exists(self, tmp_path: Path) -> None:
        """Test ValueError when target pipeline already exists."""
        _make_source_pipeline(tmp_path, "src-pipe")
        (tmp_path / "taken-id").mkdir()

        with pytest.raises(ValueError, match="Target pipeline already exists"):
            ResultsManager.replicate_pipeline(tmp_path, "src-pipe", target_pipeline_id="taken-id")

    def test_replicate_updates_pipeline_metadata(self, tmp_path: Path) -> None:
        """Test new pipeline_id, created_at, and replicated_from in pipeline.json."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(
            tmp_path, "src-pipe", target_pipeline_id="clone-1"
        )

        meta = PipelineMetadata.load(tmp_path / new_id / "pipeline.json")
        assert meta.pipeline_id == "clone-1"
        assert meta.replicated_from is not None
        assert meta.replicated_from.source_pipeline_id == "src-pipe"
        assert meta.replicated_from.replicated_at is not None

        # created_at should differ from source
        source_meta = PipelineMetadata.load(tmp_path / "src-pipe" / "pipeline.json")
        assert meta.created_at >= source_meta.created_at

    def test_replicate_updates_run_metadata(self, tmp_path: Path) -> None:
        """Test run_metadata.json gets new pipeline_id, run_id, and paths."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(
            tmp_path, "src-pipe", target_pipeline_id="clone-2"
        )

        for step in ["transcription", "cep"]:
            run_meta = RunMetadata.load(tmp_path / new_id / step / "run_metadata.json")
            assert run_meta.pipeline_id == "clone-2"
            assert run_meta.run_id == "clone-2"
            assert new_id in run_meta.output_directory
            if run_meta.checkpoint_file is not None:
                assert new_id in run_meta.checkpoint_file

    def test_replicate_preserves_run_status(self, tmp_path: Path) -> None:
        """Test that run status stays COMPLETED after replication."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(
            tmp_path, "src-pipe", target_pipeline_id="clone-3"
        )

        for step in ["transcription", "cep"]:
            run_meta = RunMetadata.load(tmp_path / new_id / step / "run_metadata.json")
            assert run_meta.status == RunStatus.COMPLETED

    def test_replicate_registers_in_index(self, tmp_path: Path) -> None:
        """Test that replicated steps appear in index.json."""
        _make_source_pipeline(tmp_path, "src-pipe")

        new_id = ResultsManager.replicate_pipeline(
            tmp_path, "src-pipe", target_pipeline_id="clone-idx"
        )

        index_path = tmp_path / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index_data = json.load(f)

        clone_entries = [r for r in index_data["runs"] if r["pipeline_id"] == new_id]
        assert len(clone_entries) == 2

        step_types = {r["pipeline_type"] for r in clone_entries}
        assert step_types == {"transcription", "cep"}

    def test_replicate_cleans_up_on_partial_failure(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that target directory is removed when post-copy operations fail."""
        _make_source_pipeline(tmp_path, "src-pipe")

        # Make PipelineMetadata.load raise after copytree succeeds
        mocker.patch.object(
            PipelineMetadata,
            "load",
            side_effect=RuntimeError("simulated metadata failure"),
        )

        with pytest.raises(RuntimeError, match="simulated metadata failure"):
            ResultsManager.replicate_pipeline(
                tmp_path, "src-pipe", target_pipeline_id="should-be-cleaned"
            )

        # Target directory should have been cleaned up
        assert not (tmp_path / "should-be-cleaned").exists()
