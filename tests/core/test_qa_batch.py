"""Tests for QA batch processing module."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from gtranscriber.config import CEPConfig, QAConfig
from gtranscriber.core.qa_batch import (
    QAGenerationTask,
    _init_cep_worker,
    _init_qa_worker,
    generate_cep_qa_for_transcription,
    generate_qa_for_transcription,
    load_transcription_tasks,
    run_batch_cep_generation,
    run_batch_qa_generation,
)
from gtranscriber.core.results_manager import ResultsManager


class _ThreadPoolCompat(ThreadPoolExecutor):
    """ThreadPoolExecutor that ignores mp_context for test compatibility."""

    def __init__(self, *args: object, mp_context: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


def create_test_enriched_data(
    gdrive_id: str = "test123", name: str = "test.mp3", transcription_text: str = "Test"
) -> dict:
    """Create a complete EnrichedRecord test data dictionary.

    Args:
        gdrive_id: Google Drive ID.
        name: Filename.
        transcription_text: Transcription text.

    Returns:
        Complete EnrichedRecord data dictionary.
    """
    return {
        "gdrive_id": gdrive_id,
        "name": name,
        "mimeType": "audio/mpeg",
        "parents": ["folder_id"],
        "webContentLink": "https://drive.google.com/test",
        "transcription_text": transcription_text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "openai/whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 30.0,
        "transcription_status": "completed",
    }


class TestQAGenerationTask:
    """Tests for QAGenerationTask dataclass."""

    def test_task_creation(self, tmp_path: Path) -> None:
        """Test creating QAGenerationTask with valid fields."""
        transcription_file = tmp_path / "test_transcription.json"
        output_file = tmp_path / "test_qa.json"

        task = QAGenerationTask(
            transcription_file=transcription_file,
            gdrive_id="test123",
            filename="test.mp3",
            output_file=output_file,
        )

        assert task.transcription_file == transcription_file
        assert task.gdrive_id == "test123"
        assert task.filename == "test.mp3"
        assert task.output_file == output_file

    def test_task_field_accessibility(self, tmp_path: Path) -> None:
        """Test that all fields are accessible."""
        task = QAGenerationTask(
            transcription_file=tmp_path / "in.json",
            gdrive_id="abc",
            filename="file.mp3",
            output_file=tmp_path / "out.json",
        )

        # Access all fields
        assert isinstance(task.transcription_file, Path)
        assert isinstance(task.gdrive_id, str)
        assert isinstance(task.filename, str)
        assert isinstance(task.output_file, Path)


class TestLoadTranscriptionTasks:
    """Tests for load_transcription_tasks function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test that empty directory returns empty list."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        tasks = load_transcription_tasks(input_dir, output_dir)

        assert len(tasks) == 0

    def test_valid_transcription_files(self, tmp_path: Path) -> None:
        """Test discovering valid transcription files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create valid transcription files
        file1 = input_dir / "file1_transcription.json"
        file1.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="id1",
                    name="interview1.mp3",
                    transcription_text="This is a test transcription.",
                )
            )
        )

        file2 = input_dir / "file2_transcription.json"
        file2.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="id2",
                    name="interview2.mp3",
                    transcription_text="Another transcription.",
                )
            )
        )

        tasks = load_transcription_tasks(input_dir, output_dir)

        assert len(tasks) == 2
        assert all(isinstance(task, QAGenerationTask) for task in tasks)

    def test_mixed_valid_invalid_files(self, tmp_path: Path) -> None:
        """Test that invalid files are skipped with warnings."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Valid file
        valid = input_dir / "valid_transcription.json"
        valid.write_text(json.dumps(create_test_enriched_data(gdrive_id="id1", name="test.mp3")))

        # Invalid JSON file
        invalid = input_dir / "invalid_transcription.json"
        invalid.write_text("not valid json {")

        # Non-transcription file (should be ignored by glob)
        other = input_dir / "other_file.json"
        other.write_text(json.dumps({"data": "test"}))

        tasks = load_transcription_tasks(input_dir, output_dir)

        # Should only get the valid transcription file
        assert len(tasks) == 1
        assert tasks[0].gdrive_id == "id1"

    def test_invalid_json_files(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of corrupt JSON files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        corrupt = input_dir / "corrupt_transcription.json"
        corrupt.write_text("{ corrupt json")

        tasks = load_transcription_tasks(input_dir, output_dir)

        assert len(tasks) == 0
        assert "Skipping invalid file" in caplog.text

    def test_gdrive_id_extraction(self, tmp_path: Path) -> None:
        """Test that gdrive_id is correctly extracted from JSON."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="extracted_id_123",
                    name="test.mp3",
                )
            )
        )

        tasks = load_transcription_tasks(input_dir, output_dir)

        assert len(tasks) == 1
        assert tasks[0].gdrive_id == "extracted_id_123"

    def test_output_filename_generation(self, tmp_path: Path) -> None:
        """Test that output filename is {gdrive_id}_qa.json."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "anything_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="myid",
                    name="original.mp3",
                )
            )
        )

        tasks = load_transcription_tasks(input_dir, output_dir)

        assert len(tasks) == 1
        assert tasks[0].output_file == output_dir / "myid_qa.json"


class TestWorkerFunctions:
    """Tests for worker initialization and execution."""

    def test_generate_qa_for_transcription_success(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test successful QA generation for a transcription."""
        # Mock the LLM client
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create input transcription file
        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "gdrive_id": "test123",
                    "name": "test.mp3",
                    "mimeType": "audio/mpeg",
                    "parents": ["folder_id"],
                    "webContentLink": "https://drive.google.com/test",
                    "transcription_text": (
                        "This is a test transcription about climate change in Brazil. " * 10
                    ),
                    "detected_language": "pt",
                    "language_probability": 0.95,
                    "model_id": "openai/whisper-large-v3",
                    "compute_device": "cpu",
                    "processing_duration_sec": 30.0,
                    "transcription_status": "completed",
                }
            )
        )

        output_file = tmp_path / "test_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="test123",
            filename="test.mp3",
            output_file=output_file,
        )

        config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        ).model_dump()

        gdrive_id, success, message = generate_qa_for_transcription(task, config_dict)

        assert gdrive_id == "test123"
        assert success is True
        assert message == "Success"
        assert output_file.exists()

    def test_generate_qa_for_transcription_validation_error(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test handling of validation errors (e.g., too short transcription)."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Create input with too-short transcription
        input_file = tmp_path / "short_transcription.json"
        input_file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="short123",
                    name="short.mp3",
                    transcription_text="Short",  # Too short
                )
            )
        )

        output_file = tmp_path / "short_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="short123",
            filename="short.mp3",
            output_file=output_file,
        )

        config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        ).model_dump()

        gdrive_id, success, message = generate_qa_for_transcription(task, config_dict)

        assert gdrive_id == "short123"
        assert success is False
        assert "short" in message.lower()

    def test_generate_qa_for_transcription_generic_exception(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test handling of generic exceptions."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Create input file that will cause exception (file doesn't exist)
        input_file = tmp_path / "nonexistent.json"
        output_file = tmp_path / "output_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="test123",
            filename="test.mp3",
            output_file=output_file,
        )

        config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        ).model_dump()

        gdrive_id, success, message = generate_qa_for_transcription(task, config_dict)

        assert gdrive_id == "test123"
        assert success is False
        assert len(message) > 0

    def test_generate_qa_for_transcription_creates_output_dir(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "gdrive_id": "test123",
                    "name": "test.mp3",
                    "mimeType": "audio/mpeg",
                    "parents": ["folder_id"],
                    "webContentLink": "https://drive.google.com/test",
                    "transcription_text": "This is a test transcription. " * 20,
                    "detected_language": "pt",
                    "language_probability": 0.95,
                    "model_id": "openai/whisper-large-v3",
                    "compute_device": "cpu",
                    "processing_duration_sec": 30.0,
                    "transcription_status": "completed",
                }
            )
        )

        # Output in nested directory that doesn't exist
        output_file = tmp_path / "nested" / "dir" / "test_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="test123",
            filename="test.mp3",
            output_file=output_file,
        )

        config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        ).model_dump()

        _gdrive_id, success, _message = generate_qa_for_transcription(task, config_dict)

        assert success is True
        assert output_file.parent.exists()
        assert output_file.exists()


class TestRunBatchQAGeneration:
    """Tests for run_batch_qa_generation orchestration."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for QA batch tests."""
        self.results_dir = (tmp_path / "results").resolve()

        mock_rc = mocker.patch("gtranscriber.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")
        mocker.patch("gtranscriber.core.qa_batch.ProcessPoolExecutor", _ThreadPoolCompat)

        self.run_dir = self.results_dir / "qa" / "test_run"
        self.outputs_dir = self.run_dir / "outputs"
        self.versioned_checkpoint = self.run_dir / "qa_checkpoint.json"

    def test_output_directory_creation(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that output directory is created."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        assert self.outputs_dir.exists()

    def test_no_remaining_tasks_early_exit(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test early exit when all files already completed."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create transcription file
        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="id1",
                    name="test.mp3",
                )
            )
        )

        # Pre-create versioned run dir with checkpoint already completed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_checkpoint.write_text(
            json.dumps(
                {
                    "total_files": 1,
                    "completed_files": ["id1"],
                    "failed_files": {},
                }
            )
        )

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        assert "already processed" in caplog.text.lower() or "all files" in caplog.text.lower()

    def test_sequential_processing_single_worker(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test sequential processing with single worker."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create test files
        for i in range(2):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"id{i}",
                        name=f"test{i}.mp3",
                        transcription_text="Test transcription text. " * 20,
                    )
                )
            )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Check outputs were created in versioned directory
        assert (self.outputs_dir / "id0_qa.json").exists()
        assert (self.outputs_dir / "id1_qa.json").exists()

    def test_sequential_processing_with_failures(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test sequential processing with some failures."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create one valid and one invalid file
        valid = input_dir / "valid_transcription.json"
        valid.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="valid",
                    name="valid.mp3",
                    transcription_text="Valid transcription text. " * 20,
                )
            )
        )

        invalid = input_dir / "invalid_transcription.json"
        invalid.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="invalid",
                    name="invalid.mp3",
                    transcription_text="Short",  # Too short
                )
            )
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        )

        # Mock generate to succeed for valid, fail for invalid is automatic
        mock_gen = mocker.patch("gtranscriber.core.qa_batch.generate_qa_for_transcription")
        mock_gen.side_effect = [
            ("valid", True, "Success"),
            ("invalid", False, "Too short"),
        ]

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Check versioned checkpoint records failures
        assert self.versioned_checkpoint.exists()

    def test_checkpoint_updated_on_success(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that checkpoint is updated after successful processing."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="test123",
                    name="test.mp3",
                    transcription_text="Test transcription text. " * 20,
                )
            )
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Check versioned checkpoint
        assert self.versioned_checkpoint.exists()

        checkpoint_data = json.loads(self.versioned_checkpoint.read_text())
        assert "test123" in checkpoint_data["completed_files"]

    def test_checkpoint_updated_on_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that checkpoint records failures."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="fail123",
                    name="test.mp3",
                    transcription_text="Short",  # Too short
                )
            )
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        )

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        checkpoint_data = json.loads(self.versioned_checkpoint.read_text())
        assert "fail123" in checkpoint_data["failed_files"]

    def test_parallel_processing_multiple_workers(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test parallel processing with multiple workers."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create multiple test files
        for i in range(3):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"id{i}",
                        name=f"test{i}.mp3",
                        transcription_text="Test transcription text. " * 20,
                    )
                )
            )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=2)

        # Check all outputs were created in versioned directory
        assert (self.outputs_dir / "id0_qa.json").exists()
        assert (self.outputs_dir / "id1_qa.json").exists()
        assert (self.outputs_dir / "id2_qa.json").exists()

    def test_worker_count_limiting(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that worker count doesn't cause issues with few tasks."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "Test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create only 2 files
        for i in range(2):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"id{i}",
                        name=f"test{i}.mp3",
                        transcription_text="Test. " * 20,
                    )
                )
            )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )

        # Use sequential processing to avoid multiprocessing pickle issues with mocks
        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Verify all output files were created in versioned directory
        assert (self.outputs_dir / "id0_qa.json").exists()
        assert (self.outputs_dir / "id1_qa.json").exists()

    def test_progress_reporting(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that progress is logged during processing."""
        caplog.set_level("INFO")
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="test123",
                    name="test.mp3",
                    transcription_text="Test transcription. " * 20,
                )
            )
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        assert "Progress:" in caplog.text or "Completed:" in caplog.text

    def test_final_summary_with_failures(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that final summary includes failure information."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create file that will fail
        file = input_dir / "fail_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="fail123",
                    name="fail.mp3",
                    transcription_text="Too short",
                )
            )
        )

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Check summary mentions failures
        assert "Failed:" in caplog.text or "failed" in caplog.text.lower()


class TestInitQAWorker:
    """Tests for _init_qa_worker initialization function."""

    def test_init_qa_worker_ollama_provider(self, mocker: MockerFixture) -> None:
        """Test worker initialization with Ollama provider."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_qa_generator = None

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            ollama_url="http://localhost:11434/v1",
        )
        config_dict = config.model_dump()

        _init_qa_worker("ollama", "llama3.1:8b", config_dict)

        assert qa_batch_module._worker_qa_generator is not None

    def test_init_qa_worker_custom_base_url(self, mocker: MockerFixture) -> None:
        """Test worker initialization with custom base URL."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_qa_generator = None

        config = QAConfig(
            provider="custom",
            model_id="custom-model",
            base_url="http://custom-llm:8080/v1",
        )
        config_dict = config.model_dump()

        _init_qa_worker("custom", "custom-model", config_dict)

        assert qa_batch_module._worker_qa_generator is not None
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "http://custom-llm:8080/v1"

    def test_init_qa_worker_openai_provider(self, mocker: MockerFixture) -> None:
        """Test worker initialization with OpenAI provider."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_qa_generator = None

        config = QAConfig(
            provider="openai",
            model_id="gpt-4",
        )
        config_dict = config.model_dump()

        _init_qa_worker("openai", "gpt-4", config_dict)

        assert qa_batch_module._worker_qa_generator is not None


class TestWorkerCountLogging:
    """Tests for worker count > CPU count logging paths."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for worker count tests."""
        mock_rc = mocker.patch("gtranscriber.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")
        mocker.patch("gtranscriber.core.qa_batch.ProcessPoolExecutor", _ThreadPoolCompat)

    def test_workers_greater_than_cpu_count_logs_info(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that using more workers than CPUs logs info message."""
        caplog.set_level("INFO")

        # Mock cpu_count to return small number
        mocker.patch("gtranscriber.core.qa_batch.mp.cpu_count", return_value=2)

        # Mock generate to succeed
        mock_gen = mocker.patch("gtranscriber.core.qa_batch.generate_qa_for_transcription")
        mock_gen.return_value = ("id1", True, "Success")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create 5 test files (more than mocked cpu_count)
        for i in range(5):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"id{i}",
                        name=f"test{i}.mp3",
                    )
                )
            )

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        # Use 4 workers (more than mocked cpu_count of 2)
        run_batch_qa_generation(input_dir, output_dir, config, num_workers=4)

        # Check that the info log about workers > CPU count was emitted
        assert "more than" in caplog.text.lower() and "cpu" in caplog.text.lower()


class TestFinalSummaryPaths:
    """Tests for final summary edge cases."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for summary tests."""
        mock_rc = mocker.patch("gtranscriber.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")

        self.run_dir = (tmp_path / "results").resolve() / "qa" / "test_run"
        self.versioned_checkpoint = self.run_dir / "qa_checkpoint.json"

    def test_final_summary_zero_total_files(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test final summary when total is zero (N/A case)."""
        caplog.set_level("INFO")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Pre-create versioned run dir with empty checkpoint
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_checkpoint.write_text(
            json.dumps(
                {
                    "total_files": 0,
                    "completed_files": [],
                    "failed_files": {},
                }
            )
        )

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        # Should exit early with "No tasks to process"
        assert "no tasks" in caplog.text.lower()


class TestParallelProcessingEdgeCases:
    """Tests for parallel processing edge cases and exception handling."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for edge case tests."""
        mock_rc = mocker.patch("gtranscriber.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")

    def test_parallel_processing_with_future_exception(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test parallel processing handles future.result() exceptions."""
        caplog.set_level("INFO")

        # We need to simulate the parallel execution path
        # Since multiprocessing doesn't work well with mocks, we mock the ProcessPoolExecutor

        mock_executor_class = mocker.patch("gtranscriber.core.qa_batch.ProcessPoolExecutor")
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)

        # Create futures that raise exceptions when getting result
        mock_future1 = Mock()
        mock_future1.result.side_effect = RuntimeError("Worker crashed")

        mock_executor.submit.return_value = mock_future1

        # Mock as_completed to return our futures
        mocker.patch(
            "gtranscriber.core.qa_batch.as_completed",
            return_value=iter([mock_future1]),
        )

        mock_executor_class.return_value = mock_executor

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create test files
        for i in range(3):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"id{i}",
                        name=f"test{i}.mp3",
                    )
                )
            )

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        # Use 2+ workers to trigger parallel path
        run_batch_qa_generation(input_dir, output_dir, config, num_workers=2)

        # Check that exception was logged
        assert "exception" in caplog.text.lower() or "failed" in caplog.text.lower()


class TestInitCEPWorker:
    """Tests for _init_cep_worker initialization function."""

    def test_init_cep_worker_without_validation(self, mocker: MockerFixture) -> None:
        """Test CEP worker initialization without validation enabled."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock CEPQAGenerator
        mock_cep_generator_class = mocker.patch("gtranscriber.core.cep.CEPQAGenerator")
        mock_cep_generator = Mock()
        mock_cep_generator_class.return_value = mock_cep_generator

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        qa_config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        )
        cep_config = CEPConfig(
            enable_validation=False,
        )

        _init_cep_worker(
            provider="ollama",
            model_id="llama3.1:8b",
            qa_config_dict=qa_config.model_dump(),
            cep_config_dict=cep_config.model_dump(),
            validator_provider=None,
            validator_model_id=None,
        )

        assert qa_batch_module._worker_cep_generator is not None
        # Should be called with validator_client=None
        call_kwargs = mock_cep_generator_class.call_args.kwargs
        assert call_kwargs["validator_client"] is None

    def test_init_cep_worker_with_validation(self, mocker: MockerFixture) -> None:
        """Test CEP worker initialization with validation enabled."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock CEPQAGenerator
        mock_cep_generator_class = mocker.patch("gtranscriber.core.cep.CEPQAGenerator")
        mock_cep_generator = Mock()
        mock_cep_generator_class.return_value = mock_cep_generator

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        qa_config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
        )
        cep_config = CEPConfig(
            enable_validation=True,
            validator_provider="ollama",
            validator_model_id="llama3.1:8b",
        )

        _init_cep_worker(
            provider="ollama",
            model_id="llama3.1:8b",
            qa_config_dict=qa_config.model_dump(),
            cep_config_dict=cep_config.model_dump(),
            validator_provider="ollama",
            validator_model_id="llama3.1:8b",
        )

        assert qa_batch_module._worker_cep_generator is not None
        # Should be called with a validator_client
        call_kwargs = mock_cep_generator_class.call_args.kwargs
        assert call_kwargs["validator_client"] is not None


class TestGenerateCEPQAForTranscription:
    """Tests for generate_cep_qa_for_transcription function."""

    def test_generate_cep_qa_success(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test successful CEP QA generation."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test transcription", "confidence": 0.9,
             "bloom_level": "understand", "reasoning_trace": "Direct recall"}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "gdrive_id": "cep_test123",
                    "name": "test.mp3",
                    "mimeType": "audio/mpeg",
                    "parents": ["folder_id"],
                    "webContentLink": "https://drive.google.com/test",
                    "transcription_text": "This is a test transcription about climate. " * 20,
                    "detected_language": "pt",
                    "language_probability": 0.95,
                    "model_id": "openai/whisper-large-v3",
                    "compute_device": "cpu",
                    "processing_duration_sec": 30.0,
                    "transcription_status": "completed",
                }
            )
        )

        output_file = tmp_path / "cep_test_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="cep_test123",
            filename="test.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        ).model_dump()

        cep_config_dict = CEPConfig(
            enable_validation=False,
        ).model_dump()

        gdrive_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert gdrive_id == "cep_test123"
        assert success is True
        assert message == "Success"
        assert output_file.exists()

    def test_generate_cep_qa_validation_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test CEP QA generation with validation error (too short transcription)."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "short_transcription.json"
        input_file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="short_cep",
                    name="short.mp3",
                    transcription_text="Short",
                )
            )
        )

        output_file = tmp_path / "short_cep_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="short_cep",
            filename="short.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(provider="ollama", model_id="llama3.1:8b").model_dump()
        cep_config_dict = CEPConfig(enable_validation=False).model_dump()

        gdrive_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert gdrive_id == "short_cep"
        assert success is False
        assert "short" in message.lower() or len(message) > 0

    def test_generate_cep_qa_generic_exception(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test CEP QA generation with generic exception (file not found)."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        # Non-existent file
        input_file = tmp_path / "nonexistent.json"
        output_file = tmp_path / "output_cep_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="test_cep",
            filename="test.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(provider="ollama", model_id="llama3.1:8b").model_dump()
        cep_config_dict = CEPConfig(enable_validation=False).model_dump()

        gdrive_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert gdrive_id == "test_cep"
        assert success is False
        assert len(message) > 0

    def test_generate_cep_qa_with_validation_enabled(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CEP QA generation with validation enabled."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What is this about?", "answer": "test transcription",
             "confidence": 0.9, "bloom_level": "understand",
             "reasoning_trace": "Direct recall from text"}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Reset global state
        import gtranscriber.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "gdrive_id": "cep_validated",
                    "name": "test.mp3",
                    "mimeType": "audio/mpeg",
                    "parents": ["folder_id"],
                    "webContentLink": "https://drive.google.com/test",
                    "transcription_text": (
                        "This is a long transcription about climate events. " * 30
                    ),
                    "detected_language": "pt",
                    "language_probability": 0.95,
                    "model_id": "openai/whisper-large-v3",
                    "compute_device": "cpu",
                    "processing_duration_sec": 30.0,
                    "transcription_status": "completed",
                }
            )
        )

        output_file = tmp_path / "cep_validated_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            gdrive_id="cep_validated",
            filename="test.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        ).model_dump()

        cep_config_dict = CEPConfig(
            enable_validation=True,
            validator_provider="ollama",
            validator_model_id="llama3.1:8b",
        ).model_dump()

        gdrive_id, success, _ = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert gdrive_id == "cep_validated"
        assert success is True
        assert output_file.exists()


class TestRunBatchCEPGeneration:
    """Tests for run_batch_cep_generation function."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for CEP batch tests."""
        self.results_dir = (tmp_path / "results").resolve()

        mock_rc = mocker.patch("gtranscriber.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_run_id", return_value="test_run")
        mocker.patch("gtranscriber.core.qa_batch.ProcessPoolExecutor", _ThreadPoolCompat)

        self.run_dir = self.results_dir / "cep" / "test_run"
        self.outputs_dir = self.run_dir / "outputs"
        self.versioned_checkpoint = self.run_dir / "cep_checkpoint.json"

    def test_run_batch_cep_creates_output_dir(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that output directory is created for CEP batch."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        assert self.outputs_dir.exists()

    def test_run_batch_cep_no_tasks_early_exit(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test early exit when no transcription files found for CEP."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        assert "no tasks" in caplog.text.lower()

    def test_run_batch_cep_all_completed_early_exit(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test early exit when all CEP files already completed."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create transcription file
        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="cep_id1",
                    name="test.mp3",
                )
            )
        )

        # Pre-create versioned run dir with checkpoint already completed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_checkpoint.write_text(
            json.dumps(
                {
                    "total_files": 1,
                    "completed_files": ["cep_id1"],
                    "failed_files": {},
                }
            )
        )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        assert "already processed" in caplog.text.lower() or "all files" in caplog.text.lower()

    def test_run_batch_cep_sequential_processing(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CEP sequential processing with single worker."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test", "confidence": 0.9,
             "bloom_level": "understand", "reasoning_trace": "Direct recall"}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create test files
        for i in range(2):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"cep_id{i}",
                        name=f"test{i}.mp3",
                        transcription_text="Test transcription text. " * 20,
                    )
                )
            )

        qa_config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Check outputs were created in versioned directory
        assert (self.outputs_dir / "cep_id0_cep_qa.json").exists()
        assert (self.outputs_dir / "cep_id1_cep_qa.json").exists()

    def test_run_batch_cep_sequential_with_failures(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CEP sequential processing with failures."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Mock generate to succeed for one, fail for another
        mock_gen = mocker.patch("gtranscriber.core.qa_batch.generate_cep_qa_for_transcription")
        mock_gen.side_effect = [
            ("valid_cep", True, "Success"),
            ("invalid_cep", False, "Too short"),
        ]

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create test files
        for i in range(2):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"{'valid' if i == 0 else 'invalid'}_cep",
                        name=f"test{i}.mp3",
                    )
                )
            )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Check versioned checkpoint records failure
        checkpoint_data = json.loads(self.versioned_checkpoint.read_text())
        assert "invalid_cep" in checkpoint_data["failed_files"]

    def test_run_batch_cep_final_summary_logged(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CEP final summary is logged."""
        caplog.set_level("INFO")
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What?", "answer": "test", "confidence": 0.9,
             "bloom_level": "understand", "reasoning_trace": "Direct recall"}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="cep_test",
                    name="test.mp3",
                    transcription_text="Test transcription text. " * 20,
                )
            )
        )

        qa_config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Check summary was logged
        assert "Batch CEP QA generation completed" in caplog.text

    def test_run_batch_cep_workers_greater_than_cpu_count(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CEP batch logs when workers > CPU count."""
        caplog.set_level("INFO")

        # Mock cpu_count to return small number
        mocker.patch("gtranscriber.core.qa_batch.mp.cpu_count", return_value=2)

        # Mock generate to succeed
        mock_gen = mocker.patch("gtranscriber.core.qa_batch.generate_cep_qa_for_transcription")
        mock_gen.return_value = ("id1", True, "Success")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create 5 test files
        for i in range(5):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"cep_id{i}",
                        name=f"test{i}.mp3",
                    )
                )
            )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        # Use 4 workers (more than mocked cpu_count of 2)
        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=4)

        # Check that info about workers > CPU count was logged
        assert "more than" in caplog.text.lower() and "cpu" in caplog.text.lower()

    def test_run_batch_cep_invalid_file_skipped(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid transcription files are skipped during CEP task loading."""
        caplog.set_level("WARNING")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create invalid JSON file
        invalid_file = input_dir / "invalid_transcription.json"
        invalid_file.write_text("{ not valid json")

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Check that warning was logged
        assert "skipping invalid file" in caplog.text.lower()

    def test_run_batch_cep_parallel_with_exception(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CEP parallel processing handles future.result() exceptions."""
        caplog.set_level("INFO")

        mock_executor_class = mocker.patch("gtranscriber.core.qa_batch.ProcessPoolExecutor")
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)

        mock_future1 = Mock()
        mock_future1.result.side_effect = RuntimeError("CEP Worker crashed")

        mock_executor.submit.return_value = mock_future1

        mocker.patch(
            "gtranscriber.core.qa_batch.as_completed",
            return_value=iter([mock_future1]),
        )

        mock_executor_class.return_value = mock_executor

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create test files
        for i in range(3):
            file = input_dir / f"file{i}_transcription.json"
            file.write_text(
                json.dumps(
                    create_test_enriched_data(
                        gdrive_id=f"cep_id{i}",
                        name=f"test{i}.mp3",
                    )
                )
            )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        # Use 2+ workers to trigger parallel path
        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=2)

        # Check that exception was logged
        assert "exception" in caplog.text.lower() or "failed" in caplog.text.lower()

    def test_run_batch_cep_final_summary_with_failures(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CEP final summary includes failure information."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create file that will fail (too short)
        file = input_dir / "fail_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    gdrive_id="fail_cep",
                    name="fail.mp3",
                    transcription_text="Short",
                )
            )
        )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Check summary mentions failures
        assert "Failed:" in caplog.text or "failed" in caplog.text.lower()

    def test_run_batch_cep_zero_total_files(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CEP final summary when total is zero."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Pre-create versioned run dir with empty checkpoint
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_checkpoint.write_text(
            json.dumps(
                {
                    "total_files": 0,
                    "completed_files": [],
                    "failed_files": {},
                }
            )
        )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        # Should exit early
        assert "no tasks" in caplog.text.lower()
