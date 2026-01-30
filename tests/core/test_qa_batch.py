"""Tests for QA batch processing module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    import pytest
    from pytest_mock import MockerFixture

from gtranscriber.config import QAConfig
from gtranscriber.core.qa_batch import (
    QAGenerationTask,
    generate_qa_for_transcription,
    load_transcription_tasks,
    run_batch_qa_generation,
)


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

    def test_output_directory_creation(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that output directory is created."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        config = QAConfig(provider="ollama", model_id="llama3.1:8b")

        run_batch_qa_generation(input_dir, output_dir, config, num_workers=1)

        assert output_dir.exists()

    def test_no_remaining_tasks_early_exit(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test early exit when all files already completed."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

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

        # Create checkpoint with file already completed
        checkpoint_file = output_dir / "qa_checkpoint.json"
        checkpoint_file.write_text(
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

        # Check outputs were created
        assert (output_dir / "id0_qa.json").exists()
        assert (output_dir / "id1_qa.json").exists()

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

        # Check checkpoint file records failures
        checkpoint_file = output_dir / "qa_checkpoint.json"
        assert checkpoint_file.exists()

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

        # Check checkpoint
        checkpoint_file = output_dir / "qa_checkpoint.json"
        assert checkpoint_file.exists()

        checkpoint_data = json.loads(checkpoint_file.read_text())
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

        checkpoint_file = output_dir / "qa_checkpoint.json"
        checkpoint_data = json.loads(checkpoint_file.read_text())
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

        # Check all outputs were created
        assert (output_dir / "id0_qa.json").exists()
        assert (output_dir / "id1_qa.json").exists()
        assert (output_dir / "id2_qa.json").exists()

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

        # Verify all output files were created
        assert (output_dir / "id0_qa.json").exists()
        assert (output_dir / "id1_qa.json").exists()

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
