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

from arandu.config import CEPConfig, QAConfig
from arandu.core.qa_batch import (
    QAGenerationTask,
    TaskLoadResult,
    _init_cep_worker,
    generate_cep_qa_for_transcription,
    load_transcription_tasks,
    run_batch_cep_generation,
)
from arandu.core.results_manager import ResultsManager


class _ThreadPoolCompat(ThreadPoolExecutor):
    """ThreadPoolExecutor that ignores mp_context for test compatibility."""

    def __init__(self, *args: object, mp_context: object = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


def create_test_enriched_data(
    file_id: str = "test123", name: str = "test.mp3", transcription_text: str = "Test"
) -> dict:
    """Create a complete EnrichedRecord test data dictionary.

    Args:
        file_id: Google Drive ID.
        name: Filename.
        transcription_text: Transcription text.

    Returns:
        Complete EnrichedRecord data dictionary.
    """
    return {
        "file_id": file_id,
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
            file_id="test123",
            filename="test.mp3",
            output_file=output_file,
        )

        assert task.transcription_file == transcription_file
        assert task.file_id == "test123"
        assert task.filename == "test.mp3"
        assert task.output_file == output_file

    def test_task_field_accessibility(self, tmp_path: Path) -> None:
        """Test that all fields are accessible."""
        task = QAGenerationTask(
            transcription_file=tmp_path / "in.json",
            file_id="abc",
            filename="file.mp3",
            output_file=tmp_path / "out.json",
        )

        # Access all fields
        assert isinstance(task.transcription_file, Path)
        assert isinstance(task.file_id, str)
        assert isinstance(task.filename, str)
        assert isinstance(task.output_file, Path)


class TestLoadTranscriptionTasks:
    """Tests for load_transcription_tasks function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test that empty directory returns empty result."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        result = load_transcription_tasks(input_dir, output_dir)

        assert isinstance(result, TaskLoadResult)
        assert len(result.tasks) == 0
        assert result.total_found == 0
        assert result.skipped_invalid == 0

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
                    file_id="id1",
                    name="interview1.mp3",
                    transcription_text="This is a test transcription.",
                )
            )
        )

        file2 = input_dir / "file2_transcription.json"
        file2.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="id2",
                    name="interview2.mp3",
                    transcription_text="Another transcription.",
                )
            )
        )

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 2
        assert all(isinstance(task, QAGenerationTask) for task in result.tasks)

    def test_mixed_valid_invalid_files(self, tmp_path: Path) -> None:
        """Test that invalid files are skipped with warnings."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Valid file
        valid = input_dir / "valid_transcription.json"
        valid.write_text(json.dumps(create_test_enriched_data(file_id="id1", name="test.mp3")))

        # Invalid JSON file
        invalid = input_dir / "invalid_transcription.json"
        invalid.write_text("not valid json {")

        # Non-transcription file (should be ignored by glob)
        other = input_dir / "other_file.json"
        other.write_text(json.dumps({"data": "test"}))

        result = load_transcription_tasks(input_dir, output_dir)

        # Should only get the valid transcription file
        assert len(result.tasks) == 1
        assert result.tasks[0].file_id == "id1"

    def test_invalid_json_files(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of corrupt JSON files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        corrupt = input_dir / "corrupt_transcription.json"
        corrupt.write_text("{ corrupt json")

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 0
        assert "Skipping invalid file" in caplog.text

    def test_file_id_extraction(self, tmp_path: Path) -> None:
        """Test that file_id is correctly extracted from JSON."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="extracted_id_123",
                    name="test.mp3",
                )
            )
        )

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 1
        assert result.tasks[0].file_id == "extracted_id_123"

    def test_output_filename_generation(self, tmp_path: Path) -> None:
        """Test that output filename is {file_id}_cep_qa.json by default."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "anything_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="myid",
                    name="original.mp3",
                )
            )
        )

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 1
        assert result.tasks[0].output_file == output_dir / "myid_cep_qa.json"

    def test_skips_invalid_transcriptions(self, tmp_path: Path) -> None:
        """Test that transcriptions with is_valid=False are skipped."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Valid transcription (is_valid=True)
        valid_data = create_test_enriched_data(file_id="valid1", name="valid.mp3")
        valid_data["is_valid"] = True
        (input_dir / "valid1_transcription.json").write_text(json.dumps(valid_data))

        # Invalid transcription (is_valid=False)
        invalid_data = create_test_enriched_data(file_id="invalid1", name="invalid.mp3")
        invalid_data["is_valid"] = False
        (input_dir / "invalid1_transcription.json").write_text(json.dumps(invalid_data))

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 1
        assert result.tasks[0].file_id == "valid1"

    def test_includes_unchecked_transcriptions(self, tmp_path: Path) -> None:
        """Test that transcriptions with is_valid=None (unchecked) are included."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Unchecked transcription (is_valid=None)
        unchecked_data = create_test_enriched_data(file_id="unchecked1", name="unchecked.mp3")
        unchecked_data["is_valid"] = None
        (input_dir / "unchecked1_transcription.json").write_text(json.dumps(unchecked_data))

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 1
        assert result.tasks[0].file_id == "unchecked1"

    def test_includes_transcriptions_without_is_valid_field(self, tmp_path: Path) -> None:
        """Test that transcriptions missing the is_valid field are included."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Transcription without is_valid key at all
        data = create_test_enriched_data(file_id="no_field", name="old.mp3")
        data.pop("is_valid", None)
        (input_dir / "no_field_transcription.json").write_text(json.dumps(data))

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 1
        assert result.tasks[0].file_id == "no_field"

    def test_filters_mixed_validity(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test filtering with a mix of valid, invalid, and unchecked transcriptions."""
        caplog.set_level("INFO")
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # is_valid=True
        d1 = create_test_enriched_data(file_id="ok1", name="ok1.mp3")
        d1["is_valid"] = True
        (input_dir / "ok1_transcription.json").write_text(json.dumps(d1))

        # is_valid=False
        d2 = create_test_enriched_data(file_id="bad1", name="bad1.mp3")
        d2["is_valid"] = False
        (input_dir / "bad1_transcription.json").write_text(json.dumps(d2))

        # is_valid=False
        d3 = create_test_enriched_data(file_id="bad2", name="bad2.mp3")
        d3["is_valid"] = False
        (input_dir / "bad2_transcription.json").write_text(json.dumps(d3))

        # is_valid=None (unchecked)
        d4 = create_test_enriched_data(file_id="unk1", name="unk1.mp3")
        d4["is_valid"] = None
        (input_dir / "unk1_transcription.json").write_text(json.dumps(d4))

        result = load_transcription_tasks(input_dir, output_dir)

        task_ids = {t.file_id for t in result.tasks}
        assert task_ids == {"ok1", "unk1"}
        assert "2 skipped as invalid" in caplog.text

    def test_skipped_invalid_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test that skipped invalid transcriptions are logged."""
        caplog.set_level("INFO")
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        invalid_data = create_test_enriched_data(file_id="bad1", name="bad.mp3")
        invalid_data["is_valid"] = False
        (input_dir / "bad1_transcription.json").write_text(json.dumps(invalid_data))

        result = load_transcription_tasks(input_dir, output_dir)

        assert len(result.tasks) == 0
        assert "Skipping invalid transcription" in caplog.text

    def test_output_filename_custom_suffix(self, tmp_path: Path) -> None:
        """Test that output_suffix parameter controls the filename suffix."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file = input_dir / "anything_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="myid",
                    name="original.mp3",
                )
            )
        )

        result = load_transcription_tasks(input_dir, output_dir, output_suffix="_custom.json")

        assert len(result.tasks) == 1
        assert result.tasks[0].output_file == output_dir / "myid_custom.json"

    def test_result_metadata_counts(self, tmp_path: Path) -> None:
        """Test that TaskLoadResult exposes correct total_found and skipped_invalid."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # 2 valid, 1 invalid, 1 corrupt JSON = 4 total files found
        d1 = create_test_enriched_data(file_id="v1", name="v1.mp3")
        d1["is_valid"] = True
        (input_dir / "v1_transcription.json").write_text(json.dumps(d1))

        d2 = create_test_enriched_data(file_id="v2", name="v2.mp3")
        (input_dir / "v2_transcription.json").write_text(json.dumps(d2))

        d3 = create_test_enriched_data(file_id="bad1", name="bad1.mp3")
        d3["is_valid"] = False
        (input_dir / "bad1_transcription.json").write_text(json.dumps(d3))

        (input_dir / "corrupt_transcription.json").write_text("{ bad json")

        result = load_transcription_tasks(input_dir, output_dir)

        assert result.total_found == 4
        assert result.skipped_invalid == 1
        assert len(result.tasks) == 2


class TestInitCEPWorker:
    """Tests for _init_cep_worker initialization function."""

    def test_init_cep_worker_without_validation(self, mocker: MockerFixture) -> None:
        """Test CEP worker initialization without validation enabled."""
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock CEPQAGenerator
        mock_cep_generator_class = mocker.patch("arandu.core.cep.CEPQAGenerator")
        mock_cep_generator = Mock()
        mock_cep_generator_class.return_value = mock_cep_generator

        # Reset global state
        import arandu.core.qa_batch as qa_batch_module

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
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock CEPQAGenerator
        mock_cep_generator_class = mocker.patch("arandu.core.cep.CEPQAGenerator")
        mock_cep_generator = Mock()
        mock_cep_generator_class.return_value = mock_cep_generator

        # Reset global state
        import arandu.core.qa_batch as qa_batch_module

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
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
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
        import arandu.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "file_id": "cep_test123",
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
            file_id="cep_test123",
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

        file_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert file_id == "cep_test123"
        assert success is True
        assert message == "Success"
        assert output_file.exists()

    def test_generate_cep_qa_validation_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test CEP QA generation with validation error (too short transcription)."""
        mocker.patch("arandu.core.llm_client.OpenAI")

        # Reset global state
        import arandu.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "short_transcription.json"
        input_file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="short_cep",
                    name="short.mp3",
                    transcription_text="Short",
                )
            )
        )

        output_file = tmp_path / "short_cep_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            file_id="short_cep",
            filename="short.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(provider="ollama", model_id="llama3.1:8b").model_dump()
        cep_config_dict = CEPConfig(enable_validation=False).model_dump()

        file_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert file_id == "short_cep"
        assert success is False
        assert "short" in message.lower() or len(message) > 0

    def test_generate_cep_qa_generic_exception(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test CEP QA generation with generic exception (file not found)."""
        mocker.patch("arandu.core.llm_client.OpenAI")

        # Reset global state
        import arandu.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        # Non-existent file
        input_file = tmp_path / "nonexistent.json"
        output_file = tmp_path / "output_cep_qa.json"

        task = QAGenerationTask(
            transcription_file=input_file,
            file_id="test_cep",
            filename="test.mp3",
            output_file=output_file,
        )

        qa_config_dict = QAConfig(provider="ollama", model_id="llama3.1:8b").model_dump()
        cep_config_dict = CEPConfig(enable_validation=False).model_dump()

        file_id, success, message = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert file_id == "test_cep"
        assert success is False
        assert len(message) > 0

    def test_generate_cep_qa_with_validation_enabled(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CEP QA generation with validation enabled."""
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
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
        import arandu.core.qa_batch as qa_batch_module

        qa_batch_module._worker_cep_generator = None

        input_file = tmp_path / "test_transcription.json"
        input_file.write_text(
            json.dumps(
                {
                    "file_id": "cep_validated",
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
            file_id="cep_validated",
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

        file_id, success, _ = generate_cep_qa_for_transcription(
            task, qa_config_dict, cep_config_dict
        )

        assert file_id == "cep_validated"
        assert success is True
        assert output_file.exists()


class TestRunBatchCEPGeneration:
    """Tests for run_batch_cep_generation function."""

    @pytest.fixture(autouse=True)
    def setup_versioning(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Configure versioned results for CEP batch tests."""
        self.results_dir = (tmp_path / "results").resolve()

        mock_rc = mocker.patch("arandu.core.qa_batch.ResultsConfig")
        mock_rc.return_value.enable_versioning = True
        mock_rc.return_value.base_dir = tmp_path / "results"

        mocker.patch.object(ResultsManager, "_generate_pipeline_id", return_value="test_run")
        mocker.patch("arandu.core.qa_batch.ProcessPoolExecutor", _ThreadPoolCompat)

        self.run_dir = self.results_dir / "test_run" / "cep"
        self.outputs_dir = self.run_dir / "outputs"
        self.versioned_checkpoint = self.run_dir / "cep_checkpoint.json"

    def test_run_batch_cep_creates_output_dir(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that output directory is created for CEP batch."""
        mocker.patch("arandu.core.llm_client.OpenAI")

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
        mocker.patch("arandu.core.llm_client.OpenAI")

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
        mocker.patch("arandu.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create transcription file
        file = input_dir / "test_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="cep_id1",
                    name="test.mp3",
                )
            )
        )

        # Mock checkpoint to report file as already completed
        mocker.patch(
            "arandu.core.qa_batch.CheckpointManager.is_completed",
            return_value=True,
        )

        qa_config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        assert "already processed" in caplog.text.lower() or "all files" in caplog.text.lower()

    def test_run_batch_cep_sequential_processing(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CEP sequential processing with single worker."""
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
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
                        file_id=f"cep_id{i}",
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
        mocker.patch("arandu.core.llm_client.OpenAI")

        # Mock generate to succeed for one, fail for another
        mock_gen = mocker.patch("arandu.core.qa_batch.generate_cep_qa_for_transcription")
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
                        file_id=f"{'valid' if i == 0 else 'invalid'}_cep",
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
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
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
                    file_id="cep_test",
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
        mocker.patch("arandu.core.qa_batch.mp.cpu_count", return_value=2)

        # Mock generate to succeed
        mock_gen = mocker.patch("arandu.core.qa_batch.generate_cep_qa_for_transcription")
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
                        file_id=f"cep_id{i}",
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
        mocker.patch("arandu.core.llm_client.OpenAI")

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

        mock_executor_class = mocker.patch("arandu.core.qa_batch.ProcessPoolExecutor")
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)

        mock_future1 = Mock()
        mock_future1.result.side_effect = RuntimeError("CEP Worker crashed")

        mock_executor.submit.return_value = mock_future1

        mocker.patch(
            "arandu.core.qa_batch.as_completed",
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
                        file_id=f"cep_id{i}",
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
        mocker.patch("arandu.core.llm_client.OpenAI")

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "cep_output"
        input_dir.mkdir()

        # Create file that will fail (too short)
        file = input_dir / "fail_transcription.json"
        file.write_text(
            json.dumps(
                create_test_enriched_data(
                    file_id="fail_cep",
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
        mocker.patch("arandu.core.llm_client.OpenAI")

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

    def test_run_batch_cep_logs_skipped_invalid(
        self, tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that the final summary includes skipped invalid count."""
        caplog.set_level("INFO")
        mock_openai = mocker.patch("arandu.core.llm_client.OpenAI")
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

        # Valid transcription
        valid_data = create_test_enriched_data(
            file_id="ok1", name="ok.mp3", transcription_text="Test text. " * 20
        )
        valid_data["is_valid"] = True
        (input_dir / "ok1_transcription.json").write_text(json.dumps(valid_data))

        # Invalid transcription (should be skipped)
        bad_data = create_test_enriched_data(file_id="bad1", name="bad.mp3")
        bad_data["is_valid"] = False
        (input_dir / "bad1_transcription.json").write_text(json.dumps(bad_data))

        qa_config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=1,
        )
        cep_config = CEPConfig(enable_validation=False)

        run_batch_cep_generation(input_dir, output_dir, qa_config, cep_config, num_workers=1)

        assert "Skipped invalid transcriptions: 1" in caplog.text
