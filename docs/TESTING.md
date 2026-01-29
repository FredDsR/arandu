# G-Transcriber Test Suite Documentation

This document provides an overview of the test suite for G-Transcriber.

## Test Structure

The test suite follows the project structure, with tests mirroring the source code organization:

```
tests/
├── conftest.py              # Shared pytest fixtures
├── test_config.py           # Configuration module tests
├── test_schemas.py          # Pydantic schema tests
└── core/
    ├── test_hardware.py     # Hardware detection tests
    ├── test_llm_client.py   # LLM client tests
    ├── test_media.py        # Media file utility tests
    ├── test_checkpoint.py   # Checkpoint management tests
    └── test_io.py           # File I/O tests
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=gtranscriber --cov-report=term
```

### Run Specific Test Module
```bash
pytest tests/core/test_hardware.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_config.py::TestQAConfig -v
```

### Run Specific Test Function
```bash
pytest tests/core/test_llm_client.py::TestLLMClient::test_generate_basic -v
```

## Test Coverage

Current coverage: **39%** (540/1379 statements)

### Modules at 100% Coverage
- `core/hardware.py` - Device detection and quantization
- `core/llm_client.py` - Unified LLM client
- `core/checkpoint.py` - Batch processing checkpoints
- `core/io.py` - File I/O operations

### Modules at High Coverage
- `config.py` - 95% (Configuration classes)
- `schemas.py` - 98% (Pydantic data models)
- `media.py` - 53% (Media file utilities)

## Test Categories

### 1. Configuration Tests (`test_config.py`)
Tests for configuration classes that load from environment variables:
- Default initialization
- Environment variable overrides
- Field validation (boundaries, constraints)
- Invalid input handling

**Example**:
```python
def test_questions_per_document_boundary_max() -> None:
    """Test maximum boundary for questions_per_document."""
    config = QAConfig(questions_per_document=50)
    assert config.questions_per_document == 50
```

### 2. Schema Tests (`test_schemas.py`)
Tests for Pydantic models that validate data structures:
- Valid/invalid initialization
- Field validators
- Computed fields
- Save/load round-trips

**Example**:
```python
def test_start_time_greater_than_end_time() -> None:
    """Test validation error when start_time >= end_time."""
    with pytest.raises(ValidationError) as exc_info:
        QAPair(
            question="Test?",
            answer="Answer",
            context="Context",
            question_type="temporal",
            confidence=0.9,
            start_time=5.0,
            end_time=3.0,
        )
    assert "start_time must be less than end_time" in str(exc_info.value)
```

### 3. Hardware Tests (`test_hardware.py`)
Tests for device detection and configuration:
- CUDA, MPS, CPU device selection
- Quantization configuration
- Error handling for unsupported architectures

**Mocking Example**:
```python
from pytest_mock import MockerFixture

def test_cuda_available_modern_architecture(mocker: MockerFixture) -> None:
    """Test CUDA device selection with modern architecture (sm_70+)."""
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.get_device_capability.return_value = (7, 5)  # sm_75

    hw_config = get_device_and_dtype(force_cpu=False)

    assert hw_config.device == "cuda:0"
    assert hw_config.dtype == torch.float16
```

### 4. LLM Client Tests (`test_llm_client.py`)
Tests for the unified LLM client supporting OpenAI, Ollama, and custom providers:
- Provider initialization
- API availability checks
- Text generation with retry logic
- Error handling

**Mocking Example**:
```python
def test_generate_retry_on_failure(mocker: MockerFixture) -> None:
    """Test that generate retries on failure (tenacity decorator)."""
    mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
    mock_client = Mock()
    
    # First two calls fail, third succeeds
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Success"
    
    mock_client.chat.completions.create.side_effect = [
        Exception("API Error 1"),
        Exception("API Error 2"),
        mock_response,
    ]
    
    client = LLMClient(LLMProvider.OLLAMA, "llama3.1:8b")
    response = client.generate("Test prompt")
    
    assert response == "Success"
    assert mock_client.chat.completions.create.call_count == 3
```

### 5. Media Tests (`test_media.py`)
Tests for media file processing utilities:
- Audio stream detection
- Duration extraction using ffprobe
- Custom exceptions
- Error handling for corrupted files

**Mocking Example**:
```python
def test_has_audio_stream_success(mocker: MockerFixture) -> None:
    """Test detecting audio stream in media file."""
    mock_result = Mock()
    mock_result.stdout = json.dumps({"streams": [{"codec_type": "audio"}]})
    mock_run = mocker.patch("subprocess.run", return_value=mock_result)

    result = has_audio_stream("test.mp4")

    assert result is True
    assert "ffprobe" in mock_run.call_args[0][0]
```

### 6. Checkpoint Tests (`test_checkpoint.py`)
Tests for batch processing checkpoint management:
- State persistence
- Progress tracking
- Corrupted checkpoint recovery
- File completion tracking

**Example**:
```python
def test_mark_completed_removes_from_failed(tmp_path: Path) -> None:
    """Test that marking as completed removes from failed list."""
    manager = CheckpointManager(tmp_path / "checkpoint.json")
    manager.mark_failed("file1", "Some error")
    
    manager.mark_completed("file1")
    
    assert "file1" in manager.state.completed_files
    assert "file1" not in manager.state.failed_files
```

### 7. I/O Tests (`test_io.py`)
Tests for file operations and temporary file management:
- Temporary directory creation
- Temporary file creation
- EnrichedRecord saving
- MIME type detection
- Cleanup operations

**Example**:
```python
def test_cleanup_temp_files_ignores_other_files(tmp_path: Path) -> None:
    """Test that cleanup only removes gtranscriber_ files."""
    (tmp_path / "gtranscriber_file1.txt").touch()
    (tmp_path / "other_file.txt").touch()

    success, failure = cleanup_temp_files(str(tmp_path))

    assert success == 1
    assert (tmp_path / "other_file.txt").exists()
```

## Testing Best Practices

### 1. Mock External Dependencies
Always mock external services to avoid real API calls:
- Google Drive API
- OpenAI/Ollama APIs
- System commands (ffprobe, ffmpeg)
- PyTorch CUDA functions

### 2. Test Error Paths
Don't just test the happy path. Test error conditions:
- Invalid inputs
- Network failures
- Permission errors
- Corrupted data

### 3. Use Descriptive Test Names
Test names should describe what they test:
```python
def test_questions_per_document_above_max() -> None:
    """Test validation error when questions_per_document is above maximum."""
```

### 4. Use Fixtures
Use pytest fixtures for common setup:
```python
@pytest.fixture
def mock_torch_cuda(mocker: MockerFixture) -> MagicMock:
    """Mock torch.cuda module for hardware detection tests."""
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = False
    return mock_cuda
```

### 5. Test Boundaries
For validated fields, test:
- Minimum valid value
- Maximum valid value
- Below minimum (should fail)
- Above maximum (should fail)

## Common Fixtures

### `tmp_path` (Built-in)
Provides a temporary directory unique to each test function:
```python
def test_example(tmp_path: Path) -> None:
    file = tmp_path / "test.txt"
    file.write_text("content")
```

### `mocker` (pytest-mock)
Provides mocking functionality:
```python
def test_example(mocker: MockerFixture) -> None:
    mock_func = mocker.patch("module.function")
    mock_func.return_value = "mocked"
```

### `caplog` (Built-in)
Captures log messages:
```python
def test_example(caplog: pytest.LogCaptureFixture) -> None:
    function_that_logs()
    assert "expected message" in caplog.text
```

## Continuous Integration

Tests are run automatically on:
- Every commit
- Every pull request
- Before merging to main

Pre-commit checklist:
```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check --fix src/ tests/

# Run tests
pytest tests/

# Check coverage
pytest tests/ --cov=gtranscriber
```

## Adding New Tests

When adding a new test:

1. **Place it in the correct location**: Mirror the source structure
2. **Name it descriptively**: Use `test_` prefix and describe what it tests
3. **Add docstring**: Explain what the test validates
4. **Mock external dependencies**: No real API calls or file system operations (except in tmp_path)
5. **Test error paths**: Not just happy paths
6. **Run locally before committing**: Ensure all tests pass

Example template:
```python
def test_function_name_scenario(mocker: MockerFixture) -> None:
    """Test that function_name handles scenario correctly."""
    # Arrange: Set up test data and mocks
    mock_dependency = mocker.patch("module.dependency")
    mock_dependency.return_value = "expected"
    
    # Act: Execute the function
    result = function_name(param="value")
    
    # Assert: Verify expectations
    assert result == "expected"
    mock_dependency.assert_called_once()
```

## Troubleshooting

### Tests Failing Locally
1. Ensure all dependencies are installed: `pip install pytest pytest-cov pytest-mock`
2. Check Python version: Requires Python 3.13+
3. Clear pytest cache: `pytest --cache-clear`

### Coverage Not Updating
1. Delete `.coverage` file
2. Run with `--cov-report=term` to see live results
3. Ensure you're testing the right modules

### Import Errors
1. Ensure `src/` is in PYTHONPATH (configured in pyproject.toml)
2. Check that `__init__.py` files exist in test directories
3. Use absolute imports: `from gtranscriber.module import function`

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Project Testing Standards](../AGENTS.md#testing)
