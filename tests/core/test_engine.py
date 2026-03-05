"""Tests for Whisper ASR engine."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from arandu.core.engine import TranscriptionResult, WhisperEngine


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self) -> None:
        """Test creating a TranscriptionResult."""
        result = TranscriptionResult(
            text="This is a test transcription.",
            segments=[
                {"start": 0.0, "end": 1.5, "text": "This is a test"},
                {"start": 1.5, "end": 3.0, "text": "transcription."},
            ],
            detected_language="en",
            language_probability=0.99,
            processing_duration_sec=2.5,
            model_id="openai/whisper-large-v3",
            device="cuda:0",
        )

        assert result.text == "This is a test transcription."
        assert len(result.segments) == 2
        assert result.detected_language == "en"
        assert result.language_probability == 0.99
        assert result.processing_duration_sec == 2.5
        assert result.model_id == "openai/whisper-large-v3"
        assert result.device == "cuda:0"

    def test_transcription_result_no_segments(self) -> None:
        """Test TranscriptionResult with no segments."""
        result = TranscriptionResult(
            text="Test",
            segments=None,
            detected_language="en",
            language_probability=0.95,
            processing_duration_sec=1.0,
            model_id="test-model",
            device="cpu",
        )

        assert result.segments is None


class TestWhisperEngineInitialization:
    """Tests for WhisperEngine initialization."""

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_default_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test WhisperEngine initialization with defaults."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        engine = WhisperEngine()

        assert engine.model_id == "openai/whisper-large-v3"
        assert engine.language is None
        assert engine._pipe is None  # Lazy initialization
        mock_device.assert_called_once()

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_custom_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test WhisperEngine with custom parameters."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cuda:0",
            dtype=torch.float16,
            device_type=DeviceType.CUDA,
        )
        mock_quant_config.return_value = {"load_in_8bit": True}

        engine = WhisperEngine(
            model_id="openai/whisper-tiny",
            force_cpu=False,
            quantize=True,
            quantize_bits=8,
            chunk_length_s=30,
            stride_length_s=5,
            language="pt",
        )

        assert engine.model_id == "openai/whisper-tiny"
        assert engine.language == "pt"
        assert "chunk_length_s" in engine.pipe_kwargs
        assert engine.pipe_kwargs["chunk_length_s"] == 30
        assert engine.pipe_kwargs["stride_length_s"] == (5, 5)

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_invalid_quantize_bits(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test validation error for invalid quantize_bits."""
        with pytest.raises(ValueError) as exc_info:
            WhisperEngine(quantize=True, quantize_bits=16)

        assert "quantize_bits must be 4 or 8" in str(exc_info.value)

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_valid_quantize_bits_4(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test valid quantize_bits=4."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = {"load_in_4bit": True}

        engine = WhisperEngine(quantize=True, quantize_bits=4)
        assert engine.quant_config == {"load_in_4bit": True}

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_valid_quantize_bits_8(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test valid quantize_bits=8."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = {"load_in_8bit": True}

        engine = WhisperEngine(quantize=True, quantize_bits=8)
        assert engine.quant_config == {"load_in_8bit": True}


class TestWhisperEnginePipeline:
    """Tests for pipeline creation and usage."""

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_pipeline_lazy_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """Test that pipeline is initialized lazily."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        engine = WhisperEngine()

        # Pipeline should not be created yet
        assert engine._pipe is None

        # Access pipe property
        _ = engine.pipe

        # Now pipeline should be created
        mock_model.from_pretrained.assert_called_once()
        mock_processor.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_model_loading_os_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of OSError during model loading."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None
        mock_model.from_pretrained.side_effect = OSError("Network error")

        engine = WhisperEngine()

        with pytest.raises(RuntimeError) as exc_info:
            _ = engine.pipe

        assert "Could not download or load the model" in str(exc_info.value)
        assert "openai/whisper-large-v3" in str(exc_info.value)

    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_model_loading_value_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of ValueError during model loading."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None
        mock_model.from_pretrained.side_effect = ValueError("Invalid model")

        engine = WhisperEngine()

        with pytest.raises(RuntimeError) as exc_info:
            _ = engine.pipe

        assert "Invalid model ID" in str(exc_info.value)
        assert "openai/whisper-large-v3" in str(exc_info.value)

    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_model_loading_unexpected_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of unexpected errors during model loading."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None
        mock_model.from_pretrained.side_effect = RuntimeError("Unexpected error")

        engine = WhisperEngine()

        with pytest.raises(RuntimeError):
            _ = engine.pipe


class TestWhisperEngineConfiguration:
    """Tests for engine configuration handling."""

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_pipe_kwargs_with_chunk_length_only(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test pipe_kwargs when only chunk_length_s is specified."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        engine = WhisperEngine(chunk_length_s=30)

        assert "chunk_length_s" in engine.pipe_kwargs
        assert "stride_length_s" not in engine.pipe_kwargs

    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_pipe_kwargs_with_stride_length_only(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test pipe_kwargs when only stride_length_s is specified."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        engine = WhisperEngine(stride_length_s=5)

        assert "stride_length_s" in engine.pipe_kwargs
        assert engine.pipe_kwargs["stride_length_s"] == (5, 5)
        assert "chunk_length_s" not in engine.pipe_kwargs


class TestWhisperEngineTranscription:
    """Tests for transcription functionality."""

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_transcribe_basic(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test basic transcription."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        # Mock pipeline result
        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = {
            "text": "This is a test transcription.",
            "chunks": [
                {
                    "text": "This is",
                    "timestamp": [0.0, 1.0],
                },
                {
                    "text": "a test transcription.",
                    "timestamp": [1.0, 3.0],
                },
            ],
            "language": "en",
            "language_probability": 0.95,
        }
        mock_pipeline.return_value = mock_pipe_instance

        engine = WhisperEngine()

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        result = engine.transcribe(audio_file)

        assert result.text == "This is a test transcription."
        assert result.detected_language == "en"
        assert result.language_probability == 0.95
        assert result.segments is not None
        assert len(result.segments) == 2
        assert result.processing_duration_sec > 0

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_transcribe_no_chunks(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription without chunks."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = {
            "text": "Simple transcription.",
            "language": "en",
        }
        mock_pipeline.return_value = mock_pipe_instance

        engine = WhisperEngine()

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        result = engine.transcribe(audio_file)

        assert result.text == "Simple transcription."
        assert result.segments is None

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_transcribe_with_language(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription with specified language."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = {
            "text": "Teste de transcrição.",
            "language": "pt",
        }
        mock_pipeline.return_value = mock_pipe_instance

        engine = WhisperEngine(language="pt")

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        result = engine.transcribe(audio_file)

        assert result.text == "Teste de transcrição."
        # Check that language was passed to pipeline
        call_kwargs = mock_pipeline.call_args.kwargs
        assert "generate_kwargs" in call_kwargs
        assert call_kwargs["generate_kwargs"]["language"] == "pt"


class TestWhisperEngineDeviceFallback:
    """Tests for device fallback logic."""

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_device_fallback_to_cpu(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """Test fallback to CPU when device placement fails."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        # Initially try CUDA
        mock_device.return_value = HardwareConfig(
            device="cuda:0",
            dtype=torch.float16,
            device_type=DeviceType.CUDA,
        )
        mock_quant_config.return_value = None

        # Mock model that fails on first device placement
        mock_model_instance = Mock()
        mock_model_instance.to.side_effect = [
            RuntimeError("CUDA error"),
            mock_model_instance,  # Success on second attempt (CPU)
        ]
        mock_model.from_pretrained.return_value = mock_model_instance

        engine = WhisperEngine()

        # Access pipe to trigger pipeline creation
        _ = engine.pipe

        # Should have called .to() twice (once for CUDA, once for CPU)
        assert mock_model_instance.to.call_count == 2


class TestWhisperEngineEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_transcribe_empty_chunks(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription with empty chunks list."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = {
            "text": "Text",
            "chunks": [],  # Empty chunks
        }
        mock_pipeline.return_value = mock_pipe_instance

        engine = WhisperEngine()

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        result = engine.transcribe(audio_file)

        assert result.segments is None

    @patch("arandu.core.engine.pipeline")
    @patch("arandu.core.engine.AutoProcessor")
    @patch("arandu.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("arandu.core.engine.get_device_and_dtype")
    @patch("arandu.core.engine.get_quantization_config")
    def test_transcribe_missing_timestamps(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test transcription with missing timestamp data."""
        import torch

        from arandu.shared.hardware import DeviceType, HardwareConfig

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = {
            "text": "Text",
            "chunks": [
                {
                    "text": "Test",
                    # Missing timestamp field
                }
            ],
        }
        mock_pipeline.return_value = mock_pipe_instance

        engine = WhisperEngine()

        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        result = engine.transcribe(audio_file)

        assert result.segments is not None
        assert len(result.segments) == 1
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 0.0
