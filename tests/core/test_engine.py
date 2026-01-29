"""Tests for Whisper ASR engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytest_mock import MockerFixture

from gtranscriber.core.engine import TranscriptionResult, WhisperEngine


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

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_default_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test WhisperEngine initialization with defaults."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_custom_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test WhisperEngine with custom parameters."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_invalid_quantize_bits(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test validation error for invalid quantize_bits."""
        with pytest.raises(ValueError) as exc_info:
            WhisperEngine(quantize=True, quantize_bits=16)

        assert "quantize_bits must be 4 or 8" in str(exc_info.value)

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_valid_quantize_bits_4(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test valid quantize_bits=4."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = {"load_in_4bit": True}

        engine = WhisperEngine(quantize=True, quantize_bits=4)
        assert engine.quant_config == {"load_in_4bit": True}

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_valid_quantize_bits_8(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test valid quantize_bits=8."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.pipeline")
    @patch("gtranscriber.core.engine.AutoProcessor")
    @patch("gtranscriber.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_pipeline_lazy_initialization(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """Test that pipeline is initialized lazily."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_model_loading_os_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of OSError during model loading."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_model_loading_value_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of ValueError during model loading."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.AutoModelForSpeechSeq2Seq")
    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_model_loading_unexpected_error(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test handling of unexpected errors during model loading."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_pipe_kwargs_with_chunk_length_only(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test pipe_kwargs when only chunk_length_s is specified."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

        mock_device.return_value = HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )
        mock_quant_config.return_value = None

        engine = WhisperEngine(chunk_length_s=30)

        assert "chunk_length_s" in engine.pipe_kwargs
        assert "stride_length_s" not in engine.pipe_kwargs

    @patch("gtranscriber.core.engine.get_device_and_dtype")
    @patch("gtranscriber.core.engine.get_quantization_config")
    def test_pipe_kwargs_with_stride_length_only(
        self,
        mock_quant_config: MagicMock,
        mock_device: MagicMock,
    ) -> None:
        """Test pipe_kwargs when only stride_length_s is specified."""
        from gtranscriber.core.hardware import DeviceType, HardwareConfig

        import torch

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
