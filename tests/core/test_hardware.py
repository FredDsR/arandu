"""Tests for hardware detection and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from gtranscriber.config import TranscriberConfig
from gtranscriber.core.hardware import (
    DeviceType,
    HardwareConfig,
    get_device_and_dtype,
    get_quantization_config,
)


class TestGetDeviceAndDtype:
    """Tests for get_device_and_dtype function."""

    def test_force_cpu(self, mocker: pytest.fixture) -> None:
        """Test forcing CPU execution."""
        hw_config = get_device_and_dtype(force_cpu=True)

        assert hw_config.device == "cpu"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CPU

    def test_cuda_available_modern_architecture(self, mocker: pytest.fixture) -> None:
        """Test CUDA device selection with modern architecture (sm_70+)."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_capability.return_value = (7, 5)  # sm_75 (Turing)

        hw_config = get_device_and_dtype(force_cpu=False)

        assert hw_config.device == "cuda:0"
        assert hw_config.dtype == torch.float16
        assert hw_config.device_type == DeviceType.CUDA

    def test_cuda_available_with_quantization(self, mocker: pytest.fixture) -> None:
        """Test CUDA device selection with quantization enabled."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_capability.return_value = (8, 0)  # sm_80 (Ampere)

        hw_config = get_device_and_dtype(force_cpu=False, quantize=True)

        assert hw_config.device == "cuda:0"
        assert hw_config.dtype == torch.float32  # Quantization uses float32
        assert hw_config.device_type == DeviceType.CUDA

    def test_cuda_old_architecture_fallback(self, mocker: pytest.fixture) -> None:
        """Test fallback to CPU for old CUDA architectures (< sm_70)."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_capability.return_value = (6, 1)  # sm_61 (Pascal)

        hw_config = get_device_and_dtype(force_cpu=False)

        assert hw_config.device == "cpu"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CPU

    def test_cuda_capability_runtime_error_fallback(self, mocker: pytest.fixture) -> None:
        """Test fallback to CPU when get_device_capability raises RuntimeError."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_capability.side_effect = RuntimeError("No CUDA device")

        hw_config = get_device_and_dtype(force_cpu=False)

        assert hw_config.device == "cpu"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CPU

    def test_mps_available(self, mocker: pytest.fixture) -> None:
        """Test MPS device selection on Apple Silicon."""
        # Mock torch.cuda as unavailable
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = False

        # Mock MPS as available
        mock_backends = mocker.MagicMock()
        mock_backends.mps.is_available.return_value = True
        mocker.patch.object(torch, "backends", mock_backends)

        hw_config = get_device_and_dtype(force_cpu=False)

        assert hw_config.device == "mps"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.MPS

    def test_cpu_fallback_no_gpu(self, mocker: pytest.fixture) -> None:
        """Test CPU fallback when no GPU is available."""
        # Mock torch.cuda as unavailable
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = False

        # Mock no MPS backend by making hasattr return False
        mock_backends = mocker.MagicMock()
        # Remove mps attribute to make hasattr(torch.backends, "mps") return False
        delattr(mock_backends, "mps")
        mocker.patch.object(torch, "backends", mock_backends)

        hw_config = get_device_and_dtype(force_cpu=False)

        assert hw_config.device == "cpu"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CPU

    def test_with_config_force_cpu(self, mocker: pytest.fixture) -> None:
        """Test using TranscriberConfig with force_cpu enabled."""
        config = TranscriberConfig(force_cpu=True)

        hw_config = get_device_and_dtype(config=config)

        assert hw_config.device == "cpu"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CPU

    def test_with_config_quantize(self, mocker: pytest.fixture) -> None:
        """Test using TranscriberConfig with quantization enabled."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True
        mock_cuda.get_device_capability.return_value = (7, 5)

        config = TranscriberConfig(quantize=True)

        hw_config = get_device_and_dtype(config=config)

        assert hw_config.device == "cuda:0"
        assert hw_config.dtype == torch.float32
        assert hw_config.device_type == DeviceType.CUDA


class TestGetQuantizationConfig:
    """Tests for get_quantization_config function."""

    def test_quantization_disabled(self, mocker: pytest.fixture) -> None:
        """Test when quantization is disabled."""
        quant_config = get_quantization_config(quantize=False)

        assert quant_config is None

    def test_quantization_8bit_cuda_available(self, mocker: pytest.fixture) -> None:
        """Test 8-bit quantization with CUDA available."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True

        quant_config = get_quantization_config(quantize=True, bits=8)

        assert quant_config == {"load_in_8bit": True}

    def test_quantization_4bit_cuda_available(self, mocker: pytest.fixture) -> None:
        """Test 4-bit quantization with CUDA available."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True

        quant_config = get_quantization_config(quantize=True, bits=4)

        assert quant_config == {"load_in_4bit": True}

    def test_quantization_no_cuda(self, mocker: pytest.fixture) -> None:
        """Test quantization returns None when CUDA is unavailable."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = False

        quant_config = get_quantization_config(quantize=True, bits=8)

        assert quant_config is None

    def test_with_config_quantize_enabled(self, mocker: pytest.fixture) -> None:
        """Test using TranscriberConfig with quantization enabled."""
        mock_cuda = mocker.patch("torch.cuda")
        mock_cuda.is_available.return_value = True

        config = TranscriberConfig(quantize=True, quantize_bits=8)

        quant_config = get_quantization_config(config=config)

        assert quant_config == {"load_in_8bit": True}

    def test_with_config_quantize_disabled(self, mocker: pytest.fixture) -> None:
        """Test using TranscriberConfig with quantization disabled."""
        config = TranscriberConfig(quantize=False)

        quant_config = get_quantization_config(config=config)

        assert quant_config is None


class TestHardwareConfig:
    """Tests for HardwareConfig dataclass."""

    def test_hardware_config_creation(self) -> None:
        """Test creating HardwareConfig instance."""
        hw_config = HardwareConfig(
            device="cuda:0",
            dtype=torch.float16,
            device_type=DeviceType.CUDA,
        )

        assert hw_config.device == "cuda:0"
        assert hw_config.dtype == torch.float16
        assert hw_config.device_type == DeviceType.CUDA
