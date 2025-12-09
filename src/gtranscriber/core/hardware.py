"""Hardware detection and configuration for G-Transcriber.

Implements hardware agnosticism with automatic detection of available
compute devices (CPU, CUDA, MPS) and appropriate dtype selection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch


class DeviceType(str, Enum):
    """Supported compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class HardwareConfig:
    """Hardware configuration for model inference."""

    device: str
    dtype: torch.dtype
    device_type: DeviceType


def get_device_and_dtype(
    force_cpu: bool = False,
    quantize: bool = False,
) -> HardwareConfig:
    """Detect the best available compute device and appropriate dtype.

    Implements hardware agnosticism by inspecting the environment and
    applying the correct strategy.

    Args:
        force_cpu: Force CPU execution even if GPU is available.
        quantize: Whether quantization will be used (affects dtype selection).

    Returns:
        HardwareConfig with device string, dtype, and device type.
    """
    # Environment override to force CPU
    if os.getenv("UV_FORCE_CPU") == "1" or os.getenv("GTRANSCRIBER_FORCE_CPU") == "1":
        force_cpu = True

    if force_cpu:
        return HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            device_type=DeviceType.CPU,
        )

    # Check for CUDA
    if torch.cuda.is_available():
        try:
            # Check CUDA compute capability (sm_70 = 7.0 = Volta architecture and newer)
            major, _ = torch.cuda.get_device_capability(0)
        except Exception:
            major = 0

        # PyTorch 2.x official wheels commonly support >= sm_70
        if major >= 7:
            dtype = torch.float16 if not quantize else torch.float32
            return HardwareConfig(
                device="cuda:0",
                dtype=dtype,
                device_type=DeviceType.CUDA,
            )
        else:
            # Fallback for older CUDA architectures
            return HardwareConfig(
                device="cpu",
                dtype=torch.float32,
                device_type=DeviceType.CPU,
            )

    # Check for Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareConfig(
            device="mps",
            dtype=torch.float32,  # MPS works best with float32
            device_type=DeviceType.MPS,
        )

    # Default to CPU
    return HardwareConfig(
        device="cpu",
        dtype=torch.float32,
        device_type=DeviceType.CPU,
    )


def get_quantization_config(
    quantize: bool = False,
    bits: int = 8,
) -> dict | None:
    """Get quantization configuration for model loading.

    Uses bitsandbytes for 8-bit or 4-bit loading to reduce VRAM usage
    on GPUs with limited memory.

    Args:
        quantize: Whether to enable quantization.
        bits: Number of bits for quantization (4 or 8).

    Returns:
        Quantization configuration dict or None if not enabled.
    """
    if not quantize:
        return None

    # bitsandbytes only works on CUDA
    if not torch.cuda.is_available():
        return None

    if bits == 4:
        return {"load_in_4bit": True}
    return {"load_in_8bit": True}
