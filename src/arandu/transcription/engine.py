"""Whisper ASR engine for Arandu.

Implements the transcription engine using Hugging Face's transformers
library with support for flexible model IDs and hardware configurations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from arandu.shared.hardware import (
    DeviceType,
    HardwareConfig,
    get_device_and_dtype,
    get_quantization_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers.pipelines import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from a transcription operation."""

    text: str
    segments: list[dict] | None
    detected_language: str
    language_probability: float
    processing_duration_sec: float
    model_id: str
    device: str


class WhisperEngine:
    """Whisper-based ASR engine with flexible model and hardware support.

    Supports any model_id from Hugging Face Hub (e.g., openai/whisper-large-v3,
    distil-whisper/distil-large-v3, openai/whisper-large-v3-turbo).
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        force_cpu: bool = False,
        quantize: bool = False,
        quantize_bits: int = 8,
        chunk_length_s: int | None = None,
        stride_length_s: int | None = None,
        language: str | None = None,
    ) -> None:
        """Initialize the Whisper engine.

        Args:
            model_id: Hugging Face model ID for the Whisper model.
            force_cpu: Force CPU execution.
            quantize: Enable quantization (8-bit or 4-bit loading).
            quantize_bits: Number of bits for quantization (4 or 8).
            chunk_length_s: Chunk length in seconds for long audio processing.
            stride_length_s: Stride length in seconds between chunks.
            language: Language code for transcription (e.g., 'pt' for Portuguese).
                If None, the language will be auto-detected.

        Raises:
            ValueError: If quantize_bits is not 4 or 8.
        """
        if quantize and quantize_bits not in (4, 8):
            raise ValueError(f"quantize_bits must be 4 or 8, got {quantize_bits}")

        self.model_id = model_id
        self.language = language

        self.pipe_kwargs = {}
        if chunk_length_s is not None:
            self.pipe_kwargs["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            self.pipe_kwargs["stride_length_s"] = (stride_length_s, stride_length_s)

        # Get hardware configuration
        self.hw_config: HardwareConfig = get_device_and_dtype(
            force_cpu=force_cpu,
            quantize=quantize,
        )

        # Get quantization config if applicable
        self.quant_config = get_quantization_config(quantize, quantize_bits)

        # Initialize pipeline lazily
        self._pipe: Pipeline | None = None

    @property
    def pipe(self) -> Pipeline:
        """Get the transcription pipeline, initializing if needed."""
        if self._pipe is None:
            self._pipe = self._create_pipeline()
        return self._pipe

    def _create_pipeline(self) -> Pipeline:
        """Create the Hugging Face ASR pipeline."""
        # Load model with appropriate configuration
        model_kwargs: dict = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        if self.quant_config:
            model_kwargs.update(self.quant_config)
        else:
            model_kwargs["torch_dtype"] = self.hw_config.dtype

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except OSError as e:
            logger.error(
                f"Failed to download or load the model '{self.model_id}'. "
                "This may be due to network issues or insufficient disk space. "
                f"Original error: {e}"
            )
            raise RuntimeError(
                f"Could not download or load the model '{self.model_id}'. "
                "Please check your internet connection and available disk space."
            ) from e
        except ValueError as e:
            logger.error(
                f"Invalid model ID '{self.model_id}' or incompatible model. Original error: {e}"
            )
            raise RuntimeError(
                f"Invalid model ID '{self.model_id}'. Please check the model name and try again."
            ) from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading the model '{self.model_id}': {e}"
            )
            raise

        # Move to device (if not using quantization which handles device placement)
        if not self.quant_config:
            try:
                model.to(self.hw_config.device)
            except Exception:
                # Fallback to CPU if device placement fails
                logger.exception(
                    f"Failed to move model to {self.hw_config.device}. Falling back to CPU."
                )
                self.hw_config = HardwareConfig(
                    device="cpu",
                    dtype=torch.float32,
                    device_type=DeviceType.CPU,
                )
                model.to(self.hw_config.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        # Build generate_kwargs for language setting
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.hw_config.dtype,
            device=self.hw_config.device if not self.quant_config else None,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
            **self.pipe_kwargs,
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe an audio or video file.

        Args:
            audio_path: Path to the audio/video file.

        Returns:
            TranscriptionResult with text, segments, and metadata.
        """
        start_time = time.time()

        result = self.pipe(str(audio_path))

        processing_duration = time.time() - start_time

        # Extract text and segments
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = None
        if chunks:
            segments = []
            for chunk in chunks:
                timestamp = chunk.get("timestamp", [0.0, 0.0])
                segments.append(
                    {
                        "text": chunk.get("text", ""),
                        "start": timestamp[0] if len(timestamp) >= 1 else 0.0,
                        "end": timestamp[1] if len(timestamp) >= 2 else 0.0,
                    }
                )

        # Extract language info if available
        detected_language = result.get("language", "unknown")
        language_probability = result.get("language_probability", 0.0)

        return TranscriptionResult(
            text=text,
            segments=segments,
            detected_language=detected_language,
            language_probability=language_probability,
            processing_duration_sec=processing_duration,
            model_id=self.model_id,
            device=self.hw_config.device,
        )


def transcribe_audio(
    audio_path: str | Path,
    model_id: str = "openai/whisper-large-v3",
    force_cpu: bool = False,
    quantize: bool = False,
    language: str | None = None,
) -> TranscriptionResult:
    """Convenience function to transcribe a single audio file.

    Args:
        audio_path: Path to the audio/video file.
        model_id: Hugging Face model ID for the Whisper model.
        force_cpu: Force CPU execution.
        quantize: Enable 8-bit quantization.
        language: Language code for transcription (e.g., 'pt' for Portuguese).
            If None, the language will be auto-detected.

    Returns:
        TranscriptionResult with transcription and metadata.
    """
    engine = WhisperEngine(
        model_id=model_id,
        force_cpu=force_cpu,
        quantize=quantize,
        language=language,
    )
    return engine.transcribe(audio_path)
