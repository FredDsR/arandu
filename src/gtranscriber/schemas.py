"""Pydantic schemas for G-Transcriber input and output data validation."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class InputRecord(BaseModel):
    """Schema for input records from Google Drive file metadata.

    Validates the existence of critical fields like gdrive_id, mimeType and parents
    before processing.
    """

    gdrive_id: str = Field(..., description="Google Drive file ID")
    name: str = Field(..., description="File name")
    mimeType: str = Field(..., description="MIME type of the file")
    parents: list[str] = Field(..., description="List of parent folder IDs")
    web_content_link: str = Field(..., alias="webContentLink", description="Direct download link")
    size_bytes: int | None = Field(None, description="File size in bytes")

    model_config = {"populate_by_name": True}

    @field_validator("parents", mode="before")
    @classmethod
    def parse_parents(cls, v: str | list[str]) -> list[str]:
        """Parse parents field from string or list format."""
        if isinstance(v, str):
            try:
                # Handle single-quoted JSON strings
                return json.loads(v.replace("'", '"'))
            except json.JSONDecodeError:
                return []
        if isinstance(v, list):
            return v
        return []


class TranscriptionSegment(BaseModel):
    """Schema for a transcription segment with timestamp information."""

    text: str = Field(..., description="Transcribed text for this segment")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class EnrichedRecord(InputRecord):
    """Schema for output records containing transcription results and metadata.

    This schema defines the format of the final JSON file that will be saved
    to Google Drive alongside the original media file.
    """

    transcription_text: str = Field(..., description="Full transcription text")
    detected_language: str = Field(..., description="Detected language code")
    language_probability: float = Field(..., description="Confidence score for detected language")
    model_id: str = Field(..., description="Hugging Face model ID used for transcription")
    compute_device: str = Field(..., description="Device used for computation (cpu/cuda/mps)")
    processing_duration_sec: float = Field(..., description="Processing time in seconds")
    transcription_status: str = Field(..., description="Status of transcription process")
    created_at_enrichment: datetime = Field(
        default_factory=datetime.now, description="Timestamp of enrichment"
    )
    segments: list[TranscriptionSegment] | None = Field(
        None, description="Detailed timestamp segments"
    )
