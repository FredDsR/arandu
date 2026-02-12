"""Tests for enrich_with_source_metadata convenience function."""

from __future__ import annotations

from datetime import datetime

import pytest

from gtranscriber.core.metadata.enrichment import enrich_with_source_metadata
from gtranscriber.schemas import EnrichedRecord, SourceMetadata


@pytest.fixture
def sample_record() -> EnrichedRecord:
    """Create a minimal EnrichedRecord for testing."""
    return EnrichedRecord(
        gdrive_id="test_id_123",
        name="Glenio_D.Elaine_30-07-2025_BARRA_20.mp4",
        mimeType="video/mp4",
        parents=["parent1"],
        webContentLink="https://example.com/download",
        transcription_text="This is a test transcription.",
        detected_language="pt",
        language_probability=0.95,
        model_id="openai/whisper-large-v3",
        compute_device="cuda",
        processing_duration_sec=10.5,
        transcription_status="completed",
        created_at_enrichment=datetime.now(),
    )


class TestEnrichWithSourceMetadata:
    """Tests for the enrichment function."""

    def test_sets_source_metadata(self, sample_record: EnrichedRecord) -> None:
        """Enrichment should set the source_metadata field."""
        assert sample_record.source_metadata is None

        catalog_row = {
            "name": "Glenio_D.Elaine_30-07-2025_BARRA_20.mp4",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "Glenio_D.Elaine_30-07-2025_BARRA_20.mp4"
            ),
        }
        enrich_with_source_metadata(sample_record, catalog_row)

        assert sample_record.source_metadata is not None
        assert isinstance(sample_record.source_metadata, SourceMetadata)

    def test_returns_record_for_chaining(self, sample_record: EnrichedRecord) -> None:
        """Enrichment should return the same record for chaining."""
        result = enrich_with_source_metadata(sample_record, {"name": "test.mp3"})
        assert result is sample_record

    def test_preserves_existing_fields(self, sample_record: EnrichedRecord) -> None:
        """Enrichment should not modify other record fields."""
        original_text = sample_record.transcription_text
        original_id = sample_record.gdrive_id

        enrich_with_source_metadata(sample_record, {"name": "test.mp3"})

        assert sample_record.transcription_text == original_text
        assert sample_record.gdrive_id == original_id

    def test_with_custom_extractor(self, sample_record: EnrichedRecord) -> None:
        """Should work with a custom extractor implementation."""

        class FixedExtractor:
            def extract(self, row: dict[str, str]) -> SourceMetadata:
                return SourceMetadata(
                    participant_name="Test Person",
                    location="Test Location",
                    extraction_confidence=1.0,
                )

        enrich_with_source_metadata(
            sample_record,
            {"name": "anything"},
            extractor=FixedExtractor(),
        )

        assert sample_record.source_metadata is not None
        assert sample_record.source_metadata.participant_name == "Test Person"
        assert sample_record.source_metadata.extraction_confidence == 1.0
