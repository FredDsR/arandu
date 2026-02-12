"""Tests for GDriveCatalogExtractor with real catalog patterns."""

from __future__ import annotations

import pytest

from gtranscriber.core.metadata.extractor import GDriveCatalogExtractor


@pytest.fixture
def extractor() -> GDriveCatalogExtractor:
    """Create extractor with default config."""
    return GDriveCatalogExtractor()


class TestUnderscoreDelimitedFilenames:
    """Test underscore-delimited filename patterns."""

    def test_researcher_location_date_seq(self, extractor: GDriveCatalogExtractor) -> None:
        """Pattern: Glenio_BarraDePelotas_30-07-2025_14.mp4"""
        row = {
            "name": "Glenio_BarraDePelotas_30-07-2025_14.mp4",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "[Glenio] Vídeos campo 30-07-2025 Barra de Pelotas/"
                "Glenio_BarraDePelotas_30-07-2025_14.mp4"
            ),
        }
        meta = extractor.extract(row)
        assert meta.researcher_name == "Glenio"
        assert meta.recording_date == "30-07-2025"
        assert meta.sequence_number == 14
        assert meta.location is not None

    def test_researcher_participant_date_location_seq(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Pattern: Dani_Henrique_15-11-2025_BARRA_08.MOV"""
        row = {
            "name": "Dani_Henrique_15-11-2025_BARRA_08.MOV",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "Entrevista Henrique 15-11-2025 BARRA de Pelotas/"
                "Dani_Henrique_15-11-2025_BARRA_08.MOV"
            ),
        }
        meta = extractor.extract(row)
        assert meta.researcher_name is not None
        assert meta.recording_date == "15-11-2025"
        assert meta.sequence_number == 8

    def test_researcher_participant_with_honorific(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Pattern: Glenio_D.Elaine_30-07-2025_BARRA_20.mp4"""
        row = {
            "name": "Glenio_D.Elaine_30-07-2025_BARRA_20.mp4",
            "gdrive_path": "",
        }
        meta = extractor.extract(row)
        assert meta.researcher_name == "Glenio"
        assert meta.participant_name == "D.Elaine"
        assert meta.recording_date == "30-07-2025"
        assert meta.sequence_number == 20

    def test_combined_researcher_participant(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Pattern: DaniBorges_D.Maria_30-07-25_03.mp4"""
        row = {
            "name": "DaniBorges_D.Maria_30-07-25_03.mp4",
            "gdrive_path": "",
        }
        meta = extractor.extract(row)
        assert meta.researcher_name is not None
        assert meta.participant_name == "D.Maria"
        assert meta.recording_date == "30-07-25"
        assert meta.sequence_number == 3


class TestDashDelimitedFilenames:
    """Test dash/space-delimited filename patterns."""

    def test_researcher_participant_date_seq(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Pattern: Dani Borges-Pescador Henrique 15-11-25 02.m4a"""
        row = {
            "name": "Dani Borges-Pescador Henrique 15-11-25 02.m4a",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/AUDIOS /"
                "Dani Borges-Pescador Henrique 15-11-25 02.m4a"
            ),
        }
        meta = extractor.extract(row)
        assert meta.researcher_name == "Dani Borges"
        assert meta.participant_name == "Pescador Henrique"
        assert meta.recording_date == "15-11-25"
        assert meta.sequence_number == 2

    def test_location_participant_date_parte(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Pattern: Barra - Célia 20.05 Parte I.mp3"""
        row = {
            "name": "Barra - Célia 20.05 Parte I.mp3",
            "gdrive_path": "",
        }
        meta = extractor.extract(row)
        assert meta.recording_date == "20.05"
        assert meta.sequence_label is not None
        assert "Parte" in meta.sequence_label


class TestCameraAndAndroidFilenames:
    """Test camera and Android auto-generated filenames."""

    def test_camera_file_no_name_metadata(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Camera files like MVI_7765.MOV have no metadata in the name."""
        row = {
            "name": "MVI_7765.MOV",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "DOQUINHAS- Fotos, Vídeos e Áudios/"
                "Dona Gilda 12-11-25- Dani Borges ORGANIZAR/"
                "MVI_7765.MOV"
            ),
        }
        meta = extractor.extract(row)
        # Should get location from path
        assert meta.location is not None
        assert "DOQUINHAS" in meta.location.upper()
        # Should get event context from subfolder
        assert meta.event_context is not None

    def test_android_vid_extracts_date(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Android VID_YYYYMMDD files should extract the date."""
        row = {
            "name": "VID_20250718_093806817~3.mp4",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "Entrevista D. Celia/"
                "VID_20250718_093806817~3.mp4"
            ),
        }
        meta = extractor.extract(row)
        assert meta.recording_date == "18-07-2025"
        assert meta.sequence_number == 3
        assert meta.location is not None


class TestPathExtraction:
    """Test metadata extraction from gdrive_path."""

    def test_barra_de_pelotas_location(self, extractor: GDriveCatalogExtractor) -> None:
        """Location folder: BARRA DE PELOTAS - Fotos, Vídeos, Áudios."""
        row = {
            "name": "test.mp3",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/AUDIOS/test.mp3"
            ),
        }
        meta = extractor.extract(row)
        assert meta.location is not None
        assert "BARRA DE PELOTAS" in meta.location.upper()

    def test_doquinhas_location(self, extractor: GDriveCatalogExtractor) -> None:
        """Location folder: DOQUINHAS- Fotos, Vídeos e Áudios."""
        row = {
            "name": "test.mp3",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "DOQUINHAS- Fotos, Vídeos e Áudios/test.mp3"
            ),
        }
        meta = extractor.extract(row)
        assert meta.location is not None
        assert "DOQUINHAS" in meta.location.upper()

    def test_entrevista_event_context(self, extractor: GDriveCatalogExtractor) -> None:
        """Subfolder like 'Entrevista D. Silvia 29-11-2025 Barra de Pelotas'."""
        row = {
            "name": "MVI_8875.MOV",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "Entrevista D. Silvia 29-11-2025 Barra de Pelotas/"
                "MVI_8875.MOV"
            ),
        }
        meta = extractor.extract(row)
        assert meta.event_context is not None
        assert "Entrevista" in meta.event_context
        # Should extract participant from event context
        assert meta.participant_name is not None
        assert "Silvia" in meta.participant_name


class TestConfidence:
    """Test confidence scoring."""

    def test_high_confidence_full_metadata(self, extractor: GDriveCatalogExtractor) -> None:
        """File with all extractable metadata should have high confidence."""
        row = {
            "name": "Glenio_D.Elaine_30-07-2025_BARRA_20.mp4",
            "gdrive_path": (
                "/Meu Drive/Projeto Desastres Climáticos/IMAGENS e ÁUDIOS/"
                "BARRA DE PELOTAS - Fotos, Vídeos, Áudios/VÍDEOS/"
                "Entrevista D. Elaine 30-07-2025 BARRA de Pelotas/"
                "Glenio_D.Elaine_30-07-2025_BARRA_20.mp4"
            ),
        }
        meta = extractor.extract(row)
        assert meta.extraction_confidence >= 0.75

    def test_low_confidence_camera_file_no_path(
        self, extractor: GDriveCatalogExtractor
    ) -> None:
        """Camera file without gdrive_path should have low confidence."""
        row = {"name": "MVI_7765.MOV", "gdrive_path": ""}
        meta = extractor.extract(row)
        assert meta.extraction_confidence <= 0.25


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_row(self, extractor: GDriveCatalogExtractor) -> None:
        """Empty row should return default SourceMetadata."""
        meta = extractor.extract({})
        assert meta.participant_name is None
        assert meta.extraction_confidence == 0.0

    def test_missing_gdrive_path(self, extractor: GDriveCatalogExtractor) -> None:
        """Row without gdrive_path should still extract from filename."""
        row = {"name": "Dani Borges-Pescador Henrique 15-11-25 02.m4a"}
        meta = extractor.extract(row)
        assert meta.researcher_name == "Dani Borges"

    def test_created_time_fallback_date(self, extractor: GDriveCatalogExtractor) -> None:
        """When no date in filename/path, created_time should be used."""
        row = {
            "name": "MVI_7765.MOV",
            "gdrive_path": "",
            "created_time": "2025-11-28T14:17:26.810Z",
        }
        meta = extractor.extract(row)
        assert meta.recording_date == "2025-11-28"

    def test_custom_researchers_and_locations(self) -> None:
        """Custom known_researchers and known_locations should be used."""
        custom = GDriveCatalogExtractor(
            known_researchers={"Alice"},
            known_locations={"LONDON"},
        )
        row = {"name": "Alice_Bob_01-01-2025_LONDON_01.mp4", "gdrive_path": ""}
        meta = custom.extract(row)
        assert meta.researcher_name == "Alice"
        assert meta.location == "LONDON"
