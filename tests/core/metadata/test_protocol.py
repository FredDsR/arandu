"""Tests for MetadataExtractor protocol conformance."""

from __future__ import annotations

from arandu.core.metadata.extractor import GDriveCatalogExtractor
from arandu.core.metadata.protocol import MetadataExtractor
from arandu.shared.schemas import SourceMetadata


class TestMetadataExtractorProtocol:
    """Verify protocol conformance."""

    def test_gdrive_extractor_satisfies_protocol(self) -> None:
        """GDriveCatalogExtractor must satisfy the MetadataExtractor protocol."""
        extractor = GDriveCatalogExtractor()
        assert isinstance(extractor, MetadataExtractor)

    def test_custom_extractor_satisfies_protocol(self) -> None:
        """A custom class with the right signature must satisfy the protocol."""

        class DummyExtractor:
            def extract(self, row: dict[str, str]) -> SourceMetadata:
                return SourceMetadata()

        assert isinstance(DummyExtractor(), MetadataExtractor)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """A class without extract() must NOT satisfy the protocol."""

        class Bad:
            pass

        assert not isinstance(Bad(), MetadataExtractor)
