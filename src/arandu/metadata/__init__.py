"""Metadata extraction for source file provenance.

Provides Protocol-based extractors for parsing interview metadata
from catalog information (filenames, paths, column data).
"""

from arandu.metadata.enrichment import enrich_with_source_metadata
from arandu.metadata.extractor import GDriveCatalogExtractor
from arandu.metadata.protocol import MetadataExtractor

__all__ = [
    "GDriveCatalogExtractor",
    "MetadataExtractor",
    "enrich_with_source_metadata",
]
