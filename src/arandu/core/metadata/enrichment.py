"""Source metadata enrichment for transcription records.

Provides the convenience function for attaching extracted metadata
to EnrichedRecord instances, following the same pattern as
``validate_enriched_record`` in ``transcription_validator.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arandu.core.metadata.extractor import GDriveCatalogExtractor

if TYPE_CHECKING:
    from arandu.core.metadata.protocol import MetadataExtractor
    from arandu.schemas import EnrichedRecord

logger = logging.getLogger(__name__)


def enrich_with_source_metadata(
    record: EnrichedRecord,
    catalog_row: dict[str, str],
    *,
    extractor: MetadataExtractor | None = None,
) -> EnrichedRecord:
    """Extract and attach source metadata to an enriched record.

    Mutates the record in-place, same pattern as ``validate_enriched_record``.

    Args:
        record: EnrichedRecord to enrich with source metadata.
        catalog_row: Raw catalog row dictionary (from CSV DictReader).
        extractor: Optional pre-instantiated extractor. If None, uses
            GDriveCatalogExtractor with defaults.

    Returns:
        The same record (mutated in-place) for chaining.
    """
    if extractor is None:
        extractor = GDriveCatalogExtractor()

    metadata = extractor.extract(catalog_row)
    record.source_metadata = metadata

    if metadata.extraction_confidence >= 0.5:
        logger.info(
            f"Extracted metadata for {record.file_id}: "
            f"participant={metadata.participant_name}, "
            f"location={metadata.location}, "
            f"confidence={metadata.extraction_confidence}"
        )
    else:
        logger.debug(
            f"Low-confidence metadata for {record.file_id}: "
            f"confidence={metadata.extraction_confidence}"
        )

    return record
