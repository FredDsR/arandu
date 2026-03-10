"""Protocol definition for metadata extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arandu.shared.schemas import SourceMetadata


@runtime_checkable
class MetadataExtractor(Protocol):
    """Protocol for catalog metadata extractors.

    Implementations parse catalog row data to extract structured
    interview metadata. Different catalog formats can provide
    different implementations.
    """

    def extract(self, row: dict[str, str]) -> SourceMetadata:
        """Extract structured metadata from a catalog row.

        Args:
            row: Dictionary of catalog column values (e.g., from CSV DictReader).

        Returns:
            SourceMetadata with extracted fields (unresolved fields are None).
        """
        ...
