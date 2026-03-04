"""Protocol definition for knowledge graph constructors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.core.kg.schemas import KGConstructionResult
    from arandu.schemas import EnrichedRecord


@runtime_checkable
class KGConstructor(Protocol):
    """Protocol for knowledge graph construction backends.

    Implementations receive transcription records and produce a corpus-level
    knowledge graph in GraphML format.  Internal pipeline steps (e.g. triple
    extraction, conceptualization) are hidden behind this single method.
    """

    def build_graph(
        self,
        records: list[EnrichedRecord],
        output_dir: Path,
    ) -> KGConstructionResult:
        """Build a knowledge graph from transcription records.

        Args:
            records: Validated transcription records to extract triples from.
            output_dir: Directory where graph artifacts are written.

        Returns:
            Result containing the graph file path, metadata, and statistics.
        """
        ...
