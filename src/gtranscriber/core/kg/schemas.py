"""Schemas for knowledge graph construction results."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — Pydantic needs Path at runtime

from pydantic import BaseModel, Field

from gtranscriber.schemas import KGMetadata  # noqa: TC001 — Pydantic needs at runtime


class KGConstructionResult(BaseModel):
    """Framework-agnostic result of a knowledge graph construction run.

    Returned by every ``KGConstructor.build_graph()`` implementation so that
    the orchestrator and CLI layer never depend on backend internals.
    """

    graph_file: Path = Field(..., description="Path to the output GraphML file")
    metadata: KGMetadata = Field(..., description="Provenance sidecar metadata")
    node_count: int = Field(..., ge=0, description="Number of nodes in the graph")
    edge_count: int = Field(..., ge=0, description="Number of edges in the graph")
    source_record_ids: list[str] = Field(
        ..., description="gdrive_ids of processed transcription records"
    )
