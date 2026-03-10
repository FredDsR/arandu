"""Knowledge Graph pipeline Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class KGMetadata(BaseModel):
    """Lightweight metadata for knowledge graph provenance tracking.

    Stored as a JSON sidecar file alongside the GraphML graph file.
    """

    graph_id: str = Field(..., description="Unique graph identifier")
    source_documents: list[str] = Field(..., description="List of source document IDs (file_ids)")
    model_id: str = Field(..., description="LLM model used for extraction")
    provider: str = Field(..., description="LLM provider")
    language: str = Field(default="pt", description="Language code for extraction (ISO 639-1)")
    created_at: datetime = Field(default_factory=datetime.now, description="When graph was created")
    total_documents: int = Field(default=0, description="Number of documents processed")
    total_nodes: int | None = Field(default=None, description="Number of nodes in the graph")
    total_edges: int | None = Field(default=None, description="Number of edges in the graph")
    backend_version: str | None = Field(
        default=None, description="KGC backend identifier and version"
    )

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> KGMetadata:
        """Load metadata from JSON file."""
        return cls.model_validate_json(Path(path).read_text())


# =============================================================================
# Evaluation Schemas
# =============================================================================


class GraphConnectivity(BaseModel):
    """Graph connectivity metrics from NetworkX analysis."""

    average_degree: float = Field(..., ge=0.0, description="Average node degree")
    connected_components: int = Field(..., ge=1, description="Number of connected components")
    largest_component_size: int = Field(..., ge=0, description="Size of largest component")
    density: float = Field(..., ge=0.0, le=1.0, description="Graph density")


class EntityCoverageResult(BaseModel):
    """Entity coverage metrics for knowledge graph evaluation."""

    total_entities: int = Field(..., ge=0, description="Total entities extracted")
    unique_entities: int = Field(..., ge=0, description="Number of unique entities")
    entity_density: float = Field(..., ge=0.0, description="Entities per 100 tokens")
    entity_type_distribution: dict[str, int] = Field(
        ..., description="Count by entity type (PERSON, LOCATION, etc.)"
    )

    @computed_field
    @property
    def entity_diversity(self) -> float:
        """Compute entity diversity as unique/total ratio."""
        if self.total_entities == 0:
            return 0.0
        return self.unique_entities / self.total_entities


class RelationMetricsResult(BaseModel):
    """Relation density and connectivity metrics."""

    total_relations: int = Field(..., ge=0, description="Total relations extracted")
    unique_relations: int = Field(..., ge=0, description="Number of unique relation types")
    relation_density: float = Field(..., ge=0.0, description="Relations per entity")
    graph_connectivity: GraphConnectivity = Field(..., description="Graph connectivity metrics")

    @computed_field
    @property
    def relation_diversity(self) -> float:
        """Compute relation diversity as unique/total ratio."""
        if self.total_relations == 0:
            return 0.0
        return self.unique_relations / self.total_relations


class SemanticQualityResult(BaseModel):
    """Semantic quality metrics for knowledge evaluation."""

    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Semantic coherence (0.0-1.0)")
    information_density: float = Field(
        ..., ge=0.0, description="(Entities + Relations) / text_length"
    )
    knowledge_coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Entities covered by QA pairs (0.0-1.0)"
    )


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report for knowledge elicitation quality."""

    dataset_name: str = Field(..., description="Name/identifier of evaluated dataset")
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.now, description="When evaluation was run"
    )
    total_documents: int = Field(..., ge=0, description="Number of documents evaluated")
    total_qa_pairs: int = Field(..., ge=0, description="Total QA pairs in dataset")

    # QA metrics (optional - computed if QA data available)
    qa_exact_match: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Exact match score"
    )
    qa_f1_score: float | None = Field(default=None, ge=0.0, le=1.0, description="F1 score")
    qa_bleu_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="BLEU score (0-100)"
    )

    # Component metrics (optional - computed if data available)
    entity_coverage: EntityCoverageResult | None = Field(
        default=None, description="Entity coverage metrics"
    )
    relation_metrics: RelationMetricsResult | None = Field(
        default=None, description="Relation metrics"
    )
    semantic_quality: SemanticQualityResult | None = Field(
        default=None, description="Semantic quality metrics"
    )

    # Summary
    recommendations: list[str] = Field(default_factory=list, description="Improvement suggestions")

    @computed_field
    @property
    def overall_score(self) -> float:
        """Compute weighted overall score from available metrics.

        Weights:
        - QA F1 score: 30%
        - Entity diversity: 20%
        - Relation density (normalized): 20%
        - Semantic coherence: 30%
        """
        components: list[float] = []
        weights: list[float] = []

        if self.qa_f1_score is not None:
            components.append(self.qa_f1_score)
            weights.append(0.3)

        if self.entity_coverage is not None:
            components.append(self.entity_coverage.entity_diversity)
            weights.append(0.2)

        if self.relation_metrics is not None:
            # Normalize relation_density (assuming max ~3.0)
            normalized_density = min(self.relation_metrics.relation_density / 3.0, 1.0)
            components.append(normalized_density)
            weights.append(0.2)

        if self.semantic_quality is not None:
            components.append(self.semantic_quality.coherence_score)
            weights.append(0.3)

        if not components:
            return 0.0

        # Compute weighted average with normalized weights
        total_weight = sum(weights)
        return sum(c * w for c, w in zip(components, weights, strict=True)) / total_weight

    def save(self, path: str | Path) -> None:
        """Save evaluation report to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EvaluationReport:
        """Load evaluation report from JSON file."""
        return cls.model_validate_json(Path(path).read_text())


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
        ..., description="file_ids of processed transcription records"
    )
