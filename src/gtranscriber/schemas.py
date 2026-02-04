"""Pydantic schemas for G-Transcriber input and output data validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

if TYPE_CHECKING:
    from typing import Self


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
    duration_milliseconds: int | None = Field(
        None, description="Media duration in milliseconds (for audio/video files)"
    )

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

    @field_validator("size_bytes", mode="before")
    @classmethod
    def parse_size_bytes(cls, v: str | int | None) -> int | None:
        """Parse size_bytes from string or int format."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return None
        return None


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

    def ensure_language_metadata(self) -> None:
        """Ensure metadata compatibility for AutoSchemaKG language routing.

        AutoSchemaKG uses a 'lang' field in metadata for language-specific prompts.
        This method can be called to ensure compatibility if needed.
        """
        # Note: This is a placeholder for future metadata handling
        # The detected_language field already provides the language code
        pass


# =============================================================================
# QA Generation Schemas
# =============================================================================


class QAPair(BaseModel):
    """Represents a single question-answer pair generated from a transcription.

    The answer must be extractive from the context (contained within it).
    """

    question: str = Field(..., description="The generated question")
    answer: str = Field(..., description="Ground truth answer (extractive from context)")
    context: str = Field(..., description="Source text segment from which QA was generated")
    question_type: Literal["factual", "conceptual", "temporal", "entity"] = Field(
        ..., description="Question generation strategy type"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Generation confidence score")
    start_time: float | None = Field(None, description="Segment start time in seconds")
    end_time: float | None = Field(None, description="Segment end time in seconds")

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Validate temporal constraints for start/end times."""
        if self.start_time is not None and self.end_time is None:
            raise ValueError("end_time required when start_time is provided")
        if self.end_time is not None and self.start_time is None:
            raise ValueError("start_time required when end_time is provided")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError("start_time must be less than end_time")
        return self


class QARecord(BaseModel):
    """Represents the complete QA dataset for a single transcription document."""

    source_gdrive_id: str = Field(..., description="Google Drive ID of original media file")
    source_filename: str = Field(..., description="Original filename")
    transcription_text: str = Field(..., description="Full transcription text")
    qa_pairs: list[QAPair] = Field(..., description="List of generated QA pairs")
    model_id: str = Field(..., description="LLM model used for generation")
    provider: Literal["openai", "ollama", "custom"] = Field(..., description="LLM provider used")
    language: str = Field(
        default="pt", description="Language used for prompt generation (ISO 639-1)"
    )
    generation_timestamp: datetime = Field(
        default_factory=datetime.now, description="When QA pairs were generated"
    )
    total_pairs: int = Field(..., description="Total number of QA pairs generated")

    @model_validator(mode="after")
    def validate_total_pairs(self) -> Self:
        """Validate that total_pairs matches actual count."""
        if self.total_pairs != len(self.qa_pairs):
            raise ValueError(
                f"total_pairs ({self.total_pairs}) must equal len(qa_pairs) ({len(self.qa_pairs)})"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save QA record to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> QARecord:
        """Load QA record from JSON file."""
        return cls.model_validate_json(Path(path).read_text())


# =============================================================================
# CEP (Cognitive Elicitation Pipeline) Schemas
# =============================================================================
# Cognitive scaffolding QA generation based on Bloom's Taxonomy with
# LLM-as-a-Judge validation.

# Bloom's Taxonomy levels for cognitive scaffolding
BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]


class QAPairCEP(QAPair):
    """Extended QA pair with CEP cognitive elicitation fields.

    Adds Bloom's taxonomy level, reasoning traces, and tacit knowledge inference
    to the base QAPair for cognitive scaffolding-based QA generation.
    """

    bloom_level: BloomLevel = Field(
        ..., description="Bloom's taxonomy cognitive level for this question"
    )
    reasoning_trace: str | None = Field(
        None, description="Logical connections between facts leading to the answer"
    )
    is_multi_hop: bool = Field(
        default=False, description="Whether answer requires connecting distant text parts"
    )
    hop_count: int | None = Field(
        None, ge=1, le=5, description="Number of reasoning hops if multi-hop"
    )
    tacit_inference: str | None = Field(
        None, description="Explanation of implicit/tacit knowledge used in the answer"
    )

    @model_validator(mode="after")
    def validate_multi_hop(self) -> Self:
        """Validate hop_count is set when is_multi_hop is True."""
        if self.is_multi_hop and self.hop_count is None:
            # Default to 2 hops if not specified
            object.__setattr__(self, "hop_count", 2)
        if not self.is_multi_hop and self.hop_count is not None:
            object.__setattr__(self, "hop_count", None)
        return self


class ValidationScore(BaseModel):
    """LLM-as-a-Judge validation scores for a QA pair.

    Evaluates faithfulness (grounding), Bloom calibration, and informativeness.
    """

    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Is answer grounded in context (1.0) or hallucinated (0.0)?",
    )
    bloom_calibration: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Does question match the proposed cognitive level?",
    )
    informativeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Does answer reveal non-obvious/tacit knowledge?",
    )
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted average of all scores")
    judge_rationale: str | None = Field(None, description="Judge model's reasoning for the scores")


class QAPairValidated(QAPairCEP):
    """QA pair with LLM-as-a-Judge validation results."""

    validation: ValidationScore | None = Field(
        None, description="LLM-as-a-Judge validation results"
    )
    is_valid: bool = Field(default=True, description="Whether pair passes validation threshold")


class QARecordCEP(BaseModel):
    """Extended QA dataset record with CEP metadata and validation summary.

    Contains Bloom-scaffolded QA pairs with optional LLM-as-a-Judge validation.
    """

    source_gdrive_id: str = Field(..., description="Google Drive ID of original media file")
    source_filename: str = Field(..., description="Original filename")
    transcription_text: str = Field(..., description="Full transcription text")
    # NOTE: QAPairValidated must be listed first in the union. Since it's a subclass
    # of QAPairCEP with optional fields that have defaults, Pydantic will try types
    # in order and QAPairValidated must match first to preserve validation fields.
    qa_pairs: list[QAPairValidated | QAPairCEP] = Field(
        ..., description="List of CEP-enhanced QA pairs"
    )
    model_id: str = Field(..., description="LLM model used for generation")
    validator_model_id: str | None = Field(
        None, description="LLM model used for validation (if enabled)"
    )
    provider: Literal["openai", "ollama", "custom"] = Field(..., description="LLM provider used")
    language: str = Field(default="pt", description="Language for prompts (ISO 639-1)")
    generation_timestamp: datetime = Field(
        default_factory=datetime.now, description="When QA pairs were generated"
    )
    total_pairs: int = Field(..., description="Total number of QA pairs generated")
    validated_pairs: int = Field(default=0, description="Number of pairs passing validation")
    bloom_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count of QA pairs per Bloom level"
    )
    validation_summary: dict[str, float] | None = Field(
        None, description="Aggregated validation metrics"
    )
    cep_version: str = Field(default="1.0", description="CEP pipeline version")

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Validate that total_pairs matches actual count."""
        if self.total_pairs != len(self.qa_pairs):
            raise ValueError(
                f"total_pairs ({self.total_pairs}) must equal len(qa_pairs) ({len(self.qa_pairs)})"
            )
        return self

    @computed_field
    @property
    def validation_rate(self) -> float:
        """Compute the percentage of pairs that passed validation."""
        if self.total_pairs == 0:
            return 0.0
        return self.validated_pairs / self.total_pairs

    def save(self, path: str | Path) -> None:
        """Save CEP QA record to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    def to_jsonl(self, path: str | Path | None = None) -> str | None:
        """Export QA pairs to JSONL format for KGQA training compatibility.

        Args:
            path: Optional path to write JSONL file. If None, returns as string.

        Returns:
            JSONL string if path is None, otherwise None.
        """
        lines = [pair.model_dump_json() for pair in self.qa_pairs]
        jsonl_content = "\n".join(lines)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(jsonl_content + "\n" if jsonl_content else "")
            return None

        return jsonl_content

    @classmethod
    def load(cls, path: str | Path) -> QARecordCEP:
        """Load CEP QA record from JSON file."""
        return cls.model_validate_json(Path(path).read_text())


# =============================================================================
# Knowledge Graph Metadata
# =============================================================================
# Note: Knowledge graphs use AutoSchemaKG's native GraphML output directly.
# NetworkX loads these with nx.read_graphml(). No custom KGNode/KGEdge classes
# are needed. This metadata class is for provenance tracking only.


class KGMetadata(BaseModel):
    """Lightweight metadata for knowledge graph provenance tracking.

    Stored as a JSON sidecar file alongside the GraphML graph file.
    """

    graph_id: str = Field(..., description="Unique graph identifier")
    source_documents: list[str] = Field(..., description="List of source document IDs (gdrive_ids)")
    model_id: str = Field(..., description="LLM model used for extraction")
    provider: str = Field(..., description="LLM provider")
    language: str = Field(default="pt", description="Language code for extraction (ISO 639-1)")
    prompt_path: str = Field(
        default="prompts/pt_prompts.json",
        description="Path to prompt template file used",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="When graph was created")

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
