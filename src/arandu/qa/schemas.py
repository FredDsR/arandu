"""QA and CEP pipeline Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from arandu.shared.judge.schemas import JudgeResultMixin
from arandu.shared.schemas import BloomLevel, SourceMetadata  # noqa: TC001

if TYPE_CHECKING:
    from typing import Self


class QAPair(BaseModel):
    """Represents a single question-answer pair generated from a transcription.

    The answer must be extractive from the context (contained within it).
    """

    question: str = Field(..., min_length=1, description="The generated question")
    answer: str = Field(
        ..., min_length=1, description="Ground truth answer (extractive from context)"
    )
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


# Bloom's Taxonomy levels for cognitive scaffolding
# BloomLevel is imported from arandu.shared.schemas


class QAPairCEP(QAPair, JudgeResultMixin):
    """Extended QA pair with CEP cognitive elicitation fields.

    Adds Bloom's taxonomy level, reasoning traces, and tacit knowledge inference
    to the base QAPair for cognitive scaffolding-based QA generation. Mixes in
    judge verdict fields so any pair can carry a ``validation`` result once
    a judge has run.
    """

    bloom_level: BloomLevel = Field(
        ..., description="Bloom's taxonomy cognitive level for this question"
    )
    reasoning_trace: str | None = Field(
        None, description="Logical connections between facts leading to the answer"
    )

    @field_validator("reasoning_trace", mode="before")
    @classmethod
    def coerce_reasoning_trace(cls, v: str | list[str] | None) -> str | None:
        """Coerce reasoning_trace from list to joined string."""
        if isinstance(v, list):
            return " -> ".join(str(item) for item in v)
        return v

    is_multi_hop: bool = Field(
        default=False, description="Whether answer requires connecting distant text parts"
    )
    hop_count: int | None = Field(
        None, ge=1, le=5, description="Number of reasoning hops if multi-hop"
    )
    tacit_inference: str | None = Field(
        None, description="Explanation of implicit/tacit knowledge used in the answer"
    )
    generation_prompt: str | None = Field(
        None, description="LLM prompt used to generate this QA pair"
    )
    generation_thinking: str | None = Field(
        None, description="Model thinking/reasoning trace for this specific QA pair"
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


class QARecordCEP(BaseModel):
    """Extended QA dataset record with CEP metadata and validation summary.

    Contains Bloom-scaffolded QA pairs with optional LLM-as-a-Judge validation.
    """

    model_config = {"populate_by_name": True}

    source_file_id: str = Field(
        ..., alias="source_gdrive_id", description="Unique ID of original media file"
    )
    source_filename: str = Field(..., description="Original filename")
    source_metadata: SourceMetadata | None = Field(
        default=None, description="Source interview metadata for provenance tracking"
    )
    transcription_text: str = Field(..., description="Full transcription text")
    qa_pairs: list[QAPairCEP] = Field(..., description="List of CEP-enhanced QA pairs")
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
