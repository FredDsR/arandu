"""QA and CEP pipeline Pydantic schemas."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from arandu.shared.judge.schemas import JudgeResultMixin
from arandu.shared.schemas import BloomLevel, SourceMetadata  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

logger = logging.getLogger(__name__)


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
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deprecated generation confidence score; no longer emitted by CEP generation.",
    )
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

    Adds the Bloom's taxonomy level and the generation rationale to the base
    QAPair for cognitive scaffolding-based QA generation. The ``reasoning_trace``,
    ``is_multi_hop``, ``hop_count`` and ``tacit_inference`` fields are legacy:
    only runs made before the reasoning-enrichment pass was removed fill them,
    and they stay so those runs still load. Mixes in
    judge verdict fields so any pair can carry a ``validation`` result once
    a judge has run.
    """

    bloom_level: BloomLevel = Field(
        ..., description="Bloom's taxonomy cognitive level for this question"
    )
    rationale: str | None = Field(
        None,
        description=(
            "Generator's reasoning for how/why it constructed this Q-A pair from the context "
            "(what prompted the question, how the answer is grounded, why it fits the Bloom "
            "level). Generation-process aid; never a judge input; not a claim about the "
            "knowledge itself."
        ),
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
    chunk_id: str | None = Field(
        default=None,
        description="Reference into the source ChunkSet for the chunker view used at generation",
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
    source_metadata_context_enabled: bool = Field(
        default=True,
        description=(
            "Whether source-metadata context was injected into the generation "
            "prompt for this record. The judge reads this to inject the same "
            "metadata grounding, keeping generation and judging symmetric "
            "regardless of judge-time config."
        ),
    )
    scaffolding_gate_enabled: bool = Field(
        default=False,
        description=(
            "Whether every pair of this record was judged before it could scaffold "
            "later pairs (CEP scaffolding gate). When True, pair verdicts in "
            "``validation`` were produced at generation time. Defaults to False so "
            "records written before the gate existed load correctly."
        ),
    )
    transcription_text: str = Field(..., description="Full transcription text")
    qa_pairs: list[QAPairCEP] = Field(..., description="List of CEP-enhanced QA pairs")
    chunker_id: str = Field(
        default="cep_4k",
        description="Identifier of the chunker view used to slice transcription_text",
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


def cep_qa_pair_id(source_file_id: str, chunk_id: str | None, idx: int) -> str:
    """Build the composite ``qa_pair_id`` of the ``idx``-th pair of a CEP record.

    Args:
        source_file_id: The record's source media file id.
        chunk_id: The pair's chunk id; ``None`` renders as ``"none"``.
        idx: The pair's position among ALL pairs of the record.

    Returns:
        ``"<file_id>:<chunk_id or 'none'>:<idx>"``.
    """
    return f"{source_file_id}:{chunk_id or 'none'}:{idx}"


class CEPPairRef(BaseModel):
    """One pair of a CEP record, addressed by its composite ``qa_pair_id``.

    Attributes:
        qa_pair_id: The pair's id, from :func:`cep_qa_pair_id`.
        idx: The pair's position among ALL pairs of its record.
        record: The record the pair belongs to.
        pair: The pair itself.
    """

    model_config = {"frozen": True}

    qa_pair_id: str
    idx: int
    record: QARecordCEP
    pair: QAPairCEP


def iter_cep_pairs(cep_dir: Path, *, strict: bool = False) -> Iterator[CEPPairRef]:
    """Yield every pair of every readable CEP record under ``cep_dir``.

    The single walk over ``results/<id>/cep/outputs/``: files in sorted
    order, pairs in record order, judge-rejected pairs included. Callers apply
    their own filters (``is_judge_rejected``, ``validation.passed``) and their
    own policy for a missing directory, which yields nothing here.

    Args:
        cep_dir: Directory of :class:`QARecordCEP` JSON files.
        strict: Raise on an unreadable or invalid file instead of skipping it.
            For callers that cache or publish a result, where a silently
            incomplete pair set is worse than a failed run.

    Yields:
        One :class:`CEPPairRef` per pair. Unless ``strict``, unreadable or
        invalid files are logged and skipped. A file whose ids repeat ids of an
        earlier file (two records with the same ``source_file_id``) is logged;
        its pairs are still yielded, and id-keyed consumers keep the later one.

    Raises:
        OSError: With ``strict``, if a file cannot be read.
        ValueError: With ``strict``, if a file is not valid UTF-8 or not a
            valid :class:`QARecordCEP` (``ValidationError`` is a subclass).
    """
    seen: set[str] = set()
    for path in sorted(cep_dir.glob("*.json")):
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # ValueError covers ValidationError and decode
            if strict:
                raise
            logger.warning("Skipping unreadable CEP file %s: %s", path, exc)
            continue
        ids = [
            cep_qa_pair_id(record.source_file_id, p.chunk_id, i)
            for i, p in enumerate(record.qa_pairs)
        ]
        repeated = seen.intersection(ids)
        if repeated:
            logger.warning(
                "CEP file %s repeats %d qa_pair_ids of an earlier file "
                "(duplicate source_file_id %r?); id-keyed lookups keep this file's pairs.",
                path,
                len(repeated),
                record.source_file_id,
            )
        seen.update(ids)
        for idx, (qa_pair_id, pair) in enumerate(zip(ids, record.qa_pairs, strict=True)):
            yield CEPPairRef(qa_pair_id=qa_pair_id, idx=idx, record=record, pair=pair)
