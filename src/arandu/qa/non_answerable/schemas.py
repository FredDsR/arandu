"""Schemas for the non-answerable benchmark (spec §7.2).

- :class:`PerturbationOutput` - structured LLM response from one swap call.
- :class:`SwapRecord` - the persisted entity-swap provenance.
- :class:`NonAnswerableItem` - one non-answerable QA item (twin of a CEP pair).
- :class:`NonAnswerableDataset` - the full dataset + generation provenance.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from arandu.shared.judge.schemas import JudgeResultMixin
from arandu.shared.schemas import BloomLevel  # noqa: TC001

if TYPE_CHECKING:
    from typing import Self

QuestionType = Literal["factual", "conceptual", "temporal", "entity"]


class PerturbationOutput(BaseModel):
    """Structured output from a single LLM perturbation call (spec §7.4)."""

    original_entity: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    replacement_entity: str = Field(..., min_length=1)
    new_question: str = Field(..., min_length=1)


class SwapRecord(BaseModel):
    """Provenance of the entity swap that produced a non-answerable item."""

    original_entity: str = Field(..., min_length=1)
    replacement_entity: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)


class NonAnswerableItem(JudgeResultMixin):
    """A non-answerable QA item derived from a validated CEP pair.

    Shares the ``qa_pair_id`` namespace with :class:`QAPairCEP` (the
    answerable twin) so retrieval, answering, and judging treat it like
    any other benchmark item. ``is_answerable`` is pinned ``False`` so
    the analysis stage classifies it into the TA / FC column.
    """

    qa_pair_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    bloom_level: BloomLevel
    question_type: QuestionType
    source_file_id: str = Field(..., min_length=1)
    chunker_id: str = Field(..., min_length=1)
    parent_qa_pair_id: str = Field(..., min_length=1)
    perturbation_method: Literal["entity_swap_llm"] = "entity_swap_llm"
    swapped_entity: SwapRecord
    is_answerable: Literal[False] = False


class NonAnswerableDataset(BaseModel):
    """Full non-answerable benchmark plus generation provenance."""

    items: list[NonAnswerableItem]
    seed_cep_dataset: str
    kg_artifact: str
    seed_count: int = Field(..., ge=0)
    perturbations_per_seed: int = Field(..., ge=1)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    rng_seed: int
    generated_at: datetime = Field(default_factory=datetime.now)

    def save(self, path: str | Path) -> None:
        """Serialize this dataset to ``path`` as JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a dataset from ``path``."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
