"""Tests for non-answerable schemas (spec §7.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from arandu.qa.non_answerable.schemas import (
    NonAnswerableDataset,
    NonAnswerableItem,
    SwapRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def _item(qa_pair_id: str = "src:chk:0:nonans") -> NonAnswerableItem:
    return NonAnswerableItem(
        qa_pair_id=qa_pair_id,
        question="Em que ano Joana viu a enchente?",
        bloom_level="remember",
        question_type="factual",
        source_file_id="src",
        chunker_id="cep_4k",
        parent_qa_pair_id="src:chk:0",
        swapped_entity=SwapRecord(
            original_entity="Maria", replacement_entity="Joana", entity_type="person"
        ),
    )


class TestNonAnswerableItem:
    """Tests for the NonAnswerableItem invariants."""

    def test_is_answerable_pinned_false(self) -> None:
        assert _item().is_answerable is False

    def test_cannot_override_is_answerable_true(self) -> None:
        with pytest.raises(ValidationError):
            _item().model_copy(update={"is_answerable": True}).model_validate(
                _item().model_dump() | {"is_answerable": True}
            )

    def test_perturbation_method_defaults(self) -> None:
        assert _item().perturbation_method == "entity_swap_llm"

    def test_unjudged_by_default(self) -> None:
        assert _item().validation is None
        assert _item().is_valid is None


class TestNonAnswerableDataset:
    """Tests for NonAnswerableDataset round-trip + field bounds."""

    def test_round_trip(self, tmp_path: Path) -> None:
        dataset = NonAnswerableDataset(
            items=[_item()],
            seed_cep_dataset="cep/outputs",
            kg_artifact="kg/outputs/kg_graphml",
            seed_count=1,
            perturbations_per_seed=1,
            success_rate=1.0,
            rng_seed=42,
        )
        path = tmp_path / "dataset.json"
        dataset.save(path)
        loaded = NonAnswerableDataset.load(path)
        assert loaded.seed_count == 1
        assert loaded.items[0].qa_pair_id == "src:chk:0:nonans"
        assert loaded.success_rate == 1.0

    def test_success_rate_bounds(self) -> None:
        with pytest.raises(ValidationError):
            NonAnswerableDataset(
                items=[],
                seed_cep_dataset="x",
                kg_artifact="y",
                seed_count=0,
                perturbations_per_seed=1,
                success_rate=1.5,
                rng_seed=42,
            )
