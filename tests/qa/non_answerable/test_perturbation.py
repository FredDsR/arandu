"""Tests for entity-swap perturbation + stratified seeding (spec §7.3, §7.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.qa.non_answerable.perturbation import (
    SeedPair,
    perturb_to_non_answerable,
    stratified_seed_sample,
)
from arandu.qa.non_answerable.schemas import PerturbationOutput
from arandu.shared.llm_client import StructuredOutputError

from .conftest import make_pair, write_cep_record

if TYPE_CHECKING:
    from pathlib import Path


class _FakeLLM:
    """Returns queued PerturbationOutputs (or raises) per generate_structured call."""

    def __init__(self, outputs: list[PerturbationOutput | Exception]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def generate_structured(
        self,
        prompt: str,
        response_model: type[PerturbationOutput],
        *,
        temperature: float = 0.3,
        **kwargs: object,
    ) -> PerturbationOutput:
        self.calls += 1
        result = self._outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _seed(question: str = "Em que ano Maria viu a enchente?") -> SeedPair:
    return SeedPair(
        parent_qa_pair_id="src:chk_00:0",
        source_file_id="src",
        pair=make_pair(question=question),
    )


class _NeverContains:
    def __contains__(self, _: object) -> bool:
        return False


class _AlwaysContains:
    def __contains__(self, _: object) -> bool:
        return True


class TestPerturb:
    def test_valid_swap_builds_item(self) -> None:
        output = PerturbationOutput(
            original_entity="Maria",
            entity_type="person",
            replacement_entity="Joana",
            new_question="Em que ano Joana viu a enchente?",
        )
        llm = _FakeLLM([output])
        item = perturb_to_non_answerable(
            _seed(),
            kg_node_set=set(),
            corpus_index=_NeverContains(),
            llm=llm,
            max_retries=3,
        )
        assert item is not None
        assert item.qa_pair_id == "src:chk_00:0:nonans"
        assert item.parent_qa_pair_id == "src:chk_00:0"
        assert item.is_answerable is False
        assert item.swapped_entity.replacement_entity == "Joana"
        assert llm.calls == 1

    def test_collision_then_success_retries(self) -> None:
        colliding = PerturbationOutput(
            original_entity="Maria",
            entity_type="person",
            replacement_entity="Itaqui",  # rejected by kg_node_set below
            new_question="...",
        )
        good = PerturbationOutput(
            original_entity="Maria",
            entity_type="person",
            replacement_entity="Joana",
            new_question="Em que ano Joana viu a enchente?",
        )
        llm = _FakeLLM([colliding, good])
        item = perturb_to_non_answerable(
            _seed(),
            kg_node_set={"itaqui"},
            corpus_index=_NeverContains(),
            llm=llm,
            max_retries=3,
        )
        assert item is not None
        assert item.swapped_entity.replacement_entity == "Joana"
        assert llm.calls == 2

    def test_all_collisions_returns_none(self) -> None:
        output = PerturbationOutput(
            original_entity="Maria",
            entity_type="person",
            replacement_entity="Joana",
            new_question="...",
        )
        llm = _FakeLLM([output, output, output])
        item = perturb_to_non_answerable(
            _seed(),
            kg_node_set=set(),
            corpus_index=_AlwaysContains(),  # every replacement "in corpus"
            llm=llm,
            max_retries=3,
        )
        assert item is None
        assert llm.calls == 3

    def test_original_entity_not_in_question_rejected(self) -> None:
        output = PerturbationOutput(
            original_entity="Pedro",  # not in the question
            entity_type="person",
            replacement_entity="Joana",
            new_question="...",
        )
        llm = _FakeLLM([output])
        item = perturb_to_non_answerable(
            _seed(),
            kg_node_set=set(),
            corpus_index=_NeverContains(),
            llm=llm,
            max_retries=1,
        )
        assert item is None

    def test_parse_error_counts_as_attempt(self) -> None:
        llm = _FakeLLM([StructuredOutputError("bad json")])
        item = perturb_to_non_answerable(
            _seed(),
            kg_node_set=set(),
            corpus_index=_NeverContains(),
            llm=llm,
            max_retries=1,
        )
        assert item is None
        assert llm.calls == 1


class TestStratifiedSeedSample:
    def test_only_validated_pairs_eligible(self, tmp_path: Path) -> None:
        cep = tmp_path / "cep"
        write_cep_record(
            cep,
            source_file_id="s1",
            pairs=[
                make_pair(question="q1 Maria?", bloom_level="remember", validated=True),
                make_pair(question="q2 Joao?", bloom_level="remember", validated=False),
            ],
        )
        seeds = stratified_seed_sample(cep, seeds_per_bloom=10, rng_seed=1)
        assert len(seeds) == 1
        assert seeds[0].pair.question == "q1 Maria?"

    def test_caps_per_bloom_and_is_reproducible(self, tmp_path: Path) -> None:
        cep = tmp_path / "cep"
        pairs = [
            make_pair(question=f"q{i} Maria?", bloom_level="remember", chunk_id=f"c{i}")
            for i in range(5)
        ]
        write_cep_record(cep, source_file_id="s1", pairs=pairs)
        first = stratified_seed_sample(cep, seeds_per_bloom=3, rng_seed=42)
        second = stratified_seed_sample(cep, seeds_per_bloom=3, rng_seed=42)
        assert len(first) == 3
        assert [s.parent_qa_pair_id for s in first] == [s.parent_qa_pair_id for s in second]

    def test_parent_id_construction(self, tmp_path: Path) -> None:
        cep = tmp_path / "cep"
        write_cep_record(
            cep,
            source_file_id="s1",
            pairs=[make_pair(question="q Maria?", chunk_id="chk_07")],
        )
        seeds = stratified_seed_sample(cep, seeds_per_bloom=10, rng_seed=1)
        assert seeds[0].parent_qa_pair_id == "s1:chk_07:0"
