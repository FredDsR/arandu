"""End-to-end tests for the non-answerable batch driver (spec §7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arandu.qa.non_answerable import batch as batch_mod
from arandu.qa.non_answerable.batch import run_generate_non_answerable_batch
from arandu.qa.non_answerable.schemas import NonAnswerableDataset, PerturbationOutput
from arandu.qa.non_answerable.settings import NonAnswerableSettings

if TYPE_CHECKING:
    from pathlib import Path


class _FakeLLM:
    """Echoes a deterministic same-type swap, deriving names from the question."""

    def __init__(self) -> None:
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
        # The seeded questions all contain "Maria"; swap it for "Joana".
        return PerturbationOutput(
            original_entity="Maria",
            entity_type="person",
            replacement_entity="Joana",
            new_question="Em que ano Joana viu a enchente?",
        )


@pytest.fixture
def patched_llm(monkeypatch: pytest.MonkeyPatch) -> _FakeLLM:
    """Replace the batch's LLM client builder with a fake."""
    fake = _FakeLLM()
    monkeypatch.setattr(batch_mod, "_build_llm_client", lambda _settings: fake)
    return fake


def _settings() -> NonAnswerableSettings:
    return NonAnswerableSettings(provider="ollama", seeds_per_bloom=10, rng_seed=7, retry_max=2)


def test_missing_cep_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
        run_generate_non_answerable_batch("nope", base_dir=tmp_path, settings=_settings())


def test_end_to_end_builds_dataset(tmp_path: Path, cep_dir: Path, patched_llm: _FakeLLM) -> None:
    # cep_dir fixture writes under tmp_path/results/run-x/cep/outputs with
    # two validated pairs (both contain "Maria"); base_dir is results/.
    base = tmp_path / "results"
    result = run_generate_non_answerable_batch("run-x", base_dir=base, settings=_settings())

    assert result.seed_count == 2
    assert result.items_built == 2
    assert result.dataset_items == 2
    assert result.success_rate == 1.0
    assert patched_llm.calls == 2

    dataset = NonAnswerableDataset.load(
        base / "run-x" / "non_answerable" / "outputs" / "dataset.json"
    )
    assert len(dataset.items) == 2
    assert all(item.is_answerable is False for item in dataset.items)
    assert all(item.parent_qa_pair_id.startswith("src1:") for item in dataset.items)
    # chunker_id is inherited from the parent QARecordCEP (default "cep_4k").
    assert all(item.chunker_id == "cep_4k" for item in dataset.items)
    # perturbations_per_seed is pinned to 1 in provenance, not configurable.
    assert dataset.perturbations_per_seed == 1


def test_kg_gate_reads_atlas_output_path(
    tmp_path: Path, cep_dir: Path, patched_llm: _FakeLLM
) -> None:
    # The fake always proposes replacement "Joana". Seed a KG graphml at the
    # real path (kg/outputs/atlas_output/kg_graphml/) with a "Joana" node:
    # every swap must now collide on the KG gate, proving _load_kg_nodes
    # reads the atlas_output segment (the buggy path would yield 0 nodes
    # and let every swap through).
    import networkx as nx

    base = tmp_path / "results"
    kg_dir = base / "run-x" / "kg" / "outputs" / "atlas_output" / "kg_graphml"
    kg_dir.mkdir(parents=True)
    graph = nx.DiGraph()
    graph.add_node("n0", label="Joana")
    nx.write_graphml(graph, str(kg_dir / "corpus_graph.graphml"))

    result = run_generate_non_answerable_batch("run-x", base_dir=base, settings=_settings())
    assert result.items_built == 0
    assert result.seeds_failed == 2


def test_resume_skips_completed_seeds(tmp_path: Path, cep_dir: Path, patched_llm: _FakeLLM) -> None:
    base = tmp_path / "results"
    first = run_generate_non_answerable_batch("run-x", base_dir=base, settings=_settings())
    assert first.items_built == 2
    calls_after_first = patched_llm.calls

    second = run_generate_non_answerable_batch("run-x", base_dir=base, settings=_settings())
    assert second.items_built == 0  # nothing newly perturbed
    assert second.dataset_items == 2  # but the dataset is rebuilt from disk
    assert second.seeds_skipped_resumed == 2
    assert patched_llm.calls == calls_after_first  # no new LLM calls on resume
    assert second.success_rate == 1.0


def test_regenerate_clears_and_rebuilds(
    tmp_path: Path, cep_dir: Path, patched_llm: _FakeLLM
) -> None:
    base = tmp_path / "results"
    run_generate_non_answerable_batch("run-x", base_dir=base, settings=_settings())
    calls_after_first = patched_llm.calls

    result = run_generate_non_answerable_batch(
        "run-x", base_dir=base, settings=_settings(), regenerate=True
    )
    assert result.items_built == 2
    assert patched_llm.calls == calls_after_first + 2  # re-perturbed every seed
