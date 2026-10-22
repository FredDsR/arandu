"""Tests for the arm-set and pooling rules of ``analysis.analyze_judge_retest``.

The pooled answer-agreement coefficients the dissertation reports depend on
which arms enter each stratum, and every table relies on one arm set per run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analysis.analyze_judge_retest import ARMS, excluded_from_pool, pool_keys, run_arms

if TYPE_CHECKING:
    from pathlib import Path

CORE_ARMS = ["null", "bm25", "khop_passage", "khop_triple", "atlas_rag"]


def _rep(*arms: str) -> dict:
    """A replica whose answer records cover ``arms`` (values are never read)."""
    return {"answers": {"records": {(arm, f"f:c:{i}"): None for i, arm in enumerate(arms)}}}


class TestRunArms:
    def test_run_without_oracle_lists_core_arms_only(self) -> None:
        assert run_arms([_rep("bm25", "null")]) == CORE_ARMS

    def test_core_arm_missing_from_every_replica_is_still_listed(self) -> None:
        # conclusions() indexes bm25 and the graph arms unconditionally.
        assert run_arms([_rep("null")]) == CORE_ARMS

    def test_oracle_in_any_replica_is_listed_for_all(self) -> None:
        # Mixed rounds (one replica predates the oracle) must share one arm set.
        assert run_arms([_rep("bm25"), _rep("bm25", "oracle")]) == ARMS

    def test_order_follows_arms(self) -> None:
        assert run_arms([_rep("oracle", "atlas_rag")]) == ARMS


class TestPooling:
    def test_null_never_pooled(self) -> None:
        for crit in ("answer_correctness", "passage_coverage", None):
            assert excluded_from_pool("null", crit)

    def test_oracle_pooled_except_for_coverage(self) -> None:
        assert not excluded_from_pool("oracle", "answer_correctness")
        assert not excluded_from_pool("oracle", "answer_faithfulness")
        assert not excluded_from_pool("oracle", None)
        assert excluded_from_pool("oracle", "passage_coverage")
        assert excluded_from_pool("oracle", "passage_coverage_pass")

    def test_retrieval_arms_always_pooled(self) -> None:
        for arm in ("bm25", "khop_passage", "khop_triple", "atlas_rag"):
            assert not excluded_from_pool(arm, "passage_coverage")

    def test_pool_keys_filters_per_criterion(self) -> None:
        keys = [("null", "a"), ("bm25", "a"), ("oracle", "a")]
        assert pool_keys(keys, "answer_correctness") == [("bm25", "a"), ("oracle", "a")]
        assert pool_keys(keys, "passage_coverage") == [("bm25", "a")]


class TestKgCollisionProbes:
    def test_flags_probe_whose_replacement_is_an_atlas_label(self, tmp_path: Path) -> None:
        import networkx as nx

        from analysis.analyze_judge_retest import kg_collision_probes

        graph = nx.DiGraph()
        # ATLAS export: hashed node id, surface text in the ``id`` attribute.
        graph.add_node("8c94882" + "0" * 57, id="Entrevistado", type="entity")
        graphml = tmp_path / "kg.graphml"
        nx.write_graphml(graph, str(graphml))

        def item(pid: str, replacement: str) -> dict:
            return {
                "qa_pair_id": f"{pid}:nonans",
                "question": "Q?",
                "bloom_level": "understand",
                "question_type": "conceptual",
                "source_file_id": "f",
                "chunker_id": "cep_4k",
                "parent_qa_pair_id": pid,
                "swapped_entity": {
                    "original_entity": "entrevistada",
                    "replacement_entity": replacement,
                    "entity_type": "pessoa",
                },
            }

        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "items": [item("f:c:1", " entrevistado "), item("f:c:2", "Joana")],
                    "seed_cep_dataset": "x",
                    "kg_artifact": str(graphml),
                    "seed_count": 2,
                    "perturbations_per_seed": 1,
                    "success_rate": 1.0,
                    "rng_seed": 42,
                }
            )
        )
        assert kg_collision_probes(dataset, graphml) == frozenset({"f:c:1:nonans"})
