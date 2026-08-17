"""The labeling config carries the ruler verbatim and nothing that primes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest

from arandu.shared.annotation.labeling_config import render_labeling_config
from arandu.shared.annotation.ruler import load_ruler


@pytest.fixture
def ruler() -> dict[str, Any]:
    return load_ruler()


@pytest.fixture
def config(ruler: dict[str, Any]) -> str:
    return render_labeling_config(ruler)


class TestStructure:
    def test_is_well_formed_xml_rooted_at_view(self, config: str) -> None:
        assert ET.fromstring(config).tag == "View"

    def test_exposes_the_three_blinded_fields_only(self, config: str) -> None:
        """`Text` is reserved for bound task variables; ruler prose uses `Header`."""
        root = ET.fromstring(config)
        values = {el.get("value") for el in root.iter("Text")}
        assert values == {"$segment", "$question", "$answer"}

    def test_no_static_element_binds_a_variable(self, config: str) -> None:
        root = ET.fromstring(config)
        for el in root.iter("Header"):
            assert not str(el.get("value", "")).startswith("$")

    def test_element_names_are_unique(self, config: str) -> None:
        """Duplicate `name` attributes make Label Studio reject the config."""
        names = [
            el.get("name") for el in ET.fromstring(config).iter() if el.get("name") is not None
        ]
        assert len(names) == len(set(names))

    def test_score_is_a_single_radio_with_shuffle_off(self, config: str) -> None:
        choices = ET.fromstring(config).find(".//Choices")
        assert choices is not None
        assert choices.get("name") == "score"
        assert choices.get("choice") == "single-radio"
        assert choices.get("shuffle") == "false"
        assert choices.get("required") == "true"

    def test_options_are_descending_anchors(self, config: str, ruler: dict[str, Any]) -> None:
        choices = ET.fromstring(config).find(".//Choices")
        assert choices is not None
        rendered = [el.get("value") for el in choices.iter("Choice")]
        expected = [f"{e['score']} - {e['label']}" for e in ruler["scale"]]
        assert rendered == expected
        assert [v.split(" - ")[0] for v in rendered] == ["5", "4", "3", "2", "1"]

    def test_rationale_is_an_optional_textarea(self, config: str) -> None:
        area = ET.fromstring(config).find(".//TextArea")
        assert area is not None
        assert area.get("name") == "rationale"
        assert area.get("required") in (None, "false")


class TestRulerFidelity:
    def test_every_scale_anchor_appears_verbatim(self, config: str, ruler: dict[str, Any]) -> None:
        for entry in ruler["scale"]:
            assert entry["label"] in config

    def test_construct_and_guide_text_appear_verbatim(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        for text in ruler["construct"].values():
            assert text in config
        for key in ("locate", "extract", "cascade_intro", "no_penalty"):
            assert ruler["guide"][key] in config
        for entry in ruler["guide"]["cascade"]:
            assert entry["condition"] in config

    def test_loss_types_appear_verbatim(self, config: str, ruler: dict[str, Any]) -> None:
        for entry in ruler["loss_types"]:
            assert entry["name"] in config
            assert entry["what"] in config

    def test_annotator_only_calibration_is_included(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        for text in ruler["annotator_only"].values():
            assert text in config

    def test_provisions_appear_verbatim(self, config: str, ruler: dict[str, Any]) -> None:
        provisions = ruler["provisions"]
        for key in ("unit_of_judgment", "question_calibrates", "out_of_scope"):
            assert provisions[key] in config
        for text in provisions["not_a_loss"]:
            assert text in config

    def test_guide_name_the_condition_appears_verbatim(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        assert ruler["guide"]["name_the_condition"] in config


class TestBlinding:
    def test_judge_only_text_never_reaches_the_annotator(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        judge_only = ruler["judge_only"]
        assert judge_only["role"] not in config
        assert judge_only["output"] not in config
        for rule in judge_only["rationale_rules"]:
            assert rule not in config

    @pytest.mark.parametrize(
        "leak", ["$pair_id", "$bloom_level", "$emic_score", "$cell_id", "$source_file_id"]
    )
    def test_no_stratification_variable_is_bound(self, config: str, leak: str) -> None:
        assert leak not in config
