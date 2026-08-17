"""The two surfaces carry the ruler verbatim between them, and nothing that primes.

The ruler used to be rendered whole onto the labeling canvas, which put two
screens of prose between the pair and the rating widget. It is now split: the
full text goes to Label Studio's instructions modal, and the canvas keeps a
collapsed summary. Fidelity is therefore asserted against the UNION of the two
surfaces, never against one of them alone; the placements that are themselves
load-bearing (the anchors on the options, the cascade in the summary) are
asserted individually on top of that.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any

import pytest

from arandu.shared.annotation.labeling_config import (
    render_expert_instruction,
    render_labeling_config,
)
from arandu.shared.annotation.ruler import load_ruler


@pytest.fixture
def ruler() -> dict[str, Any]:
    return load_ruler()


@pytest.fixture
def config(ruler: dict[str, Any]) -> str:
    return render_labeling_config(ruler)


@pytest.fixture
def instruction(ruler: dict[str, Any]) -> str:
    return render_expert_instruction(ruler)


def _decoded_config(config: str) -> str:
    """Return every attribute value and text node of the config, decoded.

    The config carries ruler prose in XML attributes, so a passage holding a
    quote or an ampersand is stored escaped. Comparing against the decoded
    values asserts what the annotator reads rather than how it was serialised.
    """
    root = ET.fromstring(config)
    parts: list[str] = []
    for element in root.iter():
        parts.extend(str(value) for value in element.attrib.values())
        if element.text:
            parts.append(element.text)
    return "\n".join(parts)


@pytest.fixture
def surfaces(config: str, instruction: str) -> str:
    """The union of both annotator-facing surfaces, decoded."""
    return _decoded_config(config) + "\n" + unescape(instruction)


def _ruler_passages(ruler: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every annotator-facing ruler passage as ``(where, text)``.

    Iterated from the ruler itself: a new key, cascade step, loss type or
    ``not_a_loss`` entry is covered the moment it is added, with no count or
    literal to update here.
    """
    guide = ruler["guide"]
    provisions = ruler["provisions"]
    passages: list[tuple[str, str]] = []
    passages += [(f"construct.{key}", text) for key, text in ruler["construct"].items()]
    passages += [(f"scale[{entry['score']}]", entry["label"]) for entry in ruler["scale"]]
    for entry in ruler["loss_types"]:
        passages.append((f"loss_types[{entry['name']}].name", entry["name"]))
        passages.append((f"loss_types[{entry['name']}].what", entry["what"]))
    passages += [
        (f"guide.{key}", guide[key])
        for key in ("locate", "extract", "cascade_intro", "no_penalty", "name_the_condition")
    ]
    passages += [
        (f"guide.cascade[{entry['score']}]", entry["condition"]) for entry in guide["cascade"]
    ]
    passages += [
        (f"provisions.{key}", provisions[key])
        for key in ("unit_of_judgment", "question_calibrates", "out_of_scope")
    ]
    passages += [
        (f"provisions.not_a_loss[{index}]", text)
        for index, text in enumerate(provisions["not_a_loss"])
    ]
    passages += [(f"annotator_only.{key}", text) for key, text in ruler["annotator_only"].items()]
    return passages


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
        assert [str(v).split(" - ")[0] for v in rendered] == ["5", "4", "3", "2", "1"]

    def test_rationale_is_an_optional_textarea(self, config: str) -> None:
        area = ET.fromstring(config).find(".//TextArea")
        assert area is not None
        assert area.get("name") == "rationale"
        assert area.get("required") in (None, "false")


class TestLayout:
    """The pair sits immediately above the widget, and the ruler is out of the way."""

    def test_the_summary_is_a_panel_inside_a_collapse(self, config: str) -> None:
        collapse = ET.fromstring(config).find("Collapse")
        assert collapse is not None
        panels = list(collapse.iter("Panel"))
        assert len(panels) == 1
        assert "resumo" in str(panels[0].get("value"))

    def test_the_pair_block_is_between_the_summary_and_the_score(self, config: str) -> None:
        root = ET.fromstring(config)
        order = [el.get("className") if el.tag == "View" else el.tag for el in root]
        assert order.index("Collapse") < order.index("emic-pair") < order.index("emic-score")

    def test_the_score_widget_and_the_pair_are_adjacent(self, config: str) -> None:
        """Nothing may be inserted between them; that regression is the whole bug."""
        root = ET.fromstring(config)
        order = [el.get("className") if el.tag == "View" else el.tag for el in root]
        assert order.index("emic-score") == order.index("emic-pair") + 1

    def test_the_pair_block_holds_all_three_bound_fields(self, config: str) -> None:
        pair = ET.fromstring(config).find("./View[@className='emic-pair']")
        assert pair is not None
        assert {el.get("value") for el in pair.iter("Text")} == {
            "$segment",
            "$question",
            "$answer",
        }

    def test_the_segment_is_wrapped_in_its_own_capped_box(self, config: str) -> None:
        """A 4000-character chunk must scroll inside itself, not push the widget away."""
        root = ET.fromstring(config)
        box = root.find(".//View[@className='emic-segment']")
        assert box is not None
        assert [el.get("value") for el in box.iter("Text")] == ["$segment"]
        style = root.find("Style")
        assert style is not None
        css = str(style.text)
        assert "max-height" in css
        assert "overflow-y: auto" in css


class TestRulerFidelity:
    def test_every_passage_appears_verbatim_across_the_two_surfaces(
        self, surfaces: str, ruler: dict[str, Any]
    ) -> None:
        missing = [where for where, text in _ruler_passages(ruler) if text not in surfaces]
        assert missing == []

    def test_the_scale_anchors_are_on_the_options_themselves(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        """The fidelity-critical placement: the annotator picks the anchor, not a number."""
        choices = ET.fromstring(config).find(".//Choices")
        assert choices is not None
        options = [str(el.get("value")) for el in choices.iter("Choice")]
        for entry in ruler["scale"]:
            assert any(entry["label"] in option for option in options)

    def test_the_cascade_is_in_the_canvas_summary(self, config: str, ruler: dict[str, Any]) -> None:
        summary = _decoded_config(config)
        assert ruler["guide"]["cascade_intro"] in summary
        for entry in ruler["guide"]["cascade"]:
            assert entry["condition"] in summary

    def test_the_no_penalty_list_is_in_the_canvas_summary(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        assert ruler["guide"]["no_penalty"] in _decoded_config(config)

    def test_the_annotator_calibration_is_in_the_canvas_summary(
        self, config: str, ruler: dict[str, Any]
    ) -> None:
        summary = _decoded_config(config)
        for text in ruler["annotator_only"].values():
            assert text in summary

    def test_the_full_ruler_is_in_the_instruction(
        self, instruction: str, ruler: dict[str, Any]
    ) -> None:
        text = unescape(instruction)
        provisions = ruler["provisions"]
        for passage in ruler["construct"].values():
            assert passage in text
        for key in ("unit_of_judgment", "question_calibrates", "out_of_scope"):
            assert provisions[key] in text
        for entry in provisions["not_a_loss"]:
            assert entry in text
        for entry in ruler["loss_types"]:
            assert entry["name"] in text
            assert entry["what"] in text
        for key, passage in ruler["guide"].items():
            if isinstance(passage, str):
                assert passage in text, key
        for entry in ruler["guide"]["cascade"]:
            assert entry["condition"] in text
        for passage in ruler["annotator_only"].values():
            assert passage in text


class TestInstructionHtml:
    def test_is_well_formed(self, instruction: str) -> None:
        """It is a fragment, so it is parsed under a synthetic root."""
        assert ET.fromstring(f"<div>{instruction}</div>") is not None

    def test_uses_plain_semantic_elements(self, instruction: str) -> None:
        root = ET.fromstring(f"<div>{instruction}</div>")
        tags = {el.tag for el in root.iter()} - {"div"}
        assert tags <= {"h1", "h2", "h3", "p", "ul", "li"}

    def test_carries_no_script(self, instruction: str) -> None:
        assert "<script" not in instruction.lower()

    def test_references_nothing_off_host(self, instruction: str) -> None:
        """A research instrument must not depend on a resource that can move."""
        targets = re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', instruction)
        assert [t for t in targets if re.match(r"^(?:[a-zA-Z][\w+.-]*:)?//", t)] == []

    def test_escapes_its_ruler_text(self, ruler: dict[str, Any]) -> None:
        rendered = render_expert_instruction(
            {**ruler, "annotator_only": {"x": 'a "quote" and <b> and &'}}
        )
        assert "&quot;quote&quot;" in rendered
        assert "&lt;b&gt;" in rendered
        assert ET.fromstring(f"<div>{rendered}</div>") is not None


class TestBlinding:
    def test_judge_only_text_reaches_neither_surface(
        self, config: str, instruction: str, surfaces: str, ruler: dict[str, Any]
    ) -> None:
        judge_only = ruler["judge_only"]
        forbidden = [judge_only["role"], judge_only["output"], *judge_only["rationale_rules"]]
        for text in forbidden:
            assert text not in surfaces
            assert text not in config
            assert text not in instruction

    @pytest.mark.parametrize(
        "leak", ["$pair_id", "$bloom_level", "$emic_score", "$cell_id", "$source_file_id"]
    )
    def test_no_stratification_variable_is_bound(
        self, config: str, instruction: str, leak: str
    ) -> None:
        assert leak not in config
        assert leak not in instruction
