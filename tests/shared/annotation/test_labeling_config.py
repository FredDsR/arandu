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


def _instruction_css(instruction: str) -> str:
    """Return the CSS of the instruction fragment's scoped ``<style>`` block."""
    style = ET.fromstring(f"<div>{instruction}</div>").find(".//style")
    assert style is not None, "the instruction carries no style block"
    return str(style.text)


def _css_color_declarations(css: str) -> list[str]:
    """Return the value of every ``color:`` declaration, excluding compounds.

    ``background-color`` and ``border-color`` are not text colour, so the
    lookbehind drops any property that ends in ``-color``. A trailing
    ``!important`` is priority, not colour, so it is stripped before comparison:
    ``inherit !important`` is still inheriting, and ``red !important`` still
    fails.
    """
    values = re.findall(r"(?<![-\w])color\s*:\s*([^;}]+)", css)
    return [re.sub(r"\s*!important$", "", value.strip()) for value in values]


def _assert_theme_agnostic(css: str) -> None:
    """Fail if the CSS names a colour instead of inheriting or tinting.

    The instrument is read under both Label Studio themes. Any literal colour
    picks one of them and makes the other unreadable, which is what happened to
    the first version of the canvas style, so this is asserted mechanically
    rather than by eye: no hex literal anywhere, no opaque ``rgb()``, every
    translucent neutral below full alpha, and ``color`` only ever ``inherit``.
    """
    assert re.search(r"#[0-9a-fA-F]{3,8}", css) is None, "hex colour literal in CSS"
    assert "rgb(" not in css, "opaque rgb() in CSS"
    alphas = [float(alpha) for alpha in re.findall(r"rgba\([^)]*,\s*([0-9.]+)\s*\)", css)]
    assert alphas, "expected translucent neutrals to carry the surfaces"
    assert all(alpha < 1.0 for alpha in alphas), "fully opaque rgba() in CSS"
    assert set(_css_color_declarations(css)) <= {"inherit"}, "CSS sets an explicit text colour"


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

    def test_every_option_hotkey_equals_its_own_score(self, config: str) -> None:
        """The regression that must never come back.

        With no explicit ``hotkey`` Label Studio numbers the options by position,
        and these run 5..1, so key 1 recorded a 5 and key 5 recorded a 1. An
        annotator pressing 5 for "preserves the meaning" stored the most severe
        distortion label, and nothing in the UI said so.
        """
        choices = ET.fromstring(config).find(".//Choices")
        assert choices is not None
        options = list(choices.iter("Choice"))
        assert options
        for option in options:
            score = re.match(r"\d+", str(option.get("value")))
            assert score is not None, option.get("value")
            assert option.get("hotkey") == score.group()

    def test_rationale_is_an_optional_textarea(self, config: str) -> None:
        area = ET.fromstring(config).find(".//TextArea")
        assert area is not None
        assert area.get("name") == "rationale"
        assert area.get("required") in (None, "false")


class TestLayout:
    """The pair sits immediately above the widget, and the ruler is out of the way."""

    def test_the_summary_is_a_panel_inside_a_collapse(self, config: str) -> None:
        collapse = ET.fromstring(config).find(".//Collapse")
        assert collapse is not None
        panels = list(collapse.iter("Panel"))
        assert len(panels) == 1
        assert "resumo" in str(panels[0].get("value"))

    def test_the_collapse_sits_inside_a_wrapper_we_can_style(self, config: str) -> None:
        """`Collapse`/`Panel` take no `className`, and their own chrome is light-theme.

        The only selector that can reach the component's DOM is a wrapper of
        ours around it, so the wrapper is part of the contract, not decoration.
        """
        wrapper = ET.fromstring(config).find("./View[@className='emic-summary-wrap']")
        assert wrapper is not None
        assert wrapper.find("Collapse") is not None

    def test_the_pair_block_is_between_the_summary_and_the_score(self, config: str) -> None:
        root = ET.fromstring(config)
        order = [el.get("className") if el.tag == "View" else el.tag for el in root]
        assert (
            order.index("emic-summary-wrap") < order.index("emic-pair") < order.index("emic-score")
        )

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


class TestStyling:
    """Both surfaces are read under a light and a dark theme, and neither picks one."""

    def test_the_canvas_style_names_no_colour(self, config: str) -> None:
        style = ET.fromstring(config).find("Style")
        assert style is not None
        _assert_theme_agnostic(str(style.text))

    def test_the_canvas_style_keeps_its_layout_duties(self, config: str) -> None:
        style = ET.fromstring(config).find("Style")
        assert style is not None
        css = str(style.text)
        for declaration in ("max-height", "overflow-y: auto", "max-width: 70ch"):
            assert declaration in css

    def test_the_canvas_style_is_scoped_to_our_own_classes(self, config: str) -> None:
        """The summary override is deliberately broad, so its scope is asserted.

        Neutralising every descendant of the wrapper is the only way to reach
        chrome whose class names we cannot read from here. That width is safe
        only while every selector is anchored to a class this config emits, so
        an unanchored rule (or one starting at a bare element or ``*``) fails.
        """
        style = ET.fromstring(config).find("Style")
        assert style is not None
        selectors = [
            part.strip()
            for group in re.findall(r"([^{}]+)\{", str(style.text))
            for part in group.split(",")
        ]
        assert selectors
        for selector in selectors:
            assert selector.startswith(".emic-"), selector

    def test_the_summary_override_neutralises_the_panel_chrome(self, config: str) -> None:
        """Label Studio's `Panel` ships a light surface and text colour of its own."""
        style = ET.fromstring(config).find("Style")
        assert style is not None
        css = str(style.text)
        assert ".emic-summary-wrap * {" in css
        for declaration in ("background: transparent !important", "color: inherit !important"):
            assert declaration in css

    def test_the_instruction_style_names_no_colour(self, instruction: str) -> None:
        _assert_theme_agnostic(_instruction_css(instruction))

    def test_the_instruction_style_is_scoped_to_its_wrapper(self, instruction: str) -> None:
        """Nothing may leak into Label Studio's own UI, which shares the modal."""
        root = ET.fromstring(f"<div>{instruction}</div>")
        wrapper = root.find("./div")
        assert wrapper is not None
        assert wrapper.get("class") == "emic-instructions"
        selectors = re.findall(r"([^{}]+)\{", _instruction_css(instruction))
        assert selectors
        for selector in selectors:
            assert selector.strip().startswith(".emic-instructions"), selector

    @pytest.mark.parametrize("level", ["h2", "h3"])
    def test_a_heading_gets_more_space_above_than_below(self, instruction: str, level: str) -> None:
        """A title must bind to the section it opens, not to the paragraph before it."""
        rule = re.search(
            rf"\.emic-instructions {level} \{{(.*?)\}}", _instruction_css(instruction), re.DOTALL
        )
        assert rule is not None
        margin = re.search(r"margin:\s*([0-9.]+)em\s+\S+\s+([0-9.]+)em", rule.group(1))
        assert margin is not None
        assert float(margin.group(1)) > float(margin.group(2))

    def test_it_still_reads_without_the_style_block(
        self, instruction: str, ruler: dict[str, Any]
    ) -> None:
        """Label Studio may sanitise the block away, so meaning rides on the markup."""
        root = ET.fromstring(f"<div>{instruction}</div>")
        assert len(list(root.iter("h1"))) == 1
        assert len(list(root.iter("h2"))) >= 5
        items = {str(li.text) for ul in root.iter("ul") for li in ul}
        for entry in ruler["scale"]:
            assert f"{entry['score']} - {entry['label']}" in items
        for element in root.iter():
            assert element.get("style") is None, "no meaning may ride on an inline style"


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
        """The wrapper and its scoped style block aside, the markup stays semantic."""
        root = ET.fromstring(f"<div>{instruction}</div>")
        tags = {el.tag for el in root.iter()} - {"div", "style"}
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
