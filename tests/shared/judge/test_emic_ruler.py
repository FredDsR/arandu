"""The emic ruler is a single source; its consumers must not drift from it.

``prompts/judge/criteria/emic_validity/ruler.pt.yaml`` holds the construct, the
scale, the loss types, the provisions and the guide. Two artifacts render it: the
judge prompt and the annotator instruction sheet. The weighted kappa between the
LLM and each annotator (spec §6) only measures agreement if both score on the
*same* ruler, so a silent divergence between these files would turn the
coefficient into a measure of translation mismatch. That is not recoverable after
annotation, which is why it is asserted here instead of reviewed by eye.

Also asserted: the ruler is signed off, no real corpus pair leaked into either
artifact, and the ordinal placeholder was not copied from the continuous criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml

from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

EMIC_DIR = get_project_root() / "prompts" / "judge" / "criteria" / "emic_validity"
RULER_PATH = EMIC_DIR / "ruler.pt.yaml"
PROMPT_PATH = EMIC_DIR / "pt" / "prompt.md"
SHEET_PATH = get_project_root() / "docs" / "emic" / "annotator-instructions.pt.md"

# Fragments of real corpus pairs that earlier drafts of the prompt used as
# illustrations. They must not reappear: see TestNoRealPairLeaked.
CORPUS_FRAGMENTS = (
    "maikinho",
    "açucareiro",
    "patram",
    "ibama",
    "negligência sistêmica",
    "dona gilda",
)


def _normalize(text: str) -> str:
    """Collapse whitespace and case so rendering differences are not divergence.

    The YAML folds long strings, the prompt keeps one long line per paragraph and
    the sheet wraps at 88 columns; a passage also gets recased when it follows a
    label ("Sentido alterado: a reformulação...") . Only the words are the ruler,
    so wrapping and initial case are tolerated here. Any changed, added or dropped
    word still fails.
    """
    return " ".join(text.split()).casefold()


def _needle(text: str) -> str:
    """Normalize a ruler passage for substring search.

    Drops the trailing period: the ruler stores full sentences, and a consumer may
    render the same sentence with a suffix instead ("... : nota 1").
    """
    return _normalize(text).rstrip(".")


@pytest.fixture(scope="module")
def ruler() -> dict[str, Any]:
    return yaml.safe_load(RULER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prompt() -> str:
    return _normalize(PROMPT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sheet() -> str:
    return _normalize(SHEET_PATH.read_text(encoding="utf-8"))


def _ruler_texts(ruler: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (label, text) for every ruler passage that must appear verbatim."""
    texts: list[tuple[str, str]] = []
    for key, value in ruler["construct"].items():
        texts.append((f"construct.{key}", value))
    for entry in ruler["scale"]:
        texts.append((f"scale.{entry['score']}", entry["label"]))
    for entry in ruler["loss_types"]:
        texts.append((f"loss_types.{entry['name']}", entry["what"]))
    provisions = ruler["provisions"]
    texts.append(("provisions.unit_of_judgment", provisions["unit_of_judgment"]))
    texts.append(("provisions.question_calibrates", provisions["question_calibrates"]))
    texts.append(("provisions.out_of_scope", provisions["out_of_scope"]))
    for i, item in enumerate(provisions["not_a_loss"]):
        texts.append((f"provisions.not_a_loss[{i}]", item))
    guide = ruler["guide"]
    for key in ("locate", "extract", "cascade_intro", "no_penalty", "name_the_condition"):
        texts.append((f"guide.{key}", guide[key]))
    for entry in guide["cascade"]:
        texts.append((f"guide.cascade.{entry['score']}", entry["condition"]))
    for entry in ruler["examples"]:
        for field in ("segment", "question", "answer", "why"):
            texts.append((f"examples[{entry['title']}].{field}", entry[field]))
    return texts


class TestRulerIsSignedOff:
    def test_gate_is_signed_off(self, ruler: dict[str, Any]) -> None:
        # The annotation instrument refuses to build while this is false (spec §6
        # of the instrument design). Flipping it is the mechanical form of the
        # anthropologist gate.
        assert ruler["signed_off"] is True
        assert ruler["gate"] == "anthropologist-validation-of-readings"
        assert ruler["signed_off_on"]

    def test_scale_is_the_ordinal_one_to_five(self, ruler: dict[str, Any]) -> None:
        assert [entry["score"] for entry in ruler["scale"]] == [5, 4, 3, 2, 1]

    def test_cascade_covers_every_score_in_severity_order(self, ruler: dict[str, Any]) -> None:
        # Order is load-bearing: first matching condition wins, so the gravest
        # failure has to come first. A pair with both an added claim and an erased
        # situated element must land on 3, never 4.
        assert [entry["score"] for entry in ruler["guide"]["cascade"]] == [1, 2, 3, 4, 5]


class TestConsumersRenderTheRuler:
    @pytest.mark.parametrize("artifact", ["prompt", "sheet"])
    def test_every_ruler_passage_appears_verbatim(
        self, artifact: str, ruler: dict[str, Any], prompt: str, sheet: str
    ) -> None:
        rendered = prompt if artifact == "prompt" else sheet
        missing = [label for label, text in _ruler_texts(ruler) if _needle(text) not in rendered]
        assert not missing, f"{artifact} diverged from ruler.pt.yaml: {missing}"

    def test_annotator_only_text_stays_out_of_the_prompt(
        self, ruler: dict[str, Any], prompt: str, sheet: str
    ) -> None:
        # Human calibration ("hesitating is expected", "not a 1 when...") is
        # negative instruction that costs a small model attention; the cascade
        # already resolves by positive condition.
        for key, text in ruler["annotator_only"].items():
            assert _needle(text) in sheet, f"sheet is missing annotator_only.{key}"
            assert _needle(text) not in prompt, f"prompt leaked annotator_only.{key}"

    def test_judge_only_text_stays_out_of_the_sheet(
        self, ruler: dict[str, Any], prompt: str, sheet: str
    ) -> None:
        judge_only = ruler["judge_only"]
        for key in ("role", "rationale_rules_closing", "output"):
            assert _needle(judge_only[key]) in prompt, f"prompt is missing judge_only.{key}"
            assert _needle(judge_only[key]) not in sheet, f"sheet leaked judge_only.{key}"


class TestBlinding:
    def test_prompt_receives_only_the_blinded_fields(self) -> None:
        raw = PROMPT_PATH.read_text(encoding="utf-8")
        assert "$context" in raw
        assert "$question" in raw
        assert "$answer" in raw

    @pytest.mark.parametrize("path", [PROMPT_PATH, SHEET_PATH])
    def test_generator_metadata_is_never_named(self, path: Path) -> None:
        # Parity/blinding (spec §5): naming the Bloom level or the tacit-inference
        # field would anchor both the model and the annotator.
        low = path.read_text(encoding="utf-8").lower()
        for forbidden in ("bloom", "tacit_inference", "inferência tácita"):
            assert forbidden not in low, f"{path.name} names {forbidden!r}"

    def test_domain_is_not_named_to_the_judge(self, ruler: dict[str, Any], prompt: str) -> None:
        # Naming the domain activates the very schemas about traditional
        # populations that this criterion exists to detect, so it is deliberately
        # absent. The interview excerpt already supplies the field.
        #
        # Checked against the rendered prompt and against the ruler's *values*, not
        # the YAML source: a comment there explains why the domain is excluded, and
        # comments never reach an artifact.
        assert "ribeirinh" not in prompt
        rendered_values = _normalize(
            " ".join(text for _, text in _ruler_texts(ruler))
            + " ".join(ruler["judge_only"]["rationale_rules"])
            + ruler["judge_only"]["role"]
        )
        assert "ribeirinh" not in rendered_values


class TestNoRealPairLeaked:
    """Examples are synthetic by decision, and the decision protects the frame.

    A real pair in the prompt makes the judge's score on it non-independent, so it
    would have to leave the 2670-pair frame; a real pair in the sheet primes the
    annotators with corpus content. These strings come from actual corpus pairs
    that earlier drafts used as illustrations.
    """

    @pytest.mark.parametrize("path", [PROMPT_PATH, SHEET_PATH, RULER_PATH])
    @pytest.mark.parametrize("fragment", CORPUS_FRAGMENTS)
    def test_no_corpus_fragment_appears(self, path: Path, fragment: str) -> None:
        assert fragment not in path.read_text(encoding="utf-8").lower(), (
            f"{path.name} contains the real-corpus fragment {fragment!r}"
        )


class TestOrdinalOutputContract:
    def test_placeholder_is_the_ordinal_range(self, ruler: dict[str, Any]) -> None:
        # The continuous criteria ask for "<0-1>". Copying that here would be
        # silently destructive: OrdinalCriterionResponse rounds fractional scores
        # half-up before the range check, so a 0.8 would land on 1 -- the
        # distortion label -- with no error raised.
        output = ruler["judge_only"]["output"]
        assert "<inteiro 1-5>" in output
        assert "<0-1>" not in output

    def test_prompt_asks_for_json_with_both_fields(self, prompt: str) -> None:
        assert "json" in prompt.lower()
        assert "rationale" in prompt
        assert "score" in prompt

    def test_rationale_rules_reach_the_prompt(self, ruler: dict[str, Any], prompt: str) -> None:
        # These constrain the judge's own writing, which is where the reframing
        # tendency shows up: a rationale that calls a complaint "systemic
        # negligence" has committed the failure it was asked to detect.
        for rule in ruler["judge_only"]["rationale_rules"]:
            assert _needle(rule) in prompt, f"prompt is missing rationale rule: {rule!r}"
