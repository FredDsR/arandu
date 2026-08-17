"""Render the Label Studio labeling config from the emic ruler.

The annotator's widget IS the instrument. The five options carry the anchor
sentences verbatim from ``ruler["scale"]`` so the human and the LLM judge score
against one ruler: the weighted kappa of spec section 6 only measures agreement
if both read the same text, and a divergence is not recoverable after
annotation.

``shuffle`` is set to ``"false"`` explicitly. Shuffling an ordinal scale destroys
its meaning, and relying on a platform default for that is not worth the risk.

Judge-only material (the role framing, the JSON output contract, the rationale
rules) is excluded: it would prime the annotator with the machine's framing.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import quoteattr

_HEADER = "Validade êmica do par"


def _attr(value: str) -> str:
    """Quote and escape a string for use as an XML attribute value."""
    return quoteattr(value)


def _paragraph(text: str) -> str:
    """Render a block of ruler prose as a static element.

    ``Header`` and not ``Text``: ``Text`` is how Label Studio binds a task
    variable, so reserving it for the three blinded fields lets a test assert
    that the bound set is exactly ``$segment``, ``$question``, ``$answer``.
    ``Header`` also takes no ``name``, which avoids the duplicate-name collision
    a repeated ``Text`` element would cause.
    """
    return f"  <Header value={_attr(text)} />"


def _header(text: str) -> str:
    """Render a section title (same element as prose; kept separate for intent)."""
    return f"  <Header value={_attr(text)} />"


def render_labeling_config(ruler: dict[str, Any]) -> str:
    """Render the ``labeling_config.xml`` body for the annotation project.

    Args:
        ruler: The parsed ruler mapping (see
            :func:`arandu.shared.annotation.ruler.load_ruler`).

    Returns:
        A ``<View>`` XML document as a string. Well-formed, self-contained, and
        deterministic: the same ruler always renders byte-identical output.
    """
    construct = ruler["construct"]
    provisions = ruler["provisions"]
    guide = ruler["guide"]
    annotator_only = ruler["annotator_only"]

    lines: list[str] = ["<View>", _header(_HEADER)]

    # The pair under judgment. These three are the ONLY task variables bound:
    # anything else would let an attentive annotator reconstruct the
    # stratification (spec section 5).
    lines += [
        _header("Trecho da entrevista"),
        '  <Text name="segment" value="$segment" />',
        _header("Pergunta"),
        '  <Text name="question" value="$question" />',
        _header("Resposta"),
        '  <Text name="answer" value="$answer" />',
    ]

    lines.append(_header("O construto"))
    lines += [
        _paragraph(construct[key])
        for key in ("emic", "non_emic", "gradient", "question", "not_faithfulness")
    ]

    lines.append(_header("O que se avalia"))
    lines += [
        _paragraph(provisions["unit_of_judgment"]),
        _paragraph(provisions["question_calibrates"]),
        _paragraph(provisions["out_of_scope"]),
    ]
    lines += [_paragraph(text) for text in provisions["not_a_loss"]]

    lines.append(_header("Tipos de perda êmica"))
    lines += [_paragraph(f"{entry['name']}: {entry['what']}") for entry in ruler["loss_types"]]

    lines.append(_header("Como pontuar"))
    lines += [
        _paragraph(guide["locate"]),
        _paragraph(guide["extract"]),
        _paragraph(guide["cascade_intro"]),
    ]
    lines += [_paragraph(f"{entry['score']}: {entry['condition']}") for entry in guide["cascade"]]
    lines.append(_paragraph(guide["no_penalty"]))
    lines += [_paragraph(text) for text in annotator_only.values()]

    lines.append(_header("Sua nota"))
    lines.append(
        '  <Choices name="score" toName="answer" choice="single-radio" '
        'shuffle="false" required="true">'
    )
    for entry in ruler["scale"]:
        # "5 - Preserva o sentido, ...": the number the study records, next to the
        # anchor sentence the judge prompt puts beside the same number.
        option = f"{entry['score']} - {entry['label']}"
        lines.append(f"    <Choice value={_attr(option)} />")
    lines.append("  </Choices>")

    lines += [
        _header("Justificativa (opcional)"),
        _paragraph(guide["name_the_condition"]),
        '  <TextArea name="rationale" toName="answer" editable="true" '
        'maxSubmissions="1" rows="3" />',
        "</View>",
    ]
    return "\n".join(lines) + "\n"
