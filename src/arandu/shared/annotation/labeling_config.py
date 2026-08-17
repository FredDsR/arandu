"""Render the two annotator-facing surfaces from the emic ruler.

The annotator's widget IS the instrument. The five options carry the anchor
sentences verbatim from ``ruler["scale"]`` so the human and the LLM judge score
against one ruler: the weighted kappa of spec section 6 only measures agreement
if both read the same text, and a divergence is not recoverable after
annotation.

The ruler is roughly two screens of Portuguese prose. Rendering all of it
between the pair and the rating widget pushed a 4000-character segment out of
view by the time the annotator reached the radio buttons, once per task, 120
times per annotator. So the ruler is split across two surfaces:

* :func:`render_expert_instruction` renders the full ruler as HTML for Label
  Studio's own instructions modal (the project's ``expert_instruction`` field,
  reachable from the help button). That is the reference an annotator opens.
* :func:`render_labeling_config` keeps a short collapsed summary on the canvas,
  closed by default, holding only what is consulted mid-doubt: the scoring
  cascade, the "does not reduce the score" list, and the annotator-only
  calibration. The pair then sits immediately above the rating widget.

``shuffle`` is set to ``"false"`` explicitly. Shuffling an ordinal scale destroys
its meaning, and relying on a platform default for that is not worth the risk.

Judge-only material (the role framing, the JSON output contract, the rationale
rules) is excluded from BOTH surfaces: it would prime the annotator with the
machine's framing.
"""

from __future__ import annotations

from html import escape
from typing import Any
from xml.sax.saxutils import quoteattr

_HEADER = "Validade êmica do par"
_SUMMARY_TITLE = "Cascata de pontuação (resumo)"

#: Canvas typography.
#:
#: Three jobs, all of them consequences of the dry run: cap the segment box so a
#: long chunk scrolls inside itself instead of pushing the rating widget off
#: screen; give the pair a visible frame so it reads as the object under
#: judgment; and make the collapsed ruler summary secondary to it. Prose is
#: capped at ~70 characters per line, which is where continuous reading stops
#: costing extra eye travel.
_STYLE = """
    .emic-summary { margin: 0 0 14px; }
    .emic-summary h4 { font-size: 0.95em; margin: 12px 0 4px; color: #3d3d3d; }
    .emic-summary h6 {
      max-width: 70ch; font-size: 0.88em; font-weight: 400; color: #5c5c5c;
      line-height: 1.5; margin: 0 0 6px;
    }
    .emic-pair {
      border: 1px solid #d0d0d0; border-radius: 6px; background: #fafafa;
      padding: 10px 14px; margin: 0 0 16px;
    }
    .emic-pair h4 {
      font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.05em;
      color: #6b6b6b; margin: 10px 0 3px;
    }
    .emic-segment {
      max-height: 260px; overflow-y: auto; background: #ffffff;
      border: 1px solid #e4e4e4; border-radius: 4px; padding: 8px 10px;
      line-height: 1.55;
    }
    .emic-score h4 { font-size: 0.95em; margin: 12px 0 6px; color: #3d3d3d; }
    .emic-score h6 {
      max-width: 70ch; font-size: 0.88em; font-weight: 400; color: #5c5c5c;
      line-height: 1.5; margin: 0 0 6px;
    }
"""


def _attr(value: str) -> str:
    """Quote and escape a string for use as an XML attribute value."""
    return quoteattr(value)


def _paragraph(text: str, indent: str) -> str:
    """Render a block of ruler prose as a static element.

    ``Header`` and not ``Text``: ``Text`` is how Label Studio binds a task
    variable, so reserving it for the three blinded fields lets a test assert
    that the bound set is exactly ``$segment``, ``$question``, ``$answer``.
    ``Header`` also takes no ``name``, which avoids the duplicate-name collision
    a repeated ``Text`` element would cause.

    Args:
        text: The ruler passage, copied verbatim.
        indent: Leading whitespace for the emitted line.

    Returns:
        A single ``<Header>`` line.
    """
    return f'{indent}<Header value={_attr(text)} size="6" />'


def _header(text: str, indent: str, size: str = "4") -> str:
    """Render a section title (same element as prose; kept separate for intent).

    Args:
        text: The title.
        indent: Leading whitespace for the emitted line.
        size: Label Studio ``Header`` size, which selects the heading level.

    Returns:
        A single ``<Header>`` line.
    """
    return f"{indent}<Header value={_attr(text)} size={_attr(size)} />"


def _summary_lines(ruler: dict[str, Any]) -> list[str]:
    """Render the collapsed canvas summary.

    Only what an annotator consults mid-doubt: the cascade (the decision
    procedure), the list of things that do not cost a point, and the
    annotator-only calibration about the fine boundaries and the narrow door to
    1. Everything else lives in the instructions modal.

    Args:
        ruler: The parsed ruler mapping.

    Returns:
        The lines of the ``<Collapse>`` block.
    """
    guide = ruler["guide"]
    indent = "        "
    lines = [
        "  <Collapse>",
        f"    <Panel value={_attr(_SUMMARY_TITLE)}>",
        '      <View className="emic-summary">',
        _paragraph(guide["cascade_intro"], indent),
    ]
    lines += [
        _paragraph(f"{entry['score']}: {entry['condition']}", indent) for entry in guide["cascade"]
    ]
    lines += [
        _header("Não reduzem a nota", indent),
        _paragraph(guide["no_penalty"], indent),
        _header("Quando hesitar", indent),
    ]
    lines += [_paragraph(text, indent) for text in ruler["annotator_only"].values()]
    lines += ["      </View>", "    </Panel>", "  </Collapse>"]
    return lines


def render_labeling_config(ruler: dict[str, Any]) -> str:
    """Render the ``labeling_config.xml`` body for the annotation project.

    Layout, top to bottom: the header, the collapsed ruler summary (closed by
    default), the pair under judgment, then the rating widget and the rationale
    box. The pair sits immediately above the widget so the annotator never has
    to hold a 4000-character segment in their head while scrolling to the radio
    buttons.

    Args:
        ruler: The parsed ruler mapping (see
            :func:`arandu.shared.annotation.ruler.load_ruler`).

    Returns:
        A ``<View>`` XML document as a string. Well-formed, self-contained, and
        deterministic: the same ruler always renders byte-identical output.
    """
    guide = ruler["guide"]

    lines: list[str] = [
        "<View>",
        f"  <Style>{_STYLE}  </Style>",
        _header(_HEADER, "  ", size="3"),
    ]

    lines += _summary_lines(ruler)

    # The pair under judgment. These three are the ONLY task variables bound:
    # anything else would let an attentive annotator reconstruct the
    # stratification (spec section 5).
    lines += [
        '  <View className="emic-pair">',
        _header("Trecho da entrevista", "    "),
        '    <View className="emic-segment">',
        '      <Text name="segment" value="$segment" />',
        "    </View>",
        _header("Pergunta", "    "),
        '    <Text name="question" value="$question" />',
        _header("Resposta", "    "),
        '    <Text name="answer" value="$answer" />',
        "  </View>",
    ]

    lines += ['  <View className="emic-score">', _header("Sua nota", "    ")]
    lines.append(
        '    <Choices name="score" toName="answer" choice="single-radio" '
        'shuffle="false" required="true">'
    )
    for entry in ruler["scale"]:
        # "5 - Preserva o sentido, ...": the number the study records, next to the
        # anchor sentence the judge prompt puts beside the same number.
        option = f"{entry['score']} - {entry['label']}"
        lines.append(f"      <Choice value={_attr(option)} />")
    lines.append("    </Choices>")

    lines += [
        _header("Justificativa (opcional)", "    "),
        _paragraph(guide["name_the_condition"], "    "),
        '    <TextArea name="rationale" toName="answer" editable="true" '
        'maxSubmissions="1" rows="3" />',
        "  </View>",
        "</View>",
    ]
    return "\n".join(lines) + "\n"


def _html_paragraphs(texts: list[str]) -> list[str]:
    """Render ruler passages as escaped ``<p>`` elements."""
    return [f"<p>{escape(text)}</p>" for text in texts]


def _html_list(items: list[str]) -> list[str]:
    """Render ruler passages as an escaped ``<ul>``."""
    return ["<ul>", *[f"  <li>{escape(item)}</li>" for item in items], "</ul>"]


def render_expert_instruction(ruler: dict[str, Any]) -> str:
    """Render the full ruler as HTML for Label Studio's instructions modal.

    This is the project's ``expert_instruction`` field, shown behind the help
    button when ``show_instruction`` is on. It carries the whole annotator-facing
    ruler so the canvas does not have to: construct, scale, loss types,
    provisions, the full scoring guide, and the annotator-only calibration.

    Plain semantic HTML with no styling and no external reference: it is
    injected into a page whose CSS is Label Studio's, and a research instrument
    must not depend on a resource that can move or disappear between the audit
    and the annotation round.

    Judge-only material is excluded here for the same reason it is excluded from
    the canvas.

    Args:
        ruler: The parsed ruler mapping (see
            :func:`arandu.shared.annotation.ruler.load_ruler`).

    Returns:
        An HTML fragment as a string. Deterministic for a given ruler.

    Raises:
        KeyError: If the ruler is missing a section the instruction needs.
    """
    construct = ruler["construct"]
    provisions = ruler["provisions"]
    guide = ruler["guide"]

    lines: list[str] = [f"<h1>{escape(_HEADER)}</h1>", "<h2>O construto</h2>"]
    lines += _html_paragraphs(
        [construct[key] for key in ("emic", "non_emic", "gradient", "question", "not_faithfulness")]
    )

    lines.append("<h2>A escala</h2>")
    lines += _html_list([f"{entry['score']} - {entry['label']}" for entry in ruler["scale"]])

    lines.append("<h2>O que se avalia</h2>")
    lines += _html_paragraphs(
        [provisions[key] for key in ("unit_of_judgment", "question_calibrates", "out_of_scope")]
    )
    lines.append("<h3>Não são perda</h3>")
    lines += _html_list(list(provisions["not_a_loss"]))

    lines.append("<h2>Tipos de perda êmica</h2>")
    lines += _html_list([f"{entry['name']}: {entry['what']}" for entry in ruler["loss_types"]])

    lines.append("<h2>Como pontuar</h2>")
    lines += _html_paragraphs([guide["locate"], guide["extract"], guide["cascade_intro"]])
    lines += _html_list([f"{entry['score']}: {entry['condition']}" for entry in guide["cascade"]])
    lines.append("<h3>Não reduzem a nota</h3>")
    lines += _html_paragraphs([guide["no_penalty"], guide["name_the_condition"]])

    lines.append("<h2>Quando hesitar</h2>")
    lines += _html_paragraphs(list(ruler["annotator_only"].values()))

    return "\n".join(lines) + "\n"
