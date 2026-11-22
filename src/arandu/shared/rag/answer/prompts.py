"""Jinja prompt loader for the answerer.

Templates live next to this module under ``prompts/answerer_<lang>.j2``.
Two templates ship today: ``pt`` (Portuguese, project default) and
``en`` (English).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# select_autoescape is irrelevant for plain-text JSON output but the
# Jinja docs recommend it as a default — keeps surprise out of edits.
_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def render_prompt(
    language: Literal["pt", "en"],
    *,
    question: str,
    passages: list[str],
) -> str:
    """Render the answerer prompt for ``language`` with the given passages.

    Args:
        language: ``"pt"`` or ``"en"``. Selects which template file is loaded.
        question: The natural-language query to put into the prompt.
        passages: Pre-resolved passage texts in rank order. May be empty
            (the template handles the "no passages" branch via Jinja).

    Returns:
        The rendered prompt string ready to feed to
        :meth:`LLMClient.generate_structured`.
    """
    template = _ENV.get_template(f"answerer_{language}.j2")
    return template.render(question=question, passages=passages)
