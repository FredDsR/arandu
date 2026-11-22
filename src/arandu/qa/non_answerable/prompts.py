"""Jinja prompt loader for non-answerable perturbation (spec §7.4).

Templates live under the project ``prompts/qa/non_answerable/<lang>.j2``.
Only ``pt`` ships today (project default); the loader is language-keyed
so an ``en`` variant can drop in without code changes.
"""

from __future__ import annotations

from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from arandu.utils.paths import get_project_root

_PROMPTS_DIR = get_project_root() / "prompts" / "qa" / "non_answerable"

_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def render_perturbation_prompt(language: Literal["pt", "en"], *, question: str) -> str:
    """Render the perturbation prompt for ``language`` around ``question``.

    Args:
        language: ``"pt"`` or ``"en"``. Selects the template file.
        question: The CEP question to perturb.

    Returns:
        The rendered prompt ready for :meth:`LLMClient.generate_structured`.
    """
    template = _ENV.get_template(f"{language}.j2")
    return template.render(question=question)
