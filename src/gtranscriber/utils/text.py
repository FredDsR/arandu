"""Cross-cutting text processing utilities for LLM response handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CODEBLOCK_PATTERN = re.compile(r"```(?:json)?\n?", re.DOTALL)


@dataclass(frozen=True)
class GenerateResult:
    """Result from LLM text generation.

    Encapsulates the generated content and any internal thinking traces
    from thinking models (e.g., Qwen3, DeepSeek-R1). For non-thinking
    models, thinking is None and content is the raw response.

    Attributes:
        content: Generated text with thinking tags removed.
        thinking: Internal thinking trace, or None for non-thinking models.
    """

    content: str
    thinking: str | None = None


def extract_thinking(response: str) -> GenerateResult:
    """Extract thinking content from LLM responses with ``<think>`` tags.

    Handles single or multiple ``<think>`` blocks. Multiple blocks are
    concatenated with newlines. Empty blocks result in ``thinking=None``.

    Args:
        response: Raw LLM response text.

    Returns:
        GenerateResult with separated thinking and content.

    Examples:
        >>> extract_thinking("<think>reasoning</think>{\"key\": 1}")
        GenerateResult(content='{"key": 1}', thinking='reasoning')
        >>> extract_thinking('{"key": 1}')
        GenerateResult(content='{"key": 1}', thinking=None)
    """
    matches = _THINK_PATTERN.findall(response)

    if not matches:
        return GenerateResult(content=response, thinking=None)

    thinking = "\n".join(m.strip() for m in matches)
    content = _THINK_PATTERN.sub(" ", response).strip()

    return GenerateResult(
        content=content,
        thinking=thinking if thinking else None,
    )


def validate_score(value: Any, default: float = 0.5) -> float:
    """Validate and clamp a score to [0.0, 1.0].

    Args:
        value: Raw score value from LLM response.
        default: Fallback score when value cannot be converted to float.

    Returns:
        Float score clamped to [0.0, 1.0].

    Examples:
        >>> validate_score(0.8)
        0.8
        >>> validate_score(1.5)
        1.0
        >>> validate_score("not_a_number")
        0.5
    """
    try:
        score = float(value)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return default


def strip_markdown_codeblock(text: str) -> str:
    """Remove markdown code block wrappers from text.

    Strips leading/trailing whitespace and removes ````json`/```` fences
    commonly added by LLMs around JSON responses.

    Args:
        text: Raw text potentially wrapped in markdown code blocks.

    Returns:
        Text with code block markers removed.

    Examples:
        >>> strip_markdown_codeblock('```json\\n{"key": 1}\\n```')
        '{"key": 1}'
        >>> strip_markdown_codeblock('{"key": 1}')
        '{"key": 1}'
    """
    text = text.strip()
    if text.startswith("```"):
        text = _CODEBLOCK_PATTERN.sub("", text)
    return text
