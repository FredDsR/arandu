"""Cross-cutting text processing utilities for LLM response handling."""

from __future__ import annotations

import re
from dataclasses import dataclass

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


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
