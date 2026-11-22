"""Answerer client — wraps LLMClient.generate_structured with spec §5.6 retry contract.

The spec's retry strategy differs from :meth:`LLMClient.generate_structured`'s
default: on each failed attempt, bump the temperature by 0.1 (rather than
retrying at the same temperature). The rationale is that a deterministic
model that produced malformed JSON once will likely produce it again at
the same temperature; nudging temperature breaks the loop.

After three exhausted attempts, fall back to an abstained record with
``fallback_reason`` recorded in :attr:`AnswerRecord.answerer_meta` for
later audit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arandu.shared.llm_client import StructuredOutputError
from arandu.shared.rag.answer.prompts import render_prompt
from arandu.shared.rag.answer.schemas import AnswererOutput

if TYPE_CHECKING:
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.rag.answer.settings import AnswererSettings

logger = logging.getLogger(__name__)


_TEMPERATURE_BUMP_PER_ATTEMPT = 0.1
_MAX_ATTEMPTS = 3
_FALLBACK_RATIONALE = "Failed to produce structured output after 3 retries"


class AnswererClient:
    """Drive the Answerer LLM end-to-end for one question + passage set.

    Stateless across calls — construct once per run (so the LLMClient
    connection is reused) and call :meth:`answer` per question.

    Attributes:
        settings: The :class:`AnswererSettings` snapshot this client
            was built with. Persisted on every :class:`AnswerRecord`
            so reruns can attribute outputs to their configuration.
    """

    def __init__(self, llm_client: LLMClient, settings: AnswererSettings) -> None:
        """Construct from an existing LLMClient + settings snapshot.

        Args:
            llm_client: Already-constructed :class:`LLMClient`. The CLI
                builds it from ``settings.provider`` + ``settings.model_id``
                via the unified client.
            settings: Per-call configuration (temperature, max_tokens,
                language, etc.). Read on every :meth:`answer` call.
        """
        self._client = llm_client
        self.settings = settings

    def answer(
        self,
        *,
        question: str,
        passage_texts: list[str],
    ) -> tuple[AnswererOutput, dict[str, object]]:
        """Run the answerer pipeline once for ``question`` over ``passage_texts``.

        Args:
            question: Verbatim question text.
            passage_texts: Pre-resolved + pre-packed passage texts in
                rank order. The caller is responsible for budget
                packing (see :func:`pack_passages`).

        Returns:
            A ``(output, meta)`` tuple. ``meta`` carries audit fields
            ready to merge into :attr:`AnswerRecord.answerer_meta`:

            - ``attempts``: count of LLM calls before success or fallback
            - ``final_temperature``: the temperature used on the
              attempt that succeeded (or the last attempt that failed
              before fallback)
            - ``fallback_reason``: present only when the output is the
              abstained fallback
        """
        prompt = render_prompt(
            self.settings.language,
            question=question,
            passages=passage_texts,
        )

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            temperature = self.settings.temperature + (attempt - 1) * _TEMPERATURE_BUMP_PER_ATTEMPT
            try:
                output = self._client.generate_structured(
                    prompt=prompt,
                    response_model=AnswererOutput,
                    temperature=temperature,
                    max_tokens=self.settings.max_tokens,
                    max_retries=0,
                )
            except StructuredOutputError as exc:
                logger.warning(
                    "Answerer attempt %d/%d failed at temperature=%.2f: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    temperature,
                    exc,
                )
                continue
            return output, {
                "attempts": attempt,
                "final_temperature": temperature,
            }

        # All attempts exhausted — return the audit-only fallback.
        logger.error(
            "Answerer exhausted %d attempts for question %r; falling back to abstained.",
            _MAX_ATTEMPTS,
            question[:80],
        )
        fallback = AnswererOutput(
            abstained=True,
            answer=None,
            rationale=_FALLBACK_RATIONALE,
        )
        return fallback, {
            "attempts": _MAX_ATTEMPTS,
            "final_temperature": self.settings.temperature
            + (_MAX_ATTEMPTS - 1) * _TEMPERATURE_BUMP_PER_ATTEMPT,
            "fallback_reason": _FALLBACK_RATIONALE,
        }
