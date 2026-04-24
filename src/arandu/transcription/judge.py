"""Transcription quality judge using heuristic and optional LLM criteria.

Evaluates transcription quality with pure-Python heuristics (script match,
repetition, content density, segment quality). When an LLM client is
supplied, adds a second filter stage with ``language_drift`` and
``hallucination_loop`` criteria to catch quality failures that heuristics
cannot detect (Latin-script language drift, formulaic Whisper hallucinations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.config import get_llm_config
from arandu.shared.judge import (
    BaseJudge,
    JudgePipeline,
    JudgeStage,
    JudgeStep,
    LLMCriterionFactory,
)
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.transcription.criteria import (
    ContentDensityCriterion,
    RepetitionCriterion,
    ScriptMatchCriterion,
    SegmentQualityCriterion,
)

if TYPE_CHECKING:
    from arandu.shared.judge.schemas import JudgePipelineResult


def build_validator_client(
    model_id: str,
    *,
    provider: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Build an ``LLMClient`` for the ``TranscriptionJudge`` LLM stage.

    Resolves the base URL from ``LLMConfig`` (which reads
    ``ARANDU_LLM_BASE_URL`` from environment / ``.env``) when not provided
    explicitly. Infers ``provider='custom'`` when a base URL is set but
    ``provider`` wasn't, falling back to ``'ollama'`` otherwise.

    Args:
        model_id: Model identifier (e.g. ``'qwen3:14b'``, ``'gemini-2.5-flash'``).
        provider: Optional provider name. ``'openai'`` / ``'ollama'`` / ``'custom'``.
        base_url: Optional base URL override.

    Returns:
        A configured ``LLMClient`` instance ready to pass as
        ``TranscriptionJudge(validator_client=...)``.
    """
    llm_config = get_llm_config()
    resolved_base_url = base_url or llm_config.base_url
    resolved_provider = provider or ("custom" if resolved_base_url else "ollama")
    return LLMClient(
        provider=LLMProvider(resolved_provider),
        model_id=model_id,
        base_url=resolved_base_url,
    )


class TranscriptionJudge(BaseJudge):
    """Evaluate transcription quality using heuristic and optional LLM criteria.

    Builds a two-stage filter pipeline:
    1. ``heuristic_filter`` — script match, repetition, content density,
       segment quality (always runs, no LLM needed).
    2. ``llm_filter`` — ``language_drift`` + ``hallucination_loop`` (runs
       only when ``validator_client`` is provided). Skipped automatically
       when the heuristic stage rejects.

    The LLM criteria target failure modes the heuristics cannot detect:

    - ``language_drift`` catches sustained Latin-script drift (e.g., English
      content in a Portuguese transcription) that ``ScriptMatchCriterion``
      passes by design.
    - ``hallucination_loop`` catches formulaic Whisper hallucinations
      (YouTube-style openings/closings, short sentence loops appearing only
      a handful of times, apology/filler loops) that ``RepetitionCriterion``
      misses because its n-gram threshold is tuned for heavy repetition.

    **Limitations of the LLM criteria** — both are text-only and cannot
    detect:

    - Plausible silence-fillers (a single coherent sentence Whisper invents
      from background noise with no loop signature).
    - Low-SNR invention (phonetically-close but wrong words in a real
      utterance).
    - Name/number substitutions that read naturally.

    These require audio-aware signals (``avg_logprob`` / ``no_speech_prob``
    per segment, VAD, multi-model cross-check) and are tracked separately.
    See the transcription-validation guide for full details.

    Args:
        language: Expected transcription language (default ``"pt"``).
        validator_client: Optional LLM client. When provided, enables the
            ``llm_filter`` stage.
        temperature: Sampling temperature for LLM criteria (default ``0.3``).
        max_tokens: Max tokens for LLM criterion responses (default ``2048``).
    """

    def __init__(
        self,
        *,
        language: str = "pt",
        validator_client: LLMClient | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize judge with language and optional LLM client.

        Args:
            language: Expected transcription language (default ``"pt"``).
            validator_client: Optional LLM client enabling the LLM filter stage.
            temperature: Sampling temperature for LLM criteria.
            max_tokens: Maximum tokens for LLM criterion responses.
        """
        self._language = language
        self._factory: LLMCriterionFactory | None = None
        if validator_client is not None:
            self._factory = LLMCriterionFactory(
                llm_client=validator_client,
                language=language,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        super().__init__()

    def _build_pipeline(self) -> JudgePipeline:
        """Build heuristic + optional LLM evaluation pipeline.

        Returns:
            Configured JudgePipeline. Always includes the heuristic filter
            stage; adds the LLM filter stage when a factory is available.
        """
        heuristic_step = JudgeStep(
            criteria=[
                ScriptMatchCriterion(),
                RepetitionCriterion(),
                ContentDensityCriterion(),
                SegmentQualityCriterion(),
            ],
        )
        stages = [
            JudgeStage(name="heuristic_filter", step=heuristic_step, mode="filter"),
        ]

        if self._factory is not None:
            llm_step = JudgeStep(
                criteria=["language_drift", "hallucination_loop"],
                factory=self._factory,
            )
            stages.append(JudgeStage(name="llm_filter", step=llm_step, mode="filter"))

        return JudgePipeline(stages=stages)

    def evaluate_transcription(
        self,
        text: str,
        *,
        expected_language: str | None = None,
        duration_ms: int | None = None,
        segments: list | None = None,
    ) -> JudgePipelineResult:
        """Evaluate a transcription with domain-specific parameters.

        Convenience method that maps transcription fields to criterion kwargs.

        Args:
            text: Transcription text.
            expected_language: Expected language code. Defaults to init language.
            duration_ms: Audio duration in milliseconds.
            segments: List of TranscriptionSegment objects.

        Returns:
            JudgePipelineResult with heuristic and (optionally) LLM scores.
        """
        return self.evaluate(
            text=text,
            expected_language=expected_language or self._language,
            duration_ms=duration_ms,
            segments=segments or [],
        )
