"""Transcription quality judge using heuristic criteria.

Evaluates transcription quality using pure-Python heuristic checks.
LLM-based criteria (language drift, hallucination loop) will be added
in a future update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.judge import BaseJudge, JudgePipeline, JudgeStage, JudgeStep
from arandu.transcription.criteria import (
    ContentDensityCriterion,
    RepetitionCriterion,
    ScriptMatchCriterion,
    SegmentQualityCriterion,
)

if TYPE_CHECKING:
    from arandu.shared.judge.schemas import JudgePipelineResult


class TranscriptionJudge(BaseJudge):
    """Evaluate transcription quality using heuristic criteria.

    Builds a single-stage heuristic filter pipeline. No LLM client needed.

    Args:
        language: Expected transcription language (default "pt").
    """

    def __init__(self, *, language: str = "pt") -> None:
        """Initialize judge with language setting.

        Args:
            language: Expected transcription language (default "pt").
        """
        self._language = language
        super().__init__()

    def _build_pipeline(self) -> JudgePipeline:
        """Build heuristic-only evaluation pipeline.

        Returns:
            Configured JudgePipeline with a single heuristic filter stage.
        """
        step = JudgeStep(
            criteria=[
                ScriptMatchCriterion(),
                RepetitionCriterion(),
                ContentDensityCriterion(),
                SegmentQualityCriterion(),
            ],
        )
        return JudgePipeline(stages=[JudgeStage(name="heuristic_filter", step=step, mode="filter")])

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
            JudgePipelineResult with heuristic scores.
        """
        return self.evaluate(
            text=text,
            expected_language=expected_language or self._language,
            duration_ms=duration_ms,
            segments=segments or [],
        )
