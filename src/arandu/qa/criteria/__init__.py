"""Answer-judging criteria for Phase C ``arandu judge-answers``.

Four criteria, all wired into the :class:`AnswerJudge` pipeline:

- ``answer_correctness`` (LLM): system answer vs gold answer.
- ``answer_faithfulness`` (LLM): system answer grounded in retrieved passages.
- ``passage_coverage`` (LLM + deterministic): retrieval-quality lens. The
  deterministic variant lives in :mod:`offset_coverage`; the LLM variant
  is loaded from ``prompts/judge/criteria/passage_coverage/`` by the
  factory.
- ``abstention`` (LLM): cross-checks the answerer's structured ``abstained``
  flag against the answer text.
"""

from arandu.qa.criteria.offset_coverage import OffsetCoverageCriterion, offset_coverage

__all__ = ["OffsetCoverageCriterion", "offset_coverage"]
