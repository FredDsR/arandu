"""``arandu judge-answers`` machinery — 4-criterion judge over AnswerRecord artifacts.

Spec §6. Public entry points:

- :func:`run_judge_answers_batch` — CLI driver.
- :class:`AnswerJudge` — pipeline composition (4 criteria in score mode).
- :class:`JudgeAnswersSettings` — Pydantic settings.
"""

from arandu.shared.rag.judge_answers.batch import run_judge_answers_batch
from arandu.shared.rag.judge_answers.judge import AnswerJudge
from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings

__all__ = ["AnswerJudge", "JudgeAnswersSettings", "run_judge_answers_batch"]
