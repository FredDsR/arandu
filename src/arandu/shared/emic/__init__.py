"""Emic-validity judge: ordinal scoring of CEP pairs (spec §5).

The ordinal ``emic_validity`` score this stage produces is the study's
measurement of emic validity. The human annotation round (spec §6) measures
agreement with it; it is not a ground truth that supersedes it.
"""

from arandu.shared.emic.batch import run_emic_judge_batch
from arandu.shared.emic.schemas import (
    EmicJudgeResult,
    EmicScope,
    EmicScore,
    EmicSourceScores,
)
from arandu.shared.emic.settings import EmicJudgeSettings

__all__ = [
    "EmicJudgeResult",
    "EmicJudgeSettings",
    "EmicScope",
    "EmicScore",
    "EmicSourceScores",
    "run_emic_judge_batch",
]
