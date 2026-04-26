"""Heuristic criteria for transcription quality validation.

Each criterion implements the JudgeCriterion protocol, evaluating a single
aspect of Whisper transcription output.
"""

from arandu.transcription.criteria.content_density import ContentDensityCriterion
from arandu.transcription.criteria.content_length import ContentLengthFloorCriterion
from arandu.transcription.criteria.repetition import RepetitionCriterion
from arandu.transcription.criteria.script_match import ScriptMatchCriterion
from arandu.transcription.criteria.segment_quality import SegmentQualityCriterion

__all__ = [
    "ContentDensityCriterion",
    "ContentLengthFloorCriterion",
    "RepetitionCriterion",
    "ScriptMatchCriterion",
    "SegmentQualityCriterion",
]
