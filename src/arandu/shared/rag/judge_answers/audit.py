"""Abstention-disagreement audit log emission (spec §6.4).

When the answerer's structured ``abstained`` flag disagrees with the
abstention judge's score (threshold τ_abstention = 0.7), the item is
flagged for an audit pass. Expected ~5% of items per the spec; the
audit list lands at
``results/<id>/judge_answers/outputs/abstention_audit.jsonl``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.rag.schemas import AnswerRecord

logger = logging.getLogger(__name__)


_AUDIT_FILENAME = "abstention_audit.jsonl"
_ABSTENTION_CRITERION = "abstention"


class AbstentionDisagreement(BaseModel):
    """One row of the abstention-audit JSONL.

    Attributes:
        qa_pair_id: Composite id from the answer record.
        retriever_id: Which arm produced the answer.
        answerer_abstained: The structured ``abstained`` flag from the
            answerer's :class:`AnswererOutput`.
        judge_score: Score from the abstention judge criterion. ``None``
            when the criterion errored (still flagged so the auditor
            sees the gap).
        judge_threshold: τ_abstention (default 0.7).
        disagreement_type: ``"answerer_abstains_judge_disagrees"`` or
            ``"answerer_commits_judge_says_abstain"``.
        answer_text: Verbatim answer text (helps the auditor judge
            without round-tripping back to the AnswerRecord).
        rationale: Answerer's rationale (same).
    """

    qa_pair_id: str
    retriever_id: str
    answerer_abstained: bool
    judge_score: float | None
    judge_threshold: float
    disagreement_type: str
    answer_text: str | None
    rationale: str = Field(default="")


def detect_disagreement(answer: AnswerRecord, threshold: float) -> AbstentionDisagreement | None:
    """Compare answerer's flag against the abstention judge's verdict.

    Args:
        answer: An :class:`AnswerRecord` that has been judged (its
            ``validation`` field is populated).
        threshold: τ_abstention. Items where the judge score crosses
            this threshold are considered "judge says abstain".

    Returns:
        :class:`AbstentionDisagreement` when the two signals disagree;
        ``None`` when they agree (or when the judge didn't run).
    """
    if answer.validation is None:
        return None
    # The pipeline is a single stage with all criteria in one step.
    for step in answer.validation.stage_results.values():
        score = step.criterion_scores.get(_ABSTENTION_CRITERION)
        if score is None:
            continue
        judge_says_abstain = score.score is not None and score.score >= threshold
        if answer.abstained == judge_says_abstain:
            return None
        if answer.abstained and not judge_says_abstain:
            disagreement_type = "answerer_abstains_judge_disagrees"
        else:
            disagreement_type = "answerer_commits_judge_says_abstain"
        return AbstentionDisagreement(
            qa_pair_id=answer.qa_pair_id,
            retriever_id=answer.retriever_id,
            answerer_abstained=answer.abstained,
            judge_score=score.score,
            judge_threshold=score.threshold,
            disagreement_type=disagreement_type,
            answer_text=answer.answer_text,
            rationale=answer.rationale,
        )
    return None


def write_audit_log(outputs_dir: Path, disagreements: list[AbstentionDisagreement]) -> Path | None:
    """Emit JSONL of disagreements to ``<outputs_dir>/abstention_audit.jsonl``.

    Args:
        outputs_dir: The judge_answers stage's outputs dir.
        disagreements: All flagged rows from this run.

    Returns:
        The audit file path when at least one disagreement was found;
        ``None`` when the list is empty (no file written — silence on
        the happy path).
    """
    if not disagreements:
        logger.info("No abstention disagreements detected; skipping audit log.")
        return None
    audit_path = outputs_dir / _AUDIT_FILENAME
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        for row in disagreements:
            f.write(row.model_dump_json() + "\n")
    logger.info("Wrote %d abstention disagreement(s) to %s.", len(disagreements), audit_path)
    return audit_path
