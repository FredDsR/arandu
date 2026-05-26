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
            when the criterion errored; the audit row is still emitted
            (with ``disagreement_type="judge_error"``) so failures stay
            visible to the auditor.
        judge_threshold: τ_abstention — pulled from the criterion's own
            :attr:`CriterionScore.threshold` so the recorded value and
            the decision threshold are the same field.
        disagreement_type: ``"answerer_abstains_judge_disagrees"``,
            ``"answerer_commits_judge_says_abstain"``, or
            ``"judge_error"`` when the criterion didn't produce a score.
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


def detect_disagreement(answer: AnswerRecord) -> AbstentionDisagreement | None:
    """Compare answerer's flag against the abstention judge's verdict.

    Args:
        answer: An :class:`AnswerRecord` that has been judged (its
            ``validation`` field is populated). The criterion's own
            :attr:`CriterionScore.threshold` is used both for the
            decision AND for the recorded ``judge_threshold`` — single
            source of truth, so config edits affect the audit without
            having to propagate the new value separately.

    Returns:
        :class:`AbstentionDisagreement` in three cases:
        - answerer abstained but judge thinks the text is a committal answer
        - answerer committed but judge reads the text as a refusal
        - the criterion errored (judge score is ``None``) — flagged so
          the auditor sees the gap rather than silently treating None
          as "judge agrees"
        Otherwise ``None`` (judge didn't run, or both signals agree).
    """
    if answer.validation is None:
        return None
    # The pipeline is a single stage with all criteria in one step.
    for step in answer.validation.stage_results.values():
        score = step.criterion_scores.get(_ABSTENTION_CRITERION)
        if score is None:
            continue
        # Treat criterion errors as their own disagreement category so
        # they aren't silently lumped in with "judge agrees".
        if score.score is None:
            return AbstentionDisagreement(
                qa_pair_id=answer.qa_pair_id,
                retriever_id=answer.retriever_id,
                answerer_abstained=answer.abstained,
                judge_score=None,
                judge_threshold=score.threshold,
                disagreement_type="judge_error",
                answer_text=answer.answer_text,
                rationale=answer.rationale,
            )
        judge_says_abstain = score.score >= score.threshold
        if answer.abstained == judge_says_abstain:
            return None
        disagreement_type = (
            "answerer_abstains_judge_disagrees"
            if answer.abstained
            else "answerer_commits_judge_says_abstain"
        )
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

    Caller contract: pass the FULL set of disagreements for the run
    (not just newly-detected ones on a resume), since this function
    overwrites in mode ``"w"``. The batch runner walks all judged
    outputs to build that complete list — see
    :func:`run_judge_answers_batch`.

    Args:
        outputs_dir: The judge_answers stage's outputs dir.
        disagreements: All flagged rows for this run (cumulative).

    Returns:
        The audit file path when at least one disagreement was found;
        ``None`` when the list is empty (the file is also REMOVED if it
        exists from a prior run, so a stale audit doesn't persist past
        a re-judge that produced zero disagreements).
    """
    audit_path = outputs_dir / _AUDIT_FILENAME
    if not disagreements:
        if audit_path.exists():
            audit_path.unlink()
            logger.info(
                "No disagreements detected this run; removed stale audit at %s.",
                audit_path,
            )
        else:
            logger.info("No abstention disagreements detected; no audit log written.")
        return None
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        for row in disagreements:
            f.write(row.model_dump_json() + "\n")
    logger.info("Wrote %d abstention disagreement(s) to %s.", len(disagreements), audit_path)
    return audit_path
