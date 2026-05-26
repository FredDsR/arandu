"""Per-record TA/TC/FA/FC classification (spec §8.1).

The joint benchmark labels every record by crossing two binary signals:
the question's ground-truth ``is_answerable`` flag (set by the
retrieve stage from the CEP / non-answerable source) and a unified
"abstained" signal that combines the answerer's structured
``abstained`` flag with the abstention judge's verdict.

Four classes:

- **TA** (True Abstention): non-answerable + abstained → correct refusal.
- **FA** (False Abstention): answerable + abstained → over-cautious.
- **TC** (True Commitment): answerable + committed → quality scored by
  ``answer_correctness``.
- **FC** (False Commitment): non-answerable + committed → hallucination.

The unified abstention rule from spec §6.4:

    abstained_for_analysis = answer_record.abstained
                             and judge.abstention.score >= threshold

i.e. both signals must agree to classify a record as abstained. When
they disagree the record is treated as committed for the confusion
matrix — that's conservative for hallucination scoring, and the
disagreement is independently recorded by the judge_answers stage's
abstention audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import AnswerRecord


ConfusionLabel = Literal["TA", "TC", "FA", "FC", "unknown"]


_ABSTENTION_CRITERION = "abstention"


def classify_record(
    record: AnswerRecord, *, threshold_override: float | None = None
) -> ConfusionLabel:
    """Classify one judged :class:`AnswerRecord` into TA/TC/FA/FC.

    Args:
        record: An :class:`AnswerRecord` that has been judged. Its
            ``validation`` must be populated with the abstention
            criterion's score.
        threshold_override: Optional explicit τ_abstention. When
            ``None`` (default), the threshold is read from the
            criterion's own :attr:`CriterionScore.threshold` so a
            single source of truth governs both judge-stage flagging
            AND analysis-stage classification (mirrors the
            judge_answers audit).

    Returns:
        ``"TA"``, ``"TC"``, ``"FA"``, ``"FC"``, or ``"unknown"`` when
        the abstention judge couldn't run (validation absent or
        criterion errored). The analysis stage skips ``"unknown"``
        records from per-arm metric denominators and counts them in
        the report's ``unknown`` field instead.
    """
    abstention_score = _abstention_score(record)
    if abstention_score is None:
        return "unknown"
    threshold = threshold_override if threshold_override is not None else abstention_score[1]
    judge_says_abstain = abstention_score[0] >= threshold
    abstained_for_analysis = record.abstained and judge_says_abstain

    if record.is_answerable:
        return "FA" if abstained_for_analysis else "TC"
    return "TA" if abstained_for_analysis else "FC"


def _abstention_score(record: AnswerRecord) -> tuple[float, float] | None:
    """Return ``(score, threshold)`` from the abstention criterion, or ``None``.

    Returns ``None`` when:
    - ``record.validation`` is None (unjudged), OR
    - the abstention criterion didn't run (not in scores dict), OR
    - the criterion errored (its ``score`` field is ``None``).
    """
    if record.validation is None:
        return None
    for step in record.validation.stage_results.values():
        criterion_score = step.criterion_scores.get(_ABSTENTION_CRITERION)
        if criterion_score is None or criterion_score.score is None:
            continue
        return criterion_score.score, criterion_score.threshold
    return None
