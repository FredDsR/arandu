"""Heuristic (non-LLM) gate criteria for the answer judge (spec §6).

Two single-responsibility gates compose into a cascade that gates the
LLM-backed scoring stages on their actual data dependencies:

- :class:`AnswerabilityGateCriterion` passes iff the item is answerable
  (has a gold answer). Used in front of *retrieval-quality* scoring
  (``passage_coverage``) and again, transitively, in front of *answer*
  scoring. A reject leaves only the always-recorded ``abstention``
  signal in the result.
- :class:`CommitmentGateCriterion` passes iff the system committed
  (did not abstain). Used in front of *answer* scoring
  (``answer_correctness`` / ``answer_faithfulness``), which need the
  committed answer text. Lives downstream of the answerability gate, so
  reaching it implies the item is already answerable.

Together they partition the analysis confusion matrix without any LLM
call: TA / FC stop at the answerability gate (skip retrieval + answer
scoring); FA passes the first gate but is stopped at the commitment
gate (skip answer scoring only; passage_coverage still runs because
retrieval can be evaluated against the gold answer regardless of
whether the system used it); TC passes both gates.
"""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion


class AnswerabilityGateCriterion(HeuristicCriterion):
    """Pass iff the item is answerable (a gold answer exists).

    Reads a single kwarg, ``is_answerable`` (bool), forwarded by the
    batch runner from :attr:`AnswerRecord.is_answerable`. Used in a
    ``filter``-mode stage so a non-answerable item short-circuits every
    later gold-using criterion (retrieval-quality + answer-quality).
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Initialise the gate; default threshold makes it a binary 0/1 gate."""
        super().__init__(name="answerability_gate", threshold=threshold)

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Pass iff ``is_answerable`` is true.

        Args:
            **kwargs: Must contain ``is_answerable`` (bool).

        Returns:
            ``(1.0, rationale)`` when answerable, else ``(0.0, rationale)``.
        """
        if bool(kwargs.get("is_answerable", False)):
            return 1.0, "answerable -> run retrieval + (if committed) answer scoring"
        return 0.0, "non-answerable -> no gold; only abstention is meaningful"


class CommitmentGateCriterion(HeuristicCriterion):
    """Pass iff the system committed (did not abstain).

    Reads the ``abstained`` kwarg as a lowercased string
    (``"true"`` / ``"false"``), the same value the abstention LLM prompt
    consumes. Used in a ``filter``-mode stage downstream of
    :class:`AnswerabilityGateCriterion`, so reaching it implies the
    item is already answerable; a reject here skips only the answer-text
    scoring stage (correctness + faithfulness), leaving any prior
    retrieval-quality score intact.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Initialise the gate; default threshold makes it a binary 0/1 gate."""
        super().__init__(name="commitment_gate", threshold=threshold)

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Pass iff ``abstained`` is the lowercased string ``"false"``.

        Args:
            **kwargs: Must contain ``abstained`` (``"true"``/``"false"``).

        Returns:
            ``(1.0, rationale)`` when committed, else ``(0.0, rationale)``.
        """
        abstained = str(kwargs.get("abstained", "false")).strip().lower() == "true"
        if not abstained:
            return 1.0, "committed -> score answer correctness + faithfulness"
        return 0.0, "abstained -> no answer text; skip answer scoring"
