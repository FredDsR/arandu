"""Heuristic (non-LLM) criteria for the answer judge (spec §6).

The commitment gate decides, deterministically and for free, whether the
gold-requiring criteria (``answer_correctness`` / ``answer_faithfulness``
/ ``passage_coverage``) should run at all. They only make sense for a
**True-Commitment** candidate: an *answerable* question that the system
actually *committed* to (did not abstain). For every other quadrant of
the analysis confusion matrix the gate rejects, short-circuiting the
gold-scoring stage:

- non-answerable + committed (FC, hallucination) — no gold answer exists;
- non-answerable + abstained (TA, correct refusal) — nothing to score;
- answerable + abstained (FA, over-cautious) — ``answer_text`` is None.

The ``abstention`` criterion runs in an earlier stage and is always
recorded, so the TA/TC/FA/FC signal is preserved regardless of the gate.
"""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.criterion import HeuristicCriterion


class CommitmentGateCriterion(HeuristicCriterion):
    """Pass only for a True-Commitment candidate (answerable + committed).

    Binary gate: ``1.0`` when the question is answerable AND the system
    committed (``abstained`` is false), else ``0.0``. Used in a
    ``filter``-mode stage so a ``0.0`` rejects the pipeline and skips the
    downstream gold-scoring criteria.

    Reads two kwargs forwarded by the batch runner:

    - ``is_answerable``: bool ground-truth answerability of the item.
    - ``abstained``: the system's structured flag as a lowercased string
      (``"true"`` / ``"false"``), the same value the abstention LLM prompt
      consumes.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Initialise the gate; default threshold makes it a binary 0/1 gate."""
        super().__init__(name="commitment_gate", threshold=threshold)

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Pass iff the item is answerable and the system committed.

        Args:
            **kwargs: Must contain ``is_answerable`` (bool) and
                ``abstained`` (``"true"``/``"false"`` string).

        Returns:
            ``(1.0, rationale)`` for a TC candidate, else ``(0.0, rationale)``.
        """
        is_answerable = bool(kwargs.get("is_answerable", False))
        abstained = str(kwargs.get("abstained", "false")).strip().lower() == "true"
        if is_answerable and not abstained:
            return 1.0, "answerable + committed (TC) -> score gold criteria"
        if not is_answerable:
            return 0.0, "non-answerable -> no gold; skip gold criteria"
        return 0.0, "answerable but abstained (FA) -> no answer text; skip gold criteria"
