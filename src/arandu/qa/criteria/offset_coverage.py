"""Deterministic offset-overlap variant of ``passage_coverage`` (spec §6.3).

Char-offset overlap fraction between the gold chunk's span and the union
of retrieved-passage spans, restricted to the same ``source_file_id``.
No LLM call — pure set arithmetic. Reported alongside the LLM variant
in ``arandu rag-analysis`` so the methodology rewrite can compare the
two signals.

Implementation note: cross-source-file overlap is **not** counted. A
retrieved passage from a different transcription doesn't cover the gold
chunk's offsets even if its character ranges happen to numerically
intersect — they're in different coordinate spaces. The guard at
``p.source_file_id == gold_chunk.source_file_id`` enforces this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arandu.shared.judge.criterion import HeuristicCriterion

if TYPE_CHECKING:
    from arandu.shared.chunking.schemas import Chunk


def offset_coverage(
    gold_chunk: Chunk,
    retrieved_passages: list[Chunk],
) -> float:
    """Char-offset overlap fraction between ``gold_chunk`` and retrieved passages.

    Args:
        gold_chunk: The CEP QA pair's source chunk (carries
            ``source_file_id``, ``start_char``, ``end_char``).
        retrieved_passages: Retrieved chunks from one retrieval arm.
            Only passages sharing ``gold_chunk.source_file_id``
            contribute to the overlap; cross-file overlap is dropped.

    Returns:
        Fraction in ``[0.0, 1.0]``. Returns 0.0 when ``gold_chunk`` has
        an empty span (defensive — shouldn't happen in practice but a
        ``ZeroDivisionError`` deep in benchmark analysis is worse than
        a clean zero).
    """
    gold_set = set(range(gold_chunk.start_char, gold_chunk.end_char))
    retrieved_set: set[int] = set()
    for p in retrieved_passages:
        if p.source_file_id == gold_chunk.source_file_id:
            retrieved_set |= set(range(p.start_char, p.end_char))
    if not gold_set:
        return 0.0
    return len(gold_set & retrieved_set) / len(gold_set)


class OffsetCoverageCriterion(HeuristicCriterion):
    """``passage_coverage`` (deterministic variant) as a :class:`HeuristicCriterion`.

    Pairs with the LLM variant of the same name loaded from
    ``prompts/judge/criteria/passage_coverage/``; both verdicts persist
    on the same ``AnswerRecord`` so analysis can report both signals
    side-by-side (the LLM variant captures semantic coverage, this one
    captures literal token-overlap — together they triangulate "did
    retrieval reach the right region of source text").
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Construct with the spec's default threshold (0.5).

        Args:
            threshold: Pass threshold. The criterion runs in ``score``
                mode (pipeline never filters on it), so the threshold
                is informational — recorded in the verdict for downstream
                cross-arm cutoffs.
        """
        super().__init__(name="offset_coverage", threshold=threshold)

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Compute :func:`offset_coverage` from the kwargs.

        Args:
            gold_chunk: :class:`Chunk` for the CEP pair's source segment.
            retrieved_passages: ``list[Chunk]`` for one retrieval result.

        Returns:
            ``(score, rationale)``. Rationale is a brief diagnostic so
            the verdict carries human-readable context alongside the
            number — useful when the audit pass spot-checks low scores.
        """
        gold_chunk = kwargs["gold_chunk"]
        retrieved_passages = kwargs["retrieved_passages"]
        score = offset_coverage(gold_chunk, retrieved_passages)
        in_file = sum(
            1 for p in retrieved_passages if p.source_file_id == gold_chunk.source_file_id
        )
        rationale = (
            f"offset_coverage={score:.3f} "
            f"({in_file}/{len(retrieved_passages)} retrieved passages share "
            f"source_file_id={gold_chunk.source_file_id!r})"
        )
        return score, rationale
