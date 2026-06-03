"""Heuristic (non-LLM) criteria for the answer judge (spec §6).

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

This module also hosts :class:`SourceRecoveryCriterion`, a deterministic
*scoring* criterion (not a gate) that lives in the retrieval-scoring
stage alongside the LLM ``passage_coverage``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arandu.shared.judge.criterion import HeuristicCriterion
from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.rag.retrievers._bm25_tokenize import english_tokenizer, portuguese_tokenizer

if TYPE_CHECKING:
    from collections.abc import Callable


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


class SourceRecoveryCriterion(HeuristicCriterion):
    """Deterministic retrieval source-recovery (spec §6.3 lineage).

    Set-precision (containment) of the retrieved passages' content tokens
    within the CEP source ``context`` the QA pair was generated from:

        score = |tokens(retrieved) ∩ tokens(context)| / |tokens(retrieved)|

    This is a retrieval-*targeting* lens distinct from the LLM
    ``passage_coverage`` (which asks semantically whether the passages
    support the gold answer): did the retriever recover the actual source
    span? It runs in the retrieval-scoring stage, for every answerable
    item (TC + FA), and never gates.

    Containment, not symmetric F1, on purpose: the source ``context`` is a
    large generation-time chunk while a retrieved passage is a shorter
    span, so a perfect retriever returns a *subset* and symmetric overlap
    would be capped low by the length mismatch. Precision answers "is what
    I retrieved drawn from the right source?".

    Returns ``score=None`` (not ``0.0``) for the cases where the metric is
    undefined or would bias the cross-arm comparison, so the analysis mean
    excludes them:

    - **payload arms** (e.g. ``KHopTripleRetriever``): linearized triples
      are not source prose, so token overlap is structurally near-zero
      regardless of relevance — the same bias that kept the deterministic
      ``offset_coverage`` variant out of the pipeline.
    - **no retrieved passages** (e.g. ``NullRetriever``): empty denominator.
    - **no source context**: empty denominator on the gold side.

    Tokenization reuses the BM25 tokenizer for the run ``language`` (lemma
    + stopword removal), so containment is over content words, not surface
    forms or stopwords. The language must match the corpus, hence it is
    threaded through from the judge settings rather than hard-coded.
    """

    def __init__(self, *, language: str = "pt", threshold: float = 0.5) -> None:
        """Initialise the criterion and build the language tokenizer once.

        Args:
            language: ``"pt"`` (default) or ``"en"``; selects the BM25
                tokenizer. Any other value falls back to Portuguese (the
                project default corpus language).
            threshold: Pass threshold (advisory; this criterion runs in
                score mode and never gates).
        """
        super().__init__(name="source_recovery", threshold=threshold)
        tokenizer = english_tokenizer if language == "en" else portuguese_tokenizer
        self._tokenize: Callable[[str], list[str]] = tokenizer()

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Unused: :meth:`_evaluate_impl` is overridden to express the
        not-applicable (``score=None``) cases a ``(float, str)`` cannot."""
        raise NotImplementedError  # pragma: no cover

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Compute containment, or ``score=None`` for the undefined cases.

        Args:
            **kwargs: Reads ``passages_are_payload`` (bool),
                ``retrieved_text`` (raw joined passage text), and
                ``context`` (gold source span).

        Returns:
            CriterionScore with the containment fraction, or ``score=None``
            for payload arms / empty passages / empty context.
        """
        if bool(kwargs.get("passages_are_payload", False)):
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="payload arm (non-prose passages); source-recovery N/A",
            )
        retrieved_tokens = set(self._tokenize(str(kwargs.get("retrieved_text", ""))))
        context_tokens = set(self._tokenize(str(kwargs.get("context", ""))))
        if not retrieved_tokens:
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="no retrieved passages; source-recovery undefined",
            )
        if not context_tokens:
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="no source context; source-recovery undefined",
            )
        overlap = len(retrieved_tokens & context_tokens)
        containment = overlap / len(retrieved_tokens)
        return CriterionScore(
            score=containment,
            threshold=self.threshold,
            rationale=(
                f"{overlap}/{len(retrieved_tokens)} retrieved content tokens "
                f"found in source context"
            ),
        )
