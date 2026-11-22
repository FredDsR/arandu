"""Tests for the heuristic criteria (spec §6)."""

from __future__ import annotations

from arandu.shared.rag.judge_answers.heuristic import (
    AnswerabilityGateCriterion,
    CommitmentGateCriterion,
    SourceRecoveryCriterion,
)


class TestAnswerabilityGateCriterion:
    """Gate passes iff the item has a gold answer (``is_answerable``)."""

    def test_answerable_passes(self) -> None:
        score = AnswerabilityGateCriterion().evaluate(is_answerable=True)
        assert score.score == 1.0
        assert score.passed is True

    def test_nonanswerable_rejects(self) -> None:
        score = AnswerabilityGateCriterion().evaluate(is_answerable=False)
        assert score.score == 0.0
        assert score.passed is False

    def test_missing_kwarg_defaults_to_reject(self) -> None:
        score = AnswerabilityGateCriterion().evaluate()
        assert score.score == 0.0
        assert score.error is None

    def test_ignores_unrelated_kwargs(self) -> None:
        score = AnswerabilityGateCriterion().evaluate(is_answerable=True, abstained="true")
        assert score.passed is True


class TestCommitmentGateCriterion:
    """Gate passes iff the system committed (did not abstain)."""

    def test_committed_passes(self) -> None:
        score = CommitmentGateCriterion().evaluate(abstained="false")
        assert score.score == 1.0
        assert score.passed is True

    def test_abstained_rejects(self) -> None:
        score = CommitmentGateCriterion().evaluate(abstained="true")
        assert score.score == 0.0
        assert score.passed is False

    def test_abstained_string_case_insensitive(self) -> None:
        score = CommitmentGateCriterion().evaluate(abstained="True")
        assert score.passed is False

    def test_missing_kwarg_defaults_to_committed(self) -> None:
        # Default "false" -> committed; the producer always sets the
        # value explicitly, so this is just the safe default.
        score = CommitmentGateCriterion().evaluate()
        assert score.passed is True


class TestSourceRecoveryCriterion:
    """Containment of retrieved tokens in the CEP source context."""

    def test_full_containment_scores_one(self) -> None:
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="Maria mora em Itaqui",
            context="Maria mora em Itaqui na beira do rio",
            passages_are_non_prose=False,
        )
        assert score.score == 1.0

    def test_english_language_uses_english_tokenizer(self) -> None:
        # language="en" must build an English tokenizer; containment still
        # computes over English content words (smoke: no PT-tokenizer reuse).
        score = SourceRecoveryCriterion(language="en").evaluate(
            retrieved_text="Maria lives in Itaqui",
            context="Maria lives in Itaqui by the river",
            passages_are_non_prose=False,
        )
        assert score.score == 1.0

    def test_partial_containment(self) -> None:
        # Retrieved content tokens: {maria, morar, brasilia}; context has
        # maria + morar but not brasilia -> 2/3.
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="Maria mora em Brasilia",
            context="Maria mora em Itaqui",
            passages_are_non_prose=False,
        )
        assert score.score is not None
        assert 0.0 < score.score < 1.0

    def test_non_prose_payload_arm_scores_none(self) -> None:
        # Triples (non-prose payload) -> N/A: token overlap is structurally
        # near-zero regardless of relevance, so it would bias the cross-arm mean.
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="(Maria, mora_em, Itaqui)",
            context="Maria mora em Itaqui",
            passages_are_non_prose=True,
        )
        assert score.score is None
        assert score.error is None

    def test_prose_payload_arm_still_scores(self) -> None:
        # Inline-prose payload (khop_passage, payload_is_prose=True) sets
        # passages_are_non_prose=False, so containment scores normally — the
        # payload IS source prose, unlike triples.
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="Maria mora em Itaqui",
            context="Maria mora em Itaqui na beira do rio",
            passages_are_non_prose=False,
        )
        assert score.score == 1.0

    def test_empty_passages_scores_none(self) -> None:
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="", context="Maria mora em Itaqui", passages_are_non_prose=False
        )
        assert score.score is None

    def test_empty_context_scores_none(self) -> None:
        score = SourceRecoveryCriterion().evaluate(
            retrieved_text="Maria mora em Itaqui", context="", passages_are_non_prose=False
        )
        assert score.score is None
