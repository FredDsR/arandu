"""Tests for the heuristic gate criteria (spec §6)."""

from __future__ import annotations

from arandu.shared.rag.judge_answers.heuristic import (
    AnswerabilityGateCriterion,
    CommitmentGateCriterion,
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
