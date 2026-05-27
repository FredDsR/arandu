"""Tests for the heuristic commitment gate (spec §6)."""

from __future__ import annotations

from arandu.shared.rag.judge_answers.heuristic import CommitmentGateCriterion


class TestCommitmentGateCriterion:
    """The gate passes only for a True-Commitment candidate."""

    def test_answerable_committed_passes(self) -> None:
        score = CommitmentGateCriterion().evaluate(is_answerable=True, abstained="false")
        assert score.score == 1.0
        assert score.passed is True

    def test_answerable_abstained_rejects(self) -> None:
        score = CommitmentGateCriterion().evaluate(is_answerable=True, abstained="true")
        assert score.score == 0.0
        assert score.passed is False

    def test_nonanswerable_committed_rejects(self) -> None:
        score = CommitmentGateCriterion().evaluate(is_answerable=False, abstained="false")
        assert score.score == 0.0
        assert score.passed is False

    def test_nonanswerable_abstained_rejects(self) -> None:
        score = CommitmentGateCriterion().evaluate(is_answerable=False, abstained="true")
        assert score.score == 0.0
        assert score.passed is False

    def test_abstained_string_case_insensitive(self) -> None:
        score = CommitmentGateCriterion().evaluate(is_answerable=True, abstained="True")
        assert score.passed is False

    def test_missing_kwargs_default_to_reject(self) -> None:
        # No kwargs -> is_answerable defaults False -> reject (never errors).
        score = CommitmentGateCriterion().evaluate()
        assert score.score == 0.0
        assert score.error is None
