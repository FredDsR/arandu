"""Tests for ``AnswererOutput`` — locks the spec §5.2 consistency invariant."""

from __future__ import annotations

import pytest

from arandu.shared.rag.answer.schemas import AnswererOutput


class TestAnswererOutputConsistency:
    def test_abstained_with_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="abstained=True requires answer=None"):
            AnswererOutput(abstained=True, answer="something", rationale="r")

    def test_not_abstained_without_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="abstained=False requires a non-empty answer"):
            AnswererOutput(abstained=False, answer=None, rationale="r")

    def test_not_abstained_with_whitespace_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="abstained=False requires a non-empty answer"):
            AnswererOutput(abstained=False, answer="   \n  ", rationale="r")

    def test_abstained_with_null_answer_valid(self) -> None:
        out = AnswererOutput(abstained=True, answer=None, rationale="not enough evidence")
        assert out.abstained is True
        assert out.answer is None

    def test_not_abstained_with_text_answer_valid(self) -> None:
        out = AnswererOutput(abstained=False, answer="Maria mora em Itaqui.", rationale="P1")
        assert out.abstained is False
        assert out.answer == "Maria mora em Itaqui."

    def test_rationale_required(self) -> None:
        # Always populated per spec — even when answering.
        with pytest.raises(ValueError):
            AnswererOutput(abstained=False, answer="x", rationale="")
